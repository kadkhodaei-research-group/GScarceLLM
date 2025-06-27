# Preprocess the data to extract gas flow rates from a string column and return a DataFrame with separate columns.

import pandas as pd
import numpy as np
import openai
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
import ast
import sys, pathlib
sys.path.append(pathlib.Path(__file__).parent / "external" / "metalhydride")


def preprocess_data(data):
    """
    Extracts gas flow rates from a string column and returns a DataFrame 
    with separate columns for each allowed gas.
    """
    gas_dicts = data['Gases {Gas: Flow rate (sccm)}'].apply(extract_gases)
    flow_rate_columns = gas_dicts.apply(extract_flow_rates)
    data_with_flow_rates = pd.concat([data, flow_rate_columns], axis=1)
    data_with_flow_rates.drop(columns=['Gases {Gas: Flow rate (sccm)}'], inplace=True)
    return data_with_flow_rates

def extract_gases(gas_str):
    gas_dict = {}
    if pd.notna(gas_str):
        gas_str = gas_str.strip('{}')
        pairs = gas_str.split(',')
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':')
                key = key.strip().strip("'")
                try:
                    value = float(value.strip())
                except ValueError:
                    continue
                gas_dict[key] = gas_dict.get(key, 0) + value
            else:
                continue
    return gas_dict

def extract_flow_rates(gas_dict):
    allowed_gases = {'CH4', 'H2', 'Ar', 'C2H2', 'C2H4'}
    flow_rates = {}
    for gas, flow_rate in gas_dict.items():
        if gas in allowed_gases:
            flow_rates[gas] = flow_rate
    return pd.Series(flow_rates)

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def embed_substrate(data, column, model="text-embedding-3-small", reduced_dim=4):
    embeddings = []
    # Generate embeddings for each substrate
    for substrate in data[column].astype(str):
        response = openai.Embedding.create(
            model=model,
            input=substrate,
            encoding_format="float",
            dimensions=reduced_dim  # Reduced dimension
        )
        embedding = response['data'][0]['embedding']  # Shortened embedding
        normalized_embedding = normalize_l2(embedding)  # Normalize
        embeddings.append(normalized_embedding)
    # Create new column names for the reduced embedding
    reduced_embedding_columns = [f'{column}_dim_{i+1}' for i in range(reduced_dim)]
    reduced_embedding_df = pd.DataFrame(embeddings, index=data.index, columns=reduced_embedding_columns)
    # Concatenate embeddings to the original dataframe
    data = pd.concat([data, reduced_embedding_df], axis=1)
    return data

def fill_missing_values(data, columns_to_impute, knn_neighbors=5):
    """
    Impute missing values in a dataset, handling numeric and non-numeric columns separately.
    """
    # Separate numeric and non-numeric columns
    numeric_columns = data[columns_to_impute].select_dtypes(include=[np.number]).columns
    non_numeric_columns = data[columns_to_impute].select_dtypes(exclude=[np.number]).columns
    # Impute numeric columns using KNNImputer
    if len(numeric_columns) > 0:
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    # Handle non-numeric columns by filling with the mode or a placeholder
    for column in non_numeric_columns:
        most_frequent = data[column].mode()[0] if not data[column].mode().empty else "Missing"
        data[column].fillna(most_frequent, inplace=True)
    return data

def normalize_attributes(data, columns_to_normalize):
    """
    Normalize numeric attributes using MinMaxScaler.
    """
    # Ensure the columns to normalize are numeric
    numeric_columns = data[columns_to_normalize].select_dtypes(include=[np.number]).columns
    non_numeric_columns = set(columns_to_normalize) - set(numeric_columns)
    if non_numeric_columns:
        print(f"Skipping non-numeric columns: {non_numeric_columns}")
    data[numeric_columns] = data[numeric_columns].fillna(0)
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def call_chatgpt(prompt, temperature=0.8):
    print("Executing API Call...")
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are an expert in data featurization"},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def featurize_substrate_llm(data, column):
    """
    Featurize the substrate column using an LLM and store raw responses.
    """
    llm_responses = []
    # Loop through each substrate to get LLM response
    for idx, substrate in data[column].astype(str).items():
        prompt = f"""
        Featurize the following substrate description: '{substrate}'.
        Return the output in a structured format using only a Python dictionary with these keys:
        - 'main_material': Main material used in the substrate (e.g., 'Cu', 'Ni', 'SiO2')
        - 'is_alloy': 1 if it's an alloy, 0 otherwise
        - 'substrate_type': Type of substrate (e.g., 'metal', 'semiconductor')
        - 'surface_coating': 1 if there's a surface coating, 0 otherwise
        - 'coating_material': Coating material if available, 'None' if not
        - 'crystallographic_orientation': Crystallographic orientation if available, 'Unknown' if not
        - 'is_multilayer': 1 if it's a multilayer substrate, 0 otherwise
        Populate with the most probable values if some data is missing.
        """
        try:
            llm_response = call_chatgpt(prompt, temperature=0.8)
            llm_responses.append(llm_response)
            print(f"LLM Response for index {idx} and substrate '{substrate}': {llm_response}")
        except Exception as e:
            print(f"Error getting LLM response for index {idx} and substrate '{substrate}': {str(e)}")
            llm_responses.append(f"Error: {str(e)}")
    data['LLM_Substrate_Features'] = llm_responses
    return data

def encode_graphene_layers(data, n_classes=2):
    """Encode 'No. of Graphene Layers' as a categorical variable."""
    data['No. of Graphene Layers'] = data['No. of Graphene Layers'].replace('ML', 10.0)
    data['No. of Graphene Layers'] = pd.to_numeric(data['No. of Graphene Layers'], errors='coerce')
    data['No. of Graphene Layers'] = data['No. of Graphene Layers'].fillna(0)
    if n_classes == 2:
        data['No. of Graphene Layers'] = data['No. of Graphene Layers'].apply(lambda x: 0 if x <= 1 else 1)
    elif n_classes == 3:
        data['No. of Graphene Layers'] = data['No. of Graphene Layers'].apply(lambda x: 0 if x < 1.5
                                                                             else 1 if 1.5 <= x < 2.5
                                                                             else 2)
    elif n_classes == 4:
        data['No. of Graphene Layers'] = data['No. of Graphene Layers'].apply(lambda x: 0 if x < 1.5
                                                                             else 1 if 1.5 <= x < 2.5
                                                                             else 2 if 2.5 <= x < 3.5
                                                                             else 3)
    data['No. of Graphene Layers'] = data['No. of Graphene Layers'].astype('category')
    return data

def process_dataset(data_path, n_classes, featurization_method='embedding'):
    # Load the dataset
    data = pd.read_csv(data_path)
    if 'CVD Method' in data.columns:
        data = data.drop(columns=['CVD Method'])
    # Define columns for imputation and normalization
    columns_to_impute = ['Pressure (mbar)', 'Temperature (°C)', 'Growth Time (min)',
                         'No. of Graphene Layers', 'H2', 'CH4', 'C2H4', 'Ar', 'C2H2']
    columns_to_normalize = ['Pressure (mbar)', 'Temperature (°C)', 'Growth Time (min)',
                            'H2', 'CH4', 'C2H4', 'Ar', 'C2H2']
    # Encode graphene layers
    data = encode_graphene_layers(data, n_classes)
    # Impute missing values and normalize numeric attributes
    data = fill_missing_values(data, columns_to_impute)
    data = normalize_attributes(data, columns_to_normalize)
    # Apply featurization to the substrate column
    if featurization_method == 'embedding':
        data = embed_substrate(data, column='Substrate', reduced_dim=4)
    elif featurization_method == 'llm':
        data = featurize_substrate_llm(data, column='Substrate')
    elif featurization_method == 'label_encoding':
        label_encoder = LabelEncoder()
        data['Substrate'] = label_encoder.fit_transform(data['Substrate'])
    elif featurization_method == 'parsed':
        data = featurize_substrate_llm(data, column='Substrate')
        # Parse the LLM_Substrate_Features column inline
        new_columns = {
            'main_material': [],
            'is_alloy': [],
            'substrate_type': [],
            'surface_coating': [],
            'coating_material': [],
            'crystallographic_orientation': [],
            'is_multilayer': []
        }
        # Note: The following loop uses the 'Substrate' column instead of 'LLM_Substrate_Features'
        # Adjust as needed.
        for response in data['Substrate']:
            try:
                response_dict = ast.literal_eval(response.strip("```python").strip("```"))
                for key in new_columns.keys():
                    new_columns[key].append(response_dict.get(key, 'Unknown'))
            except (SyntaxError, ValueError):
                for key in new_columns.keys():
                    new_columns[key].append('Error')
        for key, values in new_columns.items():
            data[key] = values
    elif featurization_method == 'none':
        pass
    output_file = f'Data/processed_dataset_with_{featurization_method}.csv'
    data.to_csv(output_file, index=False)
    print(f"Processing complete. Data saved to '{output_file}'.")
    return data