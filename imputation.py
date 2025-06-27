import pandas as pd
from utils import call_chatgpt_4o
from featurization import preprocess_data
from utils import load_graphene_data
import sys, pathlib
sys.path.append(pathlib.Path(__file__).parent / "external" / "metalhydride")

raw_data = load_graphene_data()
data_preprocessed = preprocess_data(raw_data)

def generate_prompt(row, target_variable, analysis_completed=False, prompt_type="guide"):
    """
    Generate a prompt for ChatGPT using the context of the current row.
    """
    context_vars = {
        "cvd_method": row['CVD Method'],
        "pressure": row['Pressure (mbar)'] if target_variable != "Pressure (mbar)" and not pd.isnull(row['Pressure (mbar)']) else 'unknown',
        "temperature": row['Temperature (°C)'] if target_variable != "Temperature (°C)" and not pd.isnull(row['Temperature (°C)']) else 'unknown',
        "growth_time": row['Growth Time (min)'] if target_variable != "Growth Time (min)" and not pd.isnull(row['Growth Time (min)']) else 'unknown',
        "substrate": row['Substrate'],
        "h2": row['H2'] if target_variable != "H2" and not pd.isnull(row['H2']) else 'unknown',
        "c2h4": row['C2H4'] if target_variable != "C2H4" and not pd.isnull(row['C2H4']) else 'unknown',
        "ar": row['Ar'] if target_variable != "Ar" and not pd.isnull(row['Ar']) else 'unknown',
        "c2h2": row['C2H2'] if target_variable != "C2H2" and not pd.isnull(row['C2H2']) else 'unknown',
        "ch4": row['CH4'] if target_variable != "CH4" and not pd.isnull(row['CH4']) else 'unknown'
    }

    # Helper function to identify descriptors
    def is_descriptor(value):
        return isinstance(value, str)

    # GIST FORMALISM CONTEXT VARIABLES (used only for the "gist" prompt type)
    context_vars_gist = {
        "cvd_method": row['CVD Method'],
        "pressure": row['Pressure (mbar)'],
        "temperature": row['Temperature (°C)'],
        "growth_time": row['Growth Time (min)'],
        "substrate": row['Substrate'],
        "h2": row['H2'],
        "c2h4": row['C2H4'],
        "ar": row['Ar'],
        "c2h2": row['C2H2'],
        "ch4": row['CH4']
    }

    analysis_message = "The analysis of the known ground truth values has been completed." if analysis_completed else ""

    guide = (
        f"System: You are a CVD expert with access to experimental and/or simulated data from the literature, "
        f"which you can understand to derive correlations between known and unknown quantities for given datasets. "
        f"Now, I would like you to help me impute the missing value for {target_variable}. Please use your scientific knowledge to impute this value, "
        f"based only on the current row’s context. Note that the values provided are not normalized, and the imputation should be done using your understanding of CVD processes.\n"
        f"User: For the CVD process '{context_vars['cvd_method']}', with the following parameters:\n"
        f"Pressure: {context_vars['pressure']} mbar, Temperature: {context_vars['temperature']} °C, "
        f"Growth Time: {context_vars['growth_time']} min, Substrate: {context_vars['substrate']}, "
        f"Gas Flow Rates: H2: {context_vars['h2']}, C2H4: {context_vars['c2h4']}, Ar: {context_vars['ar']}, "
        f"C2H2: {context_vars['c2h2']}. Can you determine the missing value of {target_variable}? "
        f"Please provide only the missing value as a single number without any additional text."
    )

    map = (
        f"System: You are a CVD expert with access to experimental and/or simulated data from literature, which you can understand to derive correlations between known and unknown quantities for given datasets. "
        f"Now, I would like you to help me impute the missing value for {target_variable}. Please use your scientific knowledge to impute this value based only on the current row's context.\n"
        f"Additionally, ensure that the imputation for {target_variable} follows the original data distribution, as observed in the dataset for each parameter. Here are the key characteristics of the distributions:\n"
        f"• C2H4: Almost all values are exactly 30, indicating a single known value without much variability.\n"
        f"• Pressure (mbar): Skewed right, with the majority of the data close to 0 and a second small peak around 1000 mbar.\n"
        f"• Growth Time (min): Skewed right with most values concentrated below 100 min, though there are a few outliers up to 600 min.\n"
        f"• H2: Highly right-skewed, with most values near zero and a few extending to higher values.\n"
        f"• CH4: Right-skewed, with most values concentrated near 0.\n"
        f"• Ar: Very right-skewed, with most values concentrated near zero and very few extending up to 10,000.\n"
        f"• C2H2: Multimodal, with peaks near 0, 10, and 30.\n"
        f"Note that the values provided are not normalized, and the imputation should be done using your understanding of CVD processes and the data distribution patterns observed above.\n"
        f"User: For the CVD process '{context_vars['cvd_method']}', with the following parameters:\n"
        f"• Pressure: {context_vars['pressure']} mbar\n"
        f"• Temperature: {context_vars['temperature']} °C\n"
        f"• Growth Time: {context_vars['growth_time']} min\n"
        f"• Substrate: {context_vars['substrate']}\n"
        f"• Gas Flow Rates: H2: {context_vars['h2']}, C2H4: {context_vars['c2h4']}, Ar: {context_vars['ar']}, C2H2: {context_vars['c2h2']}\n"
        f"Can you determine the missing value of {target_variable}? Please provide only the missing value as a single number without any additional text."
    )

    mapv = (
        f"System: You are a CVD expert with access to experimental and/or simulated data from literature, which you can understand to derive correlations between known and unknown quantities for given datasets. "
        f"Now, I would like you to help me impute the missing value for {target_variable}. Please use your scientific knowledge to impute this value, based only on the current row’s context, "
        f"and ensure that the imputation follows the original data distribution while introducing controlled variability.\n"
        f"The original data distributions for each attribute are as follows:\n"
        f"• C2H4: Tightly concentrated around 30 with minimal variability.\n"
        f"• Pressure (mbar): Right-skewed, with most values at low pressures and a secondary peak around 1000 mbar.\n"
        f"• Growth Time (min): Right-skewed, with most values below 100 minutes and some outliers extending to 600 minutes.\n"
        f"• H2: Right-skewed, with values concentrated near 0, but with a long tail up to 1400.\n"
        f"• CH4: Right-skewed, with most values near 0 and a few outliers extending up to 500.\n"
        f"• Ar: Highly right-skewed, with most values near 0 and some outliers extending up to 10,000.\n"
        f"• C2H2: Multimodal, with distinct peaks around 0, 10, and 30.\n"
        f"In order to introduce realistic variability and avoid repetition of imputed values, a small amount of random noise should be applied uniformly to all attributes during imputation. "
        f"This noise should be small enough to maintain the general data distribution but provide natural variations, ensuring that no extreme outliers are introduced. "
        f"The noise should be probabilistically scaled based on each attribute’s distribution, with skewed distributions respecting their inherent shape while still introducing variability.\n"
        f"User: For the CVD process '{context_vars['cvd_method']}', with the following parameters:\n"
        f"• Pressure: {context_vars['pressure']} mbar\n"
        f"• Temperature: {context_vars['temperature']} °C\n"
        f"• Growth Time: {context_vars['growth_time']} min\n"
        f"• Substrate: {context_vars['substrate']}\n"
        f"• Gas Flow Rates: H2: {context_vars['h2']}, C2H4: {context_vars['c2h4']}, Ar: {context_vars['ar']}, C2H2: {context_vars['c2h2']}\n"
        f"Can you determine the missing value of {target_variable}? Please provide only the missing value as a single number, without any additional text."
    )

    cite = (
        f"System: You are a CVD expert with access to experimental and/or simulated data from literature, which you can understand to derive correlations between known and unknown quantities for given datasets. "
        f"{analysis_message} Now, I would like you to help me impute the missing value for {target_variable}. Please use your scientific knowledge to impute this value, based only on the current row’s context. "
        f"Note that the values provided are not normalized, and the imputation should be done using your understanding of CVD processes.\n"
        f"User: For the CVD process '{context_vars['cvd_method']}', with the following parameters:\n"
        f"• Pressure: {context_vars['pressure']} mbar\n"
        f"• Temperature: {context_vars['temperature']} °C\n"
        f"• Growth Time: {context_vars['growth_time']} min\n"
        f"• Substrate: {context_vars['substrate']}\n"
        f"• Gas Flow Rates:\n"
        f"  H2: {context_vars['h2']}\n"
        f"  C2H4: {context_vars['c2h4']}\n"
        f"  Ar: {context_vars['ar']}\n"
        f"  C2H2: {context_vars['c2h2']}\n"
        f"Can you determine the missing value of {target_variable}? Please provide only the missing value as a single number without any additional text.\n"
        f"Here’s the current attribute data, which you need to impute based on this knowledge and your knowledge of the literature: {data_preprocessed[target_variable].to_string()}"
    )

    citev = (
        f"System: You are a CVD expert with access to experimental and/or simulated data from literature, which you can understand to derive correlations between known and unknown quantities for given datasets. {analysis_message} Now, I would like you to help me impute the missing value for {target_variable}. Please use your scientific knowledge to impute this value, based only on the current row’s context. Note that the values provided are not normalized, and the imputation should be done using your understanding of CVD processes.\n"
        f"User: For the CVD process '{context_vars['cvd_method']}', with the following parameters:\n"
        f"• Pressure: {context_vars['pressure']} mbar\n"
        f"• Temperature: {context_vars['temperature']} °C\n"
        f"• Growth Time: {context_vars['growth_time']} min\n"
        f"• Substrate: {context_vars['substrate']}\n"
        f"• Gas Flow Rates:\n"
        f"  H2: {context_vars['h2']}\n"
        f"  C2H4: {context_vars['c2h4']}\n"
        f"  Ar: {context_vars['ar']}\n"
        f"  C2H2: {context_vars['c2h2']}\n"
        f"Can you determine the missing value of {target_variable}? Please provide only the missing value as a single number, without any additional text.\n"
        f"Here’s the current attribute data, you need to impute based on this knowledge and your knowledge of the literature: Also make sure to impute values around the modes and also in regions of scarely populated data, to have a diverse data distribution {data_preprocessed[target_variable].dropna().iloc[:len(data_preprocessed[target_variable].dropna()) // 2].to_string()}"
    )

    gist = (
        f"System: You are a CVD expert with extensive knowledge of experimental and simulated data from the literature. "
        f"Now, I would like you to help me impute the missing value for {target_variable}. "
        f"Please use your scientific knowledge, along with the context provided by the descriptors for missing values, to make an informed imputation."
        f"\n\nUser: For the CVD process '{context_vars_gist['cvd_method']}', the parameters are as follows:\n"
        f"- Pressure: {context_vars_gist['pressure']} mbar\n"
        f"- Temperature: {context_vars_gist['temperature']} °C\n"
        f"- Growth Time: {context_vars_gist['growth_time']} min\n"
        f"- Substrate: {context_vars_gist['substrate']}\n"
        f"- Gas Flow Rates: H2: {context_vars_gist['h2']}, C2H4: {context_vars_gist['c2h4']}, Ar: {context_vars_gist['ar']}, C2H2: {context_vars_gist['c2h2']}, CH4: {context_vars_gist['ch4']}\n"
        f"\nThe dataset has been enhanced with contextually relevant descriptors to handle missing values effectively. These descriptors provide "
        f"additional insight based on the original data patterns and CVD processes. Impute the missing value for {target_variable} considering this context."
        f"\n\nPlease provide only the missing value as a single number, without any additional text. Ensure that the imputed value aligns with the original data distribution and introduces controlled variability to maintain realistic data diversity."
    )

    if prompt_type == "guide":
        return guide
    elif prompt_type == "map":
        return map
    elif prompt_type == "mapv":
        return mapv
    elif prompt_type == "cite":
        return cite
    elif prompt_type == "citev":
        return citev
    elif prompt_type == "gist":
        return gist
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Please use 'guide', 'map', 'mapv', 'cite', 'citev' or 'gist'.")


def impute_target_gpt_4o(row, target_variable, temperature, max_iterations=6, prompt_type="guide"):
    """
    Impute the target variable using ChatGPT with a feedback loop for known values.
    It iterates up to 6 times, adjusting the prompt based on the deviation from the ground truth.
    """
    ground_truth = row[target_variable]
    prompt = generate_prompt(row, target_variable, analysis_completed=False, prompt_type=prompt_type)

    deviations = []
    predictions = []
    deviation = float('inf')

    for iteration in range(max_iterations):
        imputed_value = call_chatgpt_4o(prompt, temperature)

        try:
            imputed_value = float(imputed_value.strip())
            predictions.append(imputed_value)
            deviation = abs(imputed_value - ground_truth)
            deviations.append(deviation)
        except (ValueError, TypeError):
            continue

        # Stop if the deviation is within 10% of the ground truth
        if deviation <= 0.1 * ground_truth:
            break

        # Update the prompt based on the deviation to encourage refinement
        prompt = (
            f"System: The deviation from the ground truth value was {deviation:.2f}. "
            f"Great job if it's lower than the previous prediction! "
            "Please refine the prediction if needed.\n" + prompt
        )

    return deviations, predictions, ground_truth


def impute_all_attributes(data, attributes_to_impute, temperature=0.8, retry_limit=3, perform_analysis=True, gist_formalism=False, prompt_type="guide"):
    """
    Perform imputation on a dataset with options for ground truth analysis or direct imputation.
    If gist_formalism=True and perform_analysis=True, descriptors for ground truth values are created using apply_gist_formalism.
    """
    ground_truth_predictions = {}  # To store predictions for ground truth analysis (if performed)
    imputed_data = data.copy()       # This will hold the imputed dataset

    # Helper function to identify descriptors
    def is_descriptor(value):
        return isinstance(value, str)

    for target_variable in attributes_to_impute:
        print(f"\nStarting imputation for {target_variable}...")

        # GIST Formalism: Handle ground truth analysis by creating descriptors if needed
        if gist_formalism and perform_analysis:
            print(f"\nCreating descriptors for ground truth values for {target_variable}...")
            ground_truth_rows = imputed_data[imputed_data[target_variable].notnull()]

            # Apply `apply_gist_formalism` row-wise to ensure each row has descriptors
            ground_truth_with_descriptors = ground_truth_rows.apply(
                lambda row: apply_gist_formalism(row.to_frame().T, [target_variable]).squeeze(), axis=1
            )

            predictions_for_attribute = []  # To store predictions for each ground truth row with descriptors

            # Impute on generated descriptors for ground truth analysis
            for index, row in ground_truth_with_descriptors.iterrows():
                deviations, row_predictions, ground_truth = impute_target_gpt_4o(row, target_variable, temperature, max_iterations=6, prompt_type=prompt_type)
                predictions_for_attribute.append(row_predictions)  # Store all 6 predictions for this row
                print(f"Row {index} predictions for {target_variable}: {row_predictions}")

            ground_truth_predictions[target_variable] = predictions_for_attribute
            print(f"Completed imputation analysis on descriptors for ground truth for {target_variable}.")

        # Regular Analysis or Direct Imputation Mode
        elif not gist_formalism and perform_analysis:
            print(f"\nImputation on removed ground truth for {target_variable}...")
            ground_truth_rows = data[data[target_variable].notnull()]
            predictions_for_attribute = []  # To store 6 predictions for each ground truth row

            for index, row in ground_truth_rows.iterrows():
                deviations, row_predictions, ground_truth = impute_target_gpt_4o(row, target_variable, temperature, max_iterations=6, prompt_type=prompt_type)
                predictions_for_attribute.append(row_predictions)
                print(f"Row {index} predictions for {target_variable}: {row_predictions}")

            ground_truth_predictions[target_variable] = predictions_for_attribute
            print(f"Completed imputation on removed ground truth for {target_variable}.")

        # Impute descriptors or missing values directly
        print(f"\nImputation of {'descriptors' if gist_formalism else 'missing values'} for {target_variable}...")
        retries = 0  # Track number of retries

        impute_condition = imputed_data[target_variable].apply(is_descriptor) if gist_formalism else imputed_data[target_variable].isnull()

        while impute_condition.sum() > 0 and retries < retry_limit:
            print(f"Retry #{retries + 1} for {target_variable}...")

            indices_to_impute = imputed_data[impute_condition].index
            for index in indices_to_impute:
                row = data.loc[index]
                imputed_value = call_chatgpt_4o(generate_prompt(row, target_variable, analysis_completed=False, prompt_type=prompt_type), temperature)
                print(f"Imputed value for row {index} in {target_variable}: {imputed_value}")

                try:
                    imputed_value = float(imputed_value.strip())
                    imputed_data.at[index, target_variable] = imputed_value  # Impute directly
                except (ValueError, TypeError):
                    print(f"Warning: Imputed value for row {index} in {target_variable} is NaN.")

            retries += 1
            impute_condition = imputed_data[target_variable].apply(is_descriptor) if gist_formalism else imputed_data[target_variable].isnull()
            remaining_to_impute = impute_condition.sum()
            print(f"Remaining values for {target_variable}: {remaining_to_impute}")

            if remaining_to_impute == 0:
                print(f"All {'descriptor' if gist_formalism else 'missing'} values for {target_variable} have been imputed.")
                break
        else:
            if impute_condition.sum() > 0:
                print(f"Warning: {impute_condition.sum()} values remain for {target_variable} after {retry_limit} retries.")

    return imputed_data, ground_truth_predictions if perform_analysis else None


def gist_formalism(row, target_variable):
    """
    Generates a contextualized descriptor for a specific missing attribute in a row using the pre-trained LLM (ChatGPT),
    and stores it in place of the missing value.
    """
    context_description = (
        f"The dataset represents CVD process parameters. "
        f"The CVD method used is '{row['CVD Method']}'. "
        f"Key parameters are:\n"
        f"- Pressure: {row['Pressure (mbar)'] if target_variable != 'Pressure (mbar)' and not pd.isnull(row['Pressure (mbar)']) else 'missing'} mbar\n"
        f"- Temperature: {row['Temperature (°C)'] if target_variable != 'Temperature (°C)' and not pd.isnull(row['Temperature (°C)']) else 'missing'} °C\n"
        f"- Growth Time: {row['Growth Time (min)'] if target_variable != 'Growth Time (min)' and not pd.isnull(row['Growth Time (min)']) else 'missing'} min\n"
        f"- Substrate: {row['Substrate']}\n"
        f"- Gas Flow Rates: H2: {row['H2'] if target_variable != 'H2' and not pd.isnull(row['H2']) else 'missing'}, "
        f"C2H4: {row['C2H4'] if target_variable != 'C2H4' and not pd.isnull(row['C2H4']) else 'missing'}, "
        f"Ar: {row['Ar'] if target_variable != 'Ar' and not pd.isnull(row['Ar']) else 'missing'}, "
        f"C2H2: {row['C2H2'] if target_variable != 'C2H2' and not pd.isnull(row['C2H2']) else 'missing'}, "
        f"CH4: {row['CH4'] if target_variable != 'CH4' and not pd.isnull(row['CH4']) else 'missing'}."
    )

    prompt = (
        f"{context_description}\n\n"
        f"The value for '{target_variable}' is missing. "
        "Please provide a contextually relevant descriptor that can be used as a substitute for this missing value, based on scientific knowledge of CVD processes. "
        "Please provide only a few words (4 to 5) description for the descriptor with an example range that's not too wide, without any additional text."
    )

    print(f"\nGenerating descriptor for missing '{target_variable}' in row with index {row.name}...")
    descriptor = call_chatgpt_4o(prompt, temperature=0.8)
    print(f"Descriptor generated for '{target_variable}': {descriptor}")
    row[target_variable] = descriptor
    return row


def apply_gist_formalism(data, attributes_to_impute):
    """
    Applies GIST formalism to the entire dataset by filling in each missing value with a contextually relevant descriptor.
    """
    print("\nApplying GIST formalism to dataset...\n")
    for index, row in data.iterrows():
        for target_variable in attributes_to_impute:
            if pd.isnull(row[target_variable]):
                data.loc[index] = gist_formalism(row, target_variable)
    print("\nGIST formalism applied. Preview of the dataset with descriptors:")
    print(data.head())
    return data