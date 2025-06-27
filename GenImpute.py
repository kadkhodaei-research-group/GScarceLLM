import pandas as pd
import sys
import pathlib

# ---- YOUR CUSTOM IMPORTS (leave these lines as they are) ----
from utils import call_chatgpt_4o
from featurization import preprocess_data
from utils import load_graphene_data

# If running standalone and need metalhydride support (leave as in your old script)
sys.path.append(pathlib.Path(__file__).parent / "external" / "metalhydride")

# ---- GENERALIZED PROMPT GENERATOR ----
def generate_generalized_prompt(
    row,
    context_columns,
    target_variable,
    prompt_type="guide",
    study_type="scientific experiment",
    analysis_completed=False,
    data_distribution_info=None,
    additional_context=None
):
    context_strs = []
    for col in context_columns:
        val = row[col]
        val_str = str(val) if not pd.isnull(val) and col != target_variable else "unknown"
        context_strs.append(f"{col}: {val_str}")
    context_block = ", ".join(context_strs)

    dist_block = ""
    if data_distribution_info and prompt_type in {"map", "mapv"}:
        dist_block = "\n".join(
            [f"• {col}: {desc}" for col, desc in data_distribution_info.items()]
        )
        dist_block = "Here are the key data distributions for reference:\n" + dist_block

    extra_block = f"\n\nAdditional context: {additional_context}" if additional_context else ""
    analysis_message = "The analysis of the known ground truth values has been completed." if analysis_completed else ""

    if prompt_type == "guide":
        prompt = (
            f"System: You are an expert in {study_type}. You have access to experimental and simulated data and can impute missing values by analyzing correlations among parameters."
            f"\nUser: Given these values for a single data point:\n{context_block}\n"
            f"Can you impute the missing value for '{target_variable}'? Please return only the value (no extra text).{extra_block}"
        )
    elif prompt_type == "map":
        prompt = (
            f"System: You are an expert in {study_type}, able to impute missing values based on observed data distributions and correlations. {dist_block}"
            f"\nUser: Here is the data point:\n{context_block}\n"
            f"Can you impute the missing value for '{target_variable}'? Return only the value.{extra_block}"
        )
    elif prompt_type == "mapv":
        prompt = (
            f"System: You are an expert in {study_type} with full knowledge of parameter distributions. When imputing, add small realistic noise so repeated calls do not return identical values. {dist_block}"
            f"\nUser: Context:\n{context_block}\n"
            f"Impute the missing value for '{target_variable}', returning only the value (with natural variability).{extra_block}"
        )
    elif prompt_type == "cite":
        prompt = (
            f"System: You are an expert in {study_type} and will impute missing values using data and literature knowledge. {analysis_message}"
            f"\nUser: Data:\n{context_block}\n"
            f"Impute the missing value for '{target_variable}', returning only the value. Here is the attribute data: {row[target_variable]}{extra_block}"
        )
    elif prompt_type == "citev":
        prompt = (
            f"System: As an expert in {study_type}, impute the missing value for '{target_variable}'. Make sure imputed values cover both modes and sparse regions. {analysis_message}"
            f"\nUser: Data:\n{context_block}\n"
            f"Here is partial attribute data: {row[target_variable]}{extra_block}"
        )
    elif prompt_type == "gist":
        prompt = (
            f"System: You are an expert in {study_type}. The value for '{target_variable}' is missing."
            f"\nUser: For the data point:\n{context_block}\n"
            "Please provide a 4–5 word descriptor that best characterizes the expected value, based on scientific knowledge (no extra text)."
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return prompt

# ---- GIST FORMALISM ----
def gist_formalism_generic(row, target_variable, context_columns, study_type="scientific experiment"):
    prompt = generate_generalized_prompt(
        row,
        context_columns,
        target_variable,
        prompt_type="gist",
        study_type=study_type
    )
    descriptor = call_chatgpt_4o(prompt, temperature=0.8)
    row[target_variable] = descriptor
    return row

def apply_gist_formalism(data, attributes_to_impute, context_columns, study_type="scientific experiment"):
    for index, row in data.iterrows():
        for target_variable in attributes_to_impute:
            if pd.isnull(row[target_variable]):
                data.loc[index] = gist_formalism_generic(row, target_variable, context_columns, study_type)
    return data

# ---- ITERATIVE IMPUTATION ----
def impute_target_gpt_4o(
    row, target_variable, context_columns,
    temperature=0.8, max_iterations=6,
    prompt_type="guide", study_type="scientific experiment",
    data_distribution_info=None,
    additional_context=None
):
    ground_truth = row[target_variable] if not pd.isnull(row[target_variable]) else None
    # Try to convert ground_truth to float if possible
    if ground_truth is not None:
        try:
            ground_truth = float(ground_truth)
        except Exception:
            ground_truth = None

    prompt = generate_generalized_prompt(
        row, context_columns, target_variable, prompt_type, study_type,
        data_distribution_info=data_distribution_info,
        additional_context=additional_context
    )
    deviations = []
    predictions = []
    deviation = float('inf')

    for iteration in range(max_iterations):
        imputed_value = call_chatgpt_4o(prompt, temperature)
        print(f"    Iteration {iteration+1}: Imputed value for {target_variable} is {imputed_value}")
        try:
            if isinstance(imputed_value, str):
                value = float(imputed_value.strip())
            else:
                value = float(imputed_value)
            predictions.append(value)
            if ground_truth is not None:
                deviation = abs(value - ground_truth)
                deviations.append(deviation)
                print(f"      → Deviation from ground truth: {deviation:.4f}")
        except Exception:
            predictions.append(str(imputed_value).strip() if isinstance(imputed_value, str) else imputed_value)
            print(f"      → Value could not be converted to float (kept as string)")
        if ground_truth is not None and deviation <= 0.1 * abs(ground_truth):
            print(f"      → Early stopping: deviation is below threshold.")
            break
        prompt = (
            f"System: The last imputed value deviated from the ground truth by {deviation:.2f}. "
            f"Refine your prediction if needed.\n" + prompt
        )
    return deviations, predictions, ground_truth

# ---- DATAFRAME-WIDE IMPUTATION ----
def impute_all_attributes(
    data, attributes_to_impute,
    context_columns, temperature=0.8,
    retry_limit=3, perform_analysis=True,
    gist_formalism=False, prompt_type="guide",
    study_type="scientific experiment",
    data_distribution_info=None,
    additional_context=None
):
    ground_truth_predictions = {}
    imputed_data = data.copy()

    def is_descriptor(value):
        return isinstance(value, str)

    for target_variable in attributes_to_impute:
        print(f"\n========== Starting imputation for {target_variable} ==========")

        if gist_formalism and perform_analysis:
            ground_truth_rows = imputed_data[imputed_data[target_variable].notnull()]
            ground_truth_with_descriptors = ground_truth_rows.apply(
                lambda row: gist_formalism_generic(row.to_frame().T, target_variable, context_columns, study_type).squeeze(),
                axis=1
            )
            predictions_for_attribute = []
            for index, row in ground_truth_with_descriptors.iterrows():
                deviations, row_predictions, ground_truth = impute_target_gpt_4o(
                    row, target_variable, context_columns, temperature, max_iterations=6,
                    prompt_type=prompt_type, study_type=study_type,
                    data_distribution_info=data_distribution_info,
                    additional_context=additional_context
                )
                predictions_for_attribute.append(row_predictions)
            ground_truth_predictions[target_variable] = predictions_for_attribute

        elif not gist_formalism and perform_analysis:
            ground_truth_rows = data[data[target_variable].notnull()]
            predictions_for_attribute = []
            for index, row in ground_truth_rows.iterrows():
                deviations, row_predictions, ground_truth = impute_target_gpt_4o(
                    row, target_variable, context_columns, temperature, max_iterations=6,
                    prompt_type=prompt_type, study_type=study_type,
                    data_distribution_info=data_distribution_info,
                    additional_context=additional_context
                )
                predictions_for_attribute.append(row_predictions)
            ground_truth_predictions[target_variable] = predictions_for_attribute

        impute_condition = imputed_data[target_variable].apply(is_descriptor) if gist_formalism else imputed_data[target_variable].isnull()
        retries = 0

        while impute_condition.sum() > 0 and retries < retry_limit:
            print(f"  Retry {retries+1}/{retry_limit} - {impute_condition.sum()} missing values left to impute for {target_variable}.")
            indices_to_impute = imputed_data[impute_condition].index
            for index in indices_to_impute:
                row = data.loc[index]
                print(f"    Imputing row {index} for {target_variable}...")
                prompt = generate_generalized_prompt(
                    row, context_columns, target_variable, prompt_type,
                    study_type, data_distribution_info=data_distribution_info,
                    additional_context=additional_context
                )
                imputed_value = call_chatgpt_4o(prompt, temperature)
                print(f"      → Imputed value: {imputed_value}")
                try:
                    imputed_value = float(imputed_value.strip())
                    imputed_data.at[index, target_variable] = imputed_value
                except Exception:
                    imputed_data.at[index, target_variable] = imputed_value.strip()
            retries += 1
            impute_condition = imputed_data[target_variable].apply(is_descriptor) if gist_formalism else imputed_data[target_variable].isnull()
        print(f"========== Finished imputation for {target_variable} ==========")

    return imputed_data, ground_truth_predictions if perform_analysis else None

