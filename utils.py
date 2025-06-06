
import matplotlib as mpl
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import ceil
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import matplotlib.ticker as ticker
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import seaborn as sns

'''This file contains utility functions that are used in other scripts.'''

# Set up the global matplotlib plotting style
def setup_plot_style():
    """
    Sets up the global matplotlib plotting style.
    """
    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300
    })

# Load the graphene growth conditions data
def load_graphene_data(file_path='Data/graphene_growth_conditions_layers.csv', encoding='ISO-8859-1'):
    """
    Loads the graphene growth conditions data from the Data folder.
    """
    data = pd.read_csv(file_path, encoding=encoding)
    return data

# Call ChatGPT-4o to impute missing values
def call_chatgpt_4o(prompt, temperature):
    """
    Call ChatGPT-4o with a prompt to impute missing values.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are an expert in data imputation."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

#Normalize values
def normalize_value(value, min_value, max_value):
    """
    Normalize a value between 0 and 1 based on the given min and max values.
    If min_value == max_value, return 0 to avoid division by zero.
    """
    if pd.isnull(value):
        return np.nan  # Preserve NaNs
    if min_value == max_value:
        return 0
    return (value - min_value) / (max_value - min_value)

#Post Process Predictions
def post_process_predictions(predictions_dict, data, attributes_to_impute, sort_rows=False):
    """
    Post-process stored predictions for each attribute, comparing with ground truth
    and plotting them in a 2-column grid of subplots. Each attribute gets one subplot.
    """

    n_attributes = len(attributes_to_impute)
    ncols = 2
    nrows = ceil(n_attributes / ncols)  # round up

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(12, 3.5 * nrows),  # Adjust as needed
                             sharex=False, sharey=False)

    # Ensure we can index axes easily (flatten the 2D array)
    axes = axes.flatten() if n_attributes > 1 else [axes]

    # Add a black border around the entire figure
    fig.patch.set_facecolor('white')
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(1.5)

    for idx, target_variable in enumerate(attributes_to_impute):
        print(f"\nPost-processing for {target_variable}...")

        # Current subplot
        ax = axes[idx] if idx < len(axes) else None

        attribute_predictions = predictions_dict[target_variable]
        ground_truth_values = data[data[target_variable].notnull()][target_variable].values

        # Ensure lengths match, trim predictions if necessary
        if len(attribute_predictions) != len(ground_truth_values):
            print(f"Warning: Mismatch in number of predictions vs ground truth for {target_variable}. Truncating...")
            attribute_predictions = attribute_predictions[:len(ground_truth_values)]

        mae_list_avg, mse_list_avg = [], []
        mae_list_min, mse_list_min = [], []
        all_ground_truths, all_imputed_values_avg, all_imputed_values_min = [], [], []

        # Min/max for normalization
        min_value, max_value = np.nanmin(ground_truth_values), np.nanmax(ground_truth_values)

        for i, row_preds in enumerate(attribute_predictions):
            # Filter out NaNs
            row_preds = [p for p in row_preds if not pd.isnull(p)]
            if not row_preds:
                continue

            ground_truth = ground_truth_values[i]
            avg_prediction = np.nanmean(row_preds)
            # Find the prediction with minimum deviation
            min_dev_index = np.nanargmin([abs(p - ground_truth) for p in row_preds])
            min_prediction = row_preds[min_dev_index]

            # Collect data
            all_ground_truths.append(ground_truth)
            all_imputed_values_avg.append(avg_prediction)
            all_imputed_values_min.append(min_prediction)

            # Normalize
            norm_gt = normalize_value(ground_truth, min_value, max_value)
            norm_avg = normalize_value(avg_prediction, min_value, max_value)
            norm_min = normalize_value(min_prediction, min_value, max_value)

            # Metrics
            if not np.isnan(norm_gt):
                mae_list_avg.append(mean_absolute_error([norm_gt], [norm_avg]))
                mse_list_avg.append(mean_squared_error([norm_gt], [norm_avg]))
                mae_list_min.append(mean_absolute_error([norm_gt], [norm_min]))
                mse_list_min.append(mean_squared_error([norm_gt], [norm_min]))

            print(f"Row {i}: Avg Imputation={avg_prediction}, Min Imputation={min_prediction}, GT={ground_truth}")

        # Sort rows if needed
        if sort_rows and all_ground_truths:
            combined = list(zip(all_ground_truths, all_imputed_values_avg, all_imputed_values_min))
            combined.sort(key=lambda x: x[0])
            all_ground_truths, all_imputed_values_avg, all_imputed_values_min = zip(*combined)
            all_ground_truths, all_imputed_values_avg, all_imputed_values_min = (
                list(all_ground_truths),
                list(all_imputed_values_avg),
                list(all_imputed_values_min)
            )

        # Print average metrics
        if mae_list_avg:
            avg_mae, avg_mse = np.mean(mae_list_avg), np.mean(mse_list_avg)
            min_mae, min_mse = np.mean(mae_list_min), np.mean(mse_list_min)
            print(f"Average MAE (Avg) for {target_variable}, Normalized: {avg_mae:.4f}")
            print(f"Average MSE (Avg) for {target_variable}, Normalized: {avg_mse:.4f}")
            print(f"Min Deviation MAE for {target_variable}, Normalized: {min_mae:.4f}")
            print(f"Min Deviation MSE for {target_variable}, Normalized: {min_mse:.4f}")

        # Plot if we have data
        if all_ground_truths and ax is not None:
            # Normalize for plotting
            norm_gts = [normalize_value(gt, min_value, max_value) for gt in all_ground_truths]
            norm_avg = [normalize_value(pred, min_value, max_value) for pred in all_imputed_values_avg]
            norm_min = [normalize_value(pred, min_value, max_value) for pred in all_imputed_values_min]

            ax.scatter(
                range(len(norm_gts)), norm_gts, 
                color='red', label='Ground Truth',
                marker='o', s=50, edgecolors='k'
            )
            ax.scatter(
                range(len(norm_avg)), norm_avg,
                color='blue', label='Avg Imputation',
                marker='o', s=50, edgecolors='k', alpha=0.7
            )
            ax.scatter(
                range(len(norm_min)), norm_min,
                color='orange', label='Min Deviation Imputation',
                marker='x', s=60, edgecolors='k'
            )

            ax.set_xlabel('Row Index', fontsize=11, fontweight='normal')
            ax.set_ylabel(f'Norm. {target_variable}', fontsize=11, fontweight='normal')
            ax.set_title(f'{target_variable}', fontsize=12, fontweight='normal')
            ax.grid(True, linestyle='--', alpha=0.7)

            # Show legend only on the first subplot or as you see fit
            if idx == 0:
                ax.legend(fontsize=9)
            else:
                ax.legend().remove()

    # Hide any unused subplots (in case of an odd number of attributes)
    if n_attributes < len(axes):
        for j in range(n_attributes, len(axes)):
            axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

###############################################################################
# 2. Helper Functions for Data Preparation and Imputation
###############################################################################

def clear_half_values(data, column):
    """
    Clear (set to NaN) 50% of non-null values in the specified column.
    Returns a modified copy of the DataFrame and the original values that were cleared.
    """
    data_with_cleared = data.copy()
    non_nan_indices = data_with_cleared[column].dropna().index
    sample_size = int(0.5 * len(non_nan_indices))
    sample_indices = np.random.choice(non_nan_indices, size=sample_size, replace=False)
    original_values = data_with_cleared.loc[sample_indices, column].copy()
    data_with_cleared.loc[sample_indices, column] = np.nan
    return data_with_cleared, original_values

def impute_data(data, columns_to_impute, knn_neighbors=5,weights='uniform'):
    """
    Impute missing values using KNN, taking into account all columns in columns_to_impute.
    """
    imputer = KNNImputer(n_neighbors=knn_neighbors, weights=weights)
    imputed_array = imputer.fit_transform(data[columns_to_impute])
    imputed_data_df = pd.DataFrame(imputed_array, columns=columns_to_impute, index=data.index)
    return imputed_data_df

def normalize_values(original, imputed):
    """
    Normalize both original and imputed values using MinMaxScaler.
    """
    scaler = MinMaxScaler()

    if len(original) == 0 or len(imputed) == 0:
        return original.values, imputed.values

    combined = pd.concat([original, imputed])
    scaled_combined = scaler.fit_transform(combined.values.reshape(-1, 1))

    scaled_original = scaled_combined[:len(original)]
    scaled_imputed = scaled_combined[len(original):]

    return scaled_original.flatten(), scaled_imputed.flatten()

def evaluate_imputation(imputed_data, original_values, column):
    """
    Evaluate the imputation results by calculating MAE and MSE.
    Returns a dictionary with keys "MAE" and "MSE".
    """
    original_values = original_values.reindex(imputed_data.index)
    imputed = imputed_data[column][original_values.notna()]
    original = original_values.dropna()

    if len(original) == 0 or len(imputed) == 0:
        print(f"Skipping column {column} due to insufficient data for comparison.")
        return None

    normalized_original, normalized_imputed = normalize_values(original, imputed)
    mae = mean_absolute_error(normalized_original, normalized_imputed)
    mse = mean_squared_error(normalized_original, normalized_imputed)

    return {"MAE": mae, "MSE": mse}

###############################################################################
# 3. Plotting Functions for KNN Imputation Comparison
###############################################################################
def plot_all_knn_imputations(data_preprocessed, columns_to_impute):
    """
    1) For each column in columns_to_impute, clear 50% of its data,
       run KNN imputation, then collect the ground truth + imputed values.
    2) Sort them by ground truth.
    3) Plot all attributes in a single figure with 2 columns of subplots,
       removing bold text and adding a black border.
    """
    width_cm = 17.8  # double column in cm
    height_cm = 12.0 * 1.5
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54

    n_attrs = len(columns_to_impute)
    ncols = 2
    nrows = ceil(n_attrs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width_in, height_in),
                             facecolor='white')
    axes = axes.flatten()

    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)

    for i, target_column in enumerate(columns_to_impute):
        ax = axes[i]

        data_with_cleared, original_values = clear_half_values(data_preprocessed, target_column)
        imputed_data_df = impute_data(data_with_cleared, columns_to_impute)

        original_values = original_values.reindex(imputed_data_df.index)
        original = original_values.dropna()
        imputed = imputed_data_df[target_column][original_values.notna()]

        norm_orig, norm_imp = normalize_values(original, imputed)

        combined = list(zip(norm_orig, norm_imp))
        combined.sort(key=lambda x: x[0])
        if combined:
            sorted_orig, sorted_imp = zip(*combined)
        else:
            sorted_orig, sorted_imp = [], []

        ax.scatter(range(len(sorted_orig)), sorted_orig,
                   color='red', label='Ground Truth', s=30, edgecolors='k')
        ax.scatter(range(len(sorted_imp)), sorted_imp,
                   color='blue', label='Imputed', alpha=0.7, edgecolors='k')

        ax.set_xlabel('Row Index', fontsize=10, fontweight='normal')
        ax.set_ylabel(f'{target_column} (Norm.)', fontsize=10, fontweight='normal')
        ax.set_title(f'KNN: {target_column}', fontsize=10, fontweight='normal')

        ax.grid(True, linestyle='--', alpha=0.7)

        if i == 0:
            ax.legend(fontsize=8)
        else:
            ax.legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def compare_knn_for_k_values(data_preprocessed, columns_to_impute, k_values=[4, 5, 6]):
    """
    For each K in k_values, perform KNN imputation on each target column,
       evaluate the imputation (using normalized MAE and MSE), and return a nested dictionary
       with metrics for each K and each column.
    """
    metrics = {k: {} for k in k_values}
    for target_column in columns_to_impute:
        for k in k_values:
            data_with_cleared, original_values = clear_half_values(data_preprocessed, target_column)
            imputed_data_df = impute_data(data_with_cleared, columns_to_impute, knn_neighbors=k)
            error_metrics = evaluate_imputation(imputed_data_df, original_values, target_column)
            if error_metrics is not None:
                metrics[k][target_column] = error_metrics
    return metrics

def average_metrics_over_columns(metrics, k_values=[4, 5, 6]):
    """
    Compute average MAE and MSE across all target columns for each K value.
    Returns a dictionary where each key is a K value and its value is a dict with average metrics.
    """
    avg_metrics = {}
    for k in k_values:
        mae_list = []
        mse_list = []
        for col, m in metrics[k].items():
            mae_list.append(m["MAE"])
            mse_list.append(m["MSE"])
        avg_metrics[k] = {
            "MAE": np.mean(mae_list) if mae_list else None,
            "MSE": np.mean(mse_list) if mse_list else None
        }
    return avg_metrics

###############################################################################
# 4. Bar Plot Comparison with Consistent Decimal Formatting
###############################################################################
pretty_names = {
    'Pressure (mbar)': "Pressure (mbar)",
    'Growth Time (min)': "Growth Time (min)",
    'H2': "H2 Flow Rate (sccm)",
    'CH4': "CH4 Flow Rate (sccm)",
    'Ar': "Ar Flow Rate (sccm)",
    'C2H2': "C2H2 Flow Rate (sccm)"
}

method_colors = {
    'KNN': '#4E79A7',
    'GUIDE': '#F28E2B',
    'MAP': '#E15759',
    'MAPV': '#76B7B2',
    'CITE': '#59A14F',
    'CITEV': '#EDC948',
    'GIST': '#FF5733'
}

def compare_knn_gpt4o_mae(errors, methods_mae, save_path=None):
    variables = list(errors.keys())
    knn_values = [round(errors[var]["MAE"], 3) for var in variables]
    method_names = list(methods_mae.keys())
    method_values = {method: [round(methods_mae[method][var], 3) for var in variables] for method in method_names}
    
    n_cols = 2
    n_rows = ceil(len(variables) / n_cols)
    width_in, height_in = 8, 6
    
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(width_in, height_in),
                            facecolor='white')
    axs = axs.flatten()
    
    fig.text(0.04, 0.5, "Mean Absolute Error (MAE)",
             va='center', rotation='vertical', fontsize=10)
    
    x_positions = np.arange(len(method_names) + 1)
    width = 0.4
    
    for i, var in enumerate(variables):
        ax = axs[i]
        ax.bar(x_positions[0], knn_values[i], width,
               label='KNN',
               color=method_colors['KNN'],
               alpha=0.85,
               edgecolor='black')
        
        for j, method in enumerate(method_names):
            bar_color = method_colors.get(method, '#333333')
            ax.bar(x_positions[j + 1], method_values[method][i], width,
                   label=method,
                   color=bar_color,
                   alpha=0.85,
                   edgecolor='black')
        
        attribute_title = pretty_names.get(var, var)
        ax.set_title(attribute_title, fontsize=10, pad=10, fontweight='normal')
        
        row = i // n_cols
        if row == n_rows - 1:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(['KNN'] + method_names, rotation=15)
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)
        
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    for i in range(len(variables), len(axs)):
        fig.delaxes(axs[i])
    
    fig.tight_layout(rect=[0.07, 0.03, 1, 0.97])
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()

def compare_knn_gpt4o_mse(errors, methods_mse, save_path=None):
    variables = list(errors.keys())
    knn_values = [round(errors[var]["MSE"], 3) for var in variables]
    method_names = list(methods_mse.keys())
    method_values = {method: [round(methods_mse[method][var], 3) for var in variables] for method in method_names}
    
    n_cols = 2
    n_rows = ceil(len(variables) / n_cols)
    width_in, height_in = 8, 6
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(width_in, height_in),
                            facecolor='white')
    axs = axs.flatten()
    
    fig.text(0.04, 0.5, "Mean Squared Error (MSE)",
             va='center', rotation='vertical', fontsize=10)
    
    x_positions = np.arange(len(method_names) + 1)
    width = 0.4
    
    for i, var in enumerate(variables):
        ax = axs[i]
        ax.bar(x_positions[0], knn_values[i], width,
               label='KNN',
               color=method_colors['KNN'],
               alpha=0.85,
               edgecolor='black')
        
        for j, method in enumerate(method_names):
            bar_color = method_colors.get(method, '#333333')
            ax.bar(x_positions[j + 1], method_values[method][i], width,
                   label=method,
                   color=bar_color,
                   alpha=0.85,
                   edgecolor='black')
        
        attribute_title = pretty_names.get(var, var)
        ax.set_title(attribute_title, fontsize=10, pad=10, fontweight='normal')
        
        row = i // n_cols
        if row == n_rows - 1:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(['KNN'] + method_names, rotation=15)
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([])
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.2)
        
        ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    for i in range(len(variables), len(axs)):
        fig.delaxes(axs[i])
    
    fig.tight_layout(rect=[0.07, 0.03, 1, 0.97])
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    
    plt.show()

def calculate_jsd(p, q, bins=20):
        """Calculate Jensen-Shannon Divergence (squared) between two distributions."""
        p = pd.to_numeric(p, errors='coerce').dropna()
        q = pd.to_numeric(q, errors='coerce').dropna()

        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)

        if p_hist.sum() > 0:
            p_hist /= p_hist.sum()
        if q_hist.sum() > 0:
            q_hist /= q_hist.sum()

        return jensenshannon(p_hist, q_hist) ** 2

def calculate_emd(p, q, bins=20):
    """Calculate Earth Mover's Distance (EMD) between two distributions."""
    p = pd.to_numeric(p, errors='coerce').dropna()
    q = pd.to_numeric(q, errors='coerce').dropna()

    p_hist, bin_edges = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return wasserstein_distance(bin_centers, bin_centers, p_hist, q_hist)

def plot_methods_by_rows(
    data_preprocessed,
    data_imputed_knn,
    chatgpt_imputed_datasets,
    chatgpt_labels,
    columns,
    bins=20,
    save_path=None
):
    """
    Creates a multi-row figure with:
      Rows = ["Raw Data", "KNN"] + chatgpt_labels
      Columns = the chosen attributes (columns)
    """
    method_row_labels = ["Raw Data", "KNN"] + chatgpt_labels
    n_rows = len(method_row_labels)
    n_cols = len(columns)

    # Convert figure size from centimeters to inches.
    width_cm = 17.8
    height_cm = 24.0
    fig_width_in = width_cm / 2.54
    fig_height_in = height_cm / 2.54

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_width_in, fig_height_in),
        facecolor='white'
    )
    axs = np.atleast_2d(axs)

    # Prepare dictionaries for JSD & EMD results.
    jsd_results = {m: {} for m in method_row_labels}
    emd_results = {m: {} for m in method_row_labels}

    # Optional color mapping.
    method_colors = {
        "Raw Data": "tab:blue",
        "KNN": "tab:green",
        "GUIDE": "#E15759",
        "MAPV": "#76B7B2",
        "CITEV": "#EDC948",
        "GIST": "#FF5733"
    }

    # Dictionary for attribute labels (with units where applicable).
    display_attr_labels = {
        'Pressure (mbar)': 'Pressure (mbar)',
        'Growth Time (min)': 'Growth Time (min)',
        'C2H4': 'C2H4 (sccm)',
        'C2H2': 'C2H2 (sccm)'
    }

    raw_data = data_preprocessed

    for i, method_label in enumerate(method_row_labels):
        # Choose the appropriate DataFrame.
        if method_label == "Raw Data":
            df_method = data_preprocessed
        elif method_label == "KNN":
            df_method = data_imputed_knn
        else:
            idx = chatgpt_labels.index(method_label)
            df_method = chatgpt_imputed_datasets[idx]

        row_color = method_colors.get(method_label, "tab:gray")

        for j, attr in enumerate(columns):
            ax = axs[i, j]
            series = pd.to_numeric(df_method[attr], errors='coerce').dropna()

            sns.histplot(
                series,
                bins=bins,
                stat="count",
                kde=True,
                edgecolor='black',
                alpha=0.8,
                ax=ax,
                color=row_color
            )
            ax.grid(True, linestyle='--', alpha=0.5)

            # Set black borders.
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.2)

            # Top row: set attribute titles.
            if i == 0:
                display_label = display_attr_labels.get(attr, attr)
                ax.set_title(display_label, fontsize=10, rotation=0)

            # Left column: add row labels.
            if j == 0:
                ax.set_ylabel(method_label, fontsize=10)
            else:
                ax.set_ylabel("")

            # Bottom row: adjust x-axis label.
            if i < n_rows - 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("", fontsize=9)

            # Compute JSD/EMD against raw data.
            if method_label != "Raw Data":
                raw_series = pd.to_numeric(raw_data[attr], errors='coerce').dropna()
                jsd_val = calculate_jsd(raw_series, series, bins=bins)
                emd_val = calculate_emd(raw_series, series, bins=bins)
                jsd_results[method_label][attr] = jsd_val
                emd_results[method_label][attr] = emd_val
            else:
                jsd_results["Raw Data"][attr] = 0.0
                emd_results["Raw Data"][attr] = 0.0

    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

    return jsd_results, emd_results

def plot_jsd_emd_side_by_side(jsd_results, emd_results, save_path=None):
    """
    Creates a 2×2 figure with:
      - Top-left:  JSD heatmap
      - Top-right: EMD heatmap
      - Bottom-left: JSD bar chart
      - Bottom-right: EMD bar chart

    The method order is: ["KNN", "GUIDE", "MAPV", "CITEV", "GIST"].
    Attribute labels are shortened (units removed), and x-tick labels are rotated.
    """
    final_method_order = ["KNN", "GUIDE", "MAPV", "CITEV", "GIST"]

    # Collect all attributes.
    all_attrs = set()
    for method_dict in (jsd_results, emd_results):
        for m in method_dict.keys():
            all_attrs.update(method_dict[m].keys())
    all_attrs = sorted(list(all_attrs))

    # Map full attribute names to shorter labels.
    short_attr_labels = {
        'Pressure (mbar)': 'Pressure',
        'Growth Time (min)': 'Growth Time',
        'C2H4 (sccm)': 'C2H4',
        'C2H2 (sccm)': 'C2H2'
    }
    display_attrs = [short_attr_labels.get(a, a) for a in all_attrs]

    # Build matrices for JSD and EMD.
    jsd_matrix = []
    emd_matrix = []
    for m in final_method_order:
        row_jsd = []
        row_emd = []
        for attr in all_attrs:
            row_jsd.append(jsd_results[m].get(attr, np.nan))
            row_emd.append(emd_results[m].get(attr, np.nan))
        jsd_matrix.append(row_jsd)
        emd_matrix.append(row_emd)

    jsd_matrix = np.array(jsd_matrix)
    emd_matrix = np.array(emd_matrix)

    # Create the 2×2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
    ax_jsd_heat = axes[0, 0]
    ax_emd_heat = axes[0, 1]
    ax_jsd_bar  = axes[1, 0]
    ax_emd_bar  = axes[1, 1]

    # JSD Heatmap.
    h1 = sns.heatmap(
        jsd_matrix,
        annot=True,
        xticklabels=display_attrs,
        yticklabels=final_method_order,
        cmap="viridis",
        fmt=".3f",
        ax=ax_jsd_heat
    )
    h1.set_xticklabels(display_attrs, rotation=15, ha='right')
    ax_jsd_heat.set_title("JSD", fontsize=12)
    ax_jsd_heat.set_xlabel("")
    ax_jsd_heat.set_ylabel("")

    # EMD Heatmap.
    h2 = sns.heatmap(
        emd_matrix,
        annot=True,
        xticklabels=display_attrs,
        yticklabels=final_method_order,
        cmap="viridis",
        fmt=".3f",
        ax=ax_emd_heat
    )
    h2.set_xticklabels(display_attrs, rotation=15, ha='right')
    ax_emd_heat.set_title("EMD", fontsize=12)
    ax_emd_heat.set_xlabel("")
    ax_emd_heat.set_ylabel("")

    # JSD Bar Chart.
    jsd_means = np.nanmean(jsd_matrix, axis=1)
    jsd_stds  = np.nanstd(jsd_matrix, axis=1)
    sns.barplot(
        x=final_method_order,
        y=jsd_means,
        ax=ax_jsd_bar,
        palette="muted",
        edgecolor="black",
        capsize=0.2
    )
    ax_jsd_bar.errorbar(
        x=range(len(final_method_order)),
        y=jsd_means,
        yerr=jsd_stds,
        fmt="none",
        capsize=5,
        color="black"
    )
    ax_jsd_bar.set_ylabel("JSD (Average)", fontsize=10)
    ax_jsd_bar.set_xlabel("")
    ax_jsd_bar.grid(axis="y", linestyle="--", alpha=0.7)

    # EMD Bar Chart.
    emd_means = np.nanmean(emd_matrix, axis=1)
    emd_stds  = np.nanstd(emd_matrix, axis=1)
    sns.barplot(
        x=final_method_order,
        y=emd_means,
        ax=ax_emd_bar,
        palette="muted",
        edgecolor="black",
        capsize=0.2
    )
    ax_emd_bar.errorbar(
        x=range(len(final_method_order)),
        y=emd_means,
        yerr=emd_stds,
        fmt="none",
        capsize=5,
        color="black"
    )
    ax_emd_bar.set_ylabel("EMD (Average)", fontsize=10)
    ax_emd_bar.set_xlabel("")
    ax_emd_bar.grid(axis="y", linestyle="--", alpha=0.7)

    plt.subplots_adjust(
        bottom=0.15,
        top=0.92,
        wspace=0.2,
        hspace=0.3
    )
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

#Temperature plots
    
def iterative_impute(data, idx, temperature, target="CH4", max_iterations=6):
    row = data.iloc[idx]
    ground_truth = row[target]
   
    cols = {
        "cvd_method": "CVD Method",
        "pressure": "Pressure (mbar)",
        "temperature": "Temperature (°C)",
        "growth_time": "Growth Time (min)",
        "substrate": "Substrate",
        "h2": "H2",
        "c2h4": "C2H4",
        "ar": "Ar",
        "c2h2": "C2H2"
    }
    prompt_values = {k: (row[col] if pd.notnull(row[col]) else 'unknown') for k, col in cols.items()}
    prompt_values["target"] = target

    prompt_template = (
        "System: You are a CVD expert with access to experimental and/or simulated data from literature "
        "which you can understand to derive correlations between known and unknown quantities for given datasets. "
        "User: For the CVD process {cvd_method}, with the following parameters of pressure {pressure}, temperature {temperature}, "
        "growth time {growth_time}, grown over a substrate {substrate}, and with gas flow rates - H2 {h2}, C2H4 {c2h4}, Ar {ar}, C2H2 {c2h2}, "
        "please provide only the missing value of {target} as a single number, without any additional text."
    )
    prompt = prompt_template.format(**prompt_values)
    deviations, predictions = [], []

    for i in range(max_iterations):
        print(f"\nIteration {i+1} for sample {idx} at Temp {temperature:.1f}")
        print("Prompt:", prompt)
        response = call_chatgpt_4o(prompt, temperature)
        print("Response:", response)
        try:
            value = float(response.strip())
        except ValueError:
            print(f"Iteration {i+1}: Non-numeric response: {response}")
            break
        predictions.append(value)
        deviation = abs(value - ground_truth)
        deviations.append(deviation)
        print(f"Deviation: {deviation:.2f}")
        if deviation <= 0.1 * ground_truth:
            print(f"Iteration {i+1}: Acceptable deviation reached.")
            break
        prompt = f"System: The deviation from the ground truth was {deviation:.2f}. Please refine the prediction if needed.\n" + prompt

    return deviations, predictions, ground_truth

def aggregate_imputation(data, indices, temperature, target="CH4", max_iterations=6):
    results = [iterative_impute(data, i, temperature, target, max_iterations) for i in indices]
    final_deviations = [dev[-1] for dev, preds, gt in results if dev]
    return final_deviations, np.mean(final_deviations)

setup_plot_style()

def plot_avg_deviations(avg_deviations_dict):
    """
    Plots a single point (or line) for each temperature in avg_deviations_dict.
    Axis labels are not bold. A black border covers the width of the figure.
    """
    temps = sorted(avg_deviations_dict.keys())
    avg_devs = [avg_deviations_dict[t] for t in temps]
    
    # Set figure size to approximately the text width of a typical LaTeX document (e.g., 6.5 inches)
    plt.figure(figsize=(6.5, 4))
    
    plt.plot(temps, avg_devs, marker='o', linewidth=2, color='tab:blue')
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Deviation of CH4 (sccm)", fontsize=12)
    plt.title("", fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add a black border around the plot area
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.show()