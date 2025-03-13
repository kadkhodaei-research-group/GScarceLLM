# Function for Equal-Width Binning
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score

def equal_width_binning(df, columns, n_bins_dict):
    bin_map = {}
    for column in columns:
        n_bins = n_bins_dict.get(column, 16)  # Use column-specific n_bins, default to 16
        bins = np.linspace(df[column].min(), df[column].max(), n_bins + 1)
        df[column] = np.digitize(df[column], bins) - 1
        # Create a mapping from labels to bin ranges
        bin_map[column] = {i: (bins[i], bins[i+1]) for i in range(len(bins) - 1)}
    return df, bin_map

# Function for Equal-Frequency Binning
def equal_frequency_binning(df, columns, n_bins_dict):
    bin_map = {}
    for column in columns:
        n_bins = n_bins_dict.get(column, 16)  # Use column-specific n_bins, default to 16
        df[column], bins = pd.qcut(df[column], n_bins, labels=False, retbins=True, duplicates='drop')
        # Create a mapping from labels to bin ranges
        bin_map[column] = {i: (bins[i], bins[i+1]) for i in range(len(bins) - 1)}
    return df, bin_map

# Function for K-means Binning with automatic number of clusters
def kmeans_binning(df, columns, n_bins_dict):
    bin_map = {}
    for column in columns:
        min_bins = n_bins_dict.get(column, 5)  # Use column-specific min_bins
        max_bins = n_bins_dict.get(column, 10)  # Use column-specific max_bins
        data = df[column].values.reshape(-1, 1)

        distortions = []
        silhouette_scores = []

        # Compute K-means for a range of cluster sizes (min_bins to max_bins)
        for k in range(min_bins, max_bins + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            distortions.append(kmeans.inertia_)
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)

        # Select the optimal number of clusters based on the silhouette score
        optimal_k = np.argmax(silhouette_scores) + min_bins

        # Fit KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(data)
        df[column] = kmeans.labels_

        # Store the cluster centers for the bin map
        bin_map[column] = {i: kmeans.cluster_centers_[i][0] for i in range(optimal_k)}

    return df, bin_map

# Function for Decision Tree Binning with automatic number of bins
def decision_tree_binning(df, columns):
    bin_map = {}
    for column in columns:
        data = df[column].values.reshape(-1, 1)
        clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=10)  # Limit number of leaves (bins)
        clf.fit(data, np.zeros(len(data)))  # Dummy target for unsupervised binning

        # Create labels based on the binning
        df[column] = clf.apply(data) - 1

        # Extract thresholds (split points)
        thresholds = clf.tree_.threshold
        valid_thresholds = thresholds[thresholds != -2]  # Filter out -2 (leaves)

        if len(valid_thresholds) > 0:
            valid_thresholds = np.sort(np.unique(valid_thresholds))
            bin_map[column] = {i: (valid_thresholds[i-1], valid_thresholds[i]) if i > 0 
                               else (-np.inf, valid_thresholds[0])
                               for i in range(len(valid_thresholds))}
            # Add the last range (upper bound)
            bin_map[column][len(valid_thresholds)] = (valid_thresholds[-1], np.inf)
        else:
            bin_map[column] = "No valid splits"
    return df, bin_map

def supervised_tree_binning(df, columns, target_column, min_splits=2, max_leaf_nodes=10):
    bin_map = {}
    target = df[target_column]
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Skipping non-numeric column: {column}")
            continue

        data = df[column].fillna(df[column].median()).values.reshape(-1, 1)
        if np.all(data == data[0]):
            print(f"Skipping column with no variance: {column}")
            bin_map[column] = "No variance"
            continue

        clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_splits)
        clf.fit(data, target)
        df[column] = clf.apply(data) - 1

        thresholds = clf.tree_.threshold
        valid_thresholds = thresholds[thresholds != -2]

        if len(valid_thresholds) > 0:
            valid_thresholds = np.sort(np.unique(valid_thresholds))
            bin_map[column] = {i: (valid_thresholds[i-1], valid_thresholds[i]) if i > 0 
                               else (-np.inf, valid_thresholds[0])
                               for i in range(len(valid_thresholds))}
            bin_map[column][len(valid_thresholds)] = (valid_thresholds[-1], np.inf)
        else:
            bin_map[column] = "No valid splits"
    return df, bin_map

# Main function to load CSV and apply binning methods
def apply_binning(data, method, columns, n_bins_dict):
    # Use the provided data (df) and apply the chosen binning method.
    if method == 'equal_width':
        return equal_width_binning(data, columns, n_bins_dict)
    elif method == 'equal_frequency':
        return equal_frequency_binning(data, columns, n_bins_dict)
    elif method == 'kmeans':
        return kmeans_binning(data, columns, n_bins_dict)
    elif method == 'decision_tree':
        return decision_tree_binning(data, columns)
    elif method == 'supervised_tree_binning':
        # Here the target column is hard-coded to 'No. of Graphene Layers'
        return supervised_tree_binning(data, columns, target_column='No. of Graphene Layers', 
                                       min_splits=2, max_leaf_nodes=10)
    else:
        raise ValueError("Invalid method. Choose from 'equal_width', 'equal_frequency', 'kmeans', 'decision_tree', or 'supervised_tree_binning'.")