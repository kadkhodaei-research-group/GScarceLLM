import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from utils import setup_plot_style
from collections import Counter
import sys, pathlib
sys.path.append(pathlib.Path(__file__).parent / "external" / "metalhydride")

# import xgboost  # XGBoost import commented out

class_label_map = {
    0: "Single",
    1: "Bilayer",
    2: "Multilayer"
}

##############################################################################
# Preprocessing, SHAP, and t-SNE Functions
##############################################################################

def preprocess_with_sampling(X, y, sampling_method=None):
    """
    Apply optional sampling techniques to handle class imbalance.
    - If sampling_method == 'SMOTE', but the smallest class has <= k_neighbors (5) samples,
      skip SMOTE to avoid errors.
    - Otherwise, run SMOTE with strategy='auto' so it only upsamples the minority.
    - 'NearMiss' will always run undersampling.
    - If sampling_method is None or not recognized, return (X, y) unchanged.
    """
    if sampling_method == 'SMOTE':
        # Count occurrences of each class
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        
        # SMOTE needs at least (k_neighbors + 1) samples in every class to upsample
        if min_count <= 5:
            # Too few minority samples → skip SMOTE altogether
            return X, y
        
        # Otherwise, perform SMOTE (only upsamples minority to match majority)
        smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    elif sampling_method == 'NearMiss':
        from imblearn.under_sampling import NearMiss
        near_miss = NearMiss()
        X_resampled, y_resampled = near_miss.fit_resample(X, y)
        return X_resampled, y_resampled

    # No sampling requested or unrecognized method: return original data
    return X, y


def manual_shap(model, X, feature_names, num_samples=100):
    """
    Approximate SHAP values by perturbing each feature and measuring
    the change in model predictions.
    """
    if num_samples > len(X):
        num_samples = len(X)
    X_sampled = X.sample(n=num_samples, random_state=42)
    shap_values_approx = pd.DataFrame(0, index=X_sampled.index, columns=feature_names)
    base_preds = model.predict_proba(X_sampled)[:, 1]

    for feature in feature_names:
        X_perturbed = X_sampled.copy()
        X_perturbed[feature] = X_perturbed[feature].mean()
        perturbed_preds = model.predict_proba(X_perturbed)[:, 1]
        shap_values_approx[feature] = base_preds - perturbed_preds

    return shap_values_approx

def plot_tsne_with_decision_boundary(model, X_scaled, y, title):
    """
    Plot t-SNE visualization with decision boundaries.
    Uses the global `class_label_map` to rename classes.
    """
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap

    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)

    x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
    y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X_embedded)
    original_points = knn.kneighbors(grid, return_distance=False)
    grid_original = X_scaled[original_points.ravel()]

    proba = model.predict_proba(grid_original)
    num_classes = proba.shape[1]

    if num_classes == 2:
        grid_predictions = proba[:, 1].reshape(xx.shape)
        ctf = plt.contourf(xx, yy, grid_predictions, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.colorbar(ctf, label=f"Probability ({class_label_map.get(1, 'Class 1')})")
    else:
        class_argmax = np.argmax(proba, axis=1).reshape(xx.shape)
        cmap_multi = plt.cm.get_cmap('tab10', num_classes)
        ctf = plt.contourf(xx, yy, class_argmax, alpha=0.8, cmap=cmap_multi)
        cbar = plt.colorbar(ctf, ticks=range(num_classes))
        cbar.set_label("Predicted Class")
        cbar.ax.set_yticklabels([class_label_map.get(c, f"Class {c}") for c in range(num_classes)])

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                c=y, edgecolor='k', cmap=plt.cm.coolwarm, alpha=0.8)

    unique_classes = np.unique(y)
    legend_handles = []
    from matplotlib.patches import Patch
    for i, cls_ in enumerate(unique_classes):
        color_i = plt.cm.coolwarm(i / len(unique_classes))
        legend_handles.append(Patch(color=color_i, label=class_label_map.get(cls_, f"Class {cls_}")))
    plt.legend(handles=legend_handles, title="Classes")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

##############################################################################
# SVM Training/Evaluation Function
##############################################################################
def train_evaluate_svm_classifier(data, target_column, sampling_method=None):
    """
    Train and evaluate an SVM classifier with GridSearchCV.
    Produces a confusion matrix, learning curve, SHAP-based feature importances,
    and a t-SNE plot with decision boundaries.
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    X = data[numeric_columns].drop(columns=[target_column], errors='ignore')
    y = data[target_column]

    X_resampled, y_resampled = preprocess_with_sampling(X, y, sampling_method=sampling_method)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly']
    }
    svc = SVC(probability=True)
    grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.show()

    print(classification_report(y_test, y_pred))

    train_sizes, train_scores, test_scores = learning_curve(
        grid_search.best_estimator_, X_resampled, y_resampled, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="gainsboro")
    plt.plot(train_sizes, train_mean, 'o-', color="black", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()

    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    shap_approx_df = manual_shap(grid_search.best_estimator_, X_test_scaled_df, feature_names=X.columns)
    shap_approx_df.mean().sort_values(ascending=False).plot(kind='bar', title="Feature Importances")
    plt.show()

    plot_tsne_with_decision_boundary(grid_search.best_estimator_, X_test_scaled, y_test,
                                     "t-SNE with Decision Boundaries")

    return grid_search.best_estimator_

##############################################################################
# Individual and Aggregate Learning Curves Plots
##############################################################################
def plot_model_learning_curves(result_df, save_path=None):
    """
    Create a 2×2 figure of learning curves for:
    Random Forest, Decision Tree, SVM, and XGBoost.
    """
    # Commented out XGBoost from the list:
    models = ["Random Forest", "Decision Tree", "SVM"]  # , "XGBoost"  <-- XGBoost commented out
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), facecolor='white')
    axs = axs.ravel()

    for i, model_name in enumerate(models):
        ax = axs[i]
        model_data = result_df[result_df['model'] == model_name].sort_values('train_size')
        train_sizes = model_data['train_size'].values
        train_mean  = model_data['train_mean'].values
        train_std   = model_data['train_std'].values
        cv_mean     = model_data['cv_mean'].values
        cv_std      = model_data['cv_std'].values

        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='red')
        ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std,
                        alpha=0.2, color='green')
        ax.plot(train_sizes, train_mean, '-o', color='red', label='Training score')
        ax.plot(train_sizes, cv_mean,   '-o', color='green', label='Cross-validation score')
        ax.set_title(model_name, fontsize=12)
        ax.set_xlabel("Training Size", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches="tight", dpi=300)
    plt.show()

def plot_aggregate_learning_curves(data, target_column, save_path=None):
    """
    Train four models (SVM, Decision Tree, Random Forest, and XGBoost)
    on the provided data, compute their learning curves, and plot a 2×2 figure.
    """
    import matplotlib as mpl
    # from xgboost import XGBClassifier  # XGBoost import commented out
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    # from xgboost import XGBClassifier  # XGBoost import commented out

    setup_plot_style()

    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    X = data[numeric_columns]
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    param_grid_dt = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }
    param_grid_xgb = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100]
    }

    # SVM
    svm = SVC(probability=True, random_state=42)
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
    grid_svm.fit(X_scaled, y)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
    grid_dt.fit(X_scaled, y)

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
    grid_rf.fit(X_scaled, y)

    # XGBoost
    # from xgboost import XGBClassifier  # XGBoost import commented out
    # xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    # grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
    # grid_xgb.fit(X_scaled, y)

    models = {
        "SVM": grid_svm.best_estimator_,
        "Decision Tree": grid_dt.best_estimator_,
        "Random Forest": grid_rf.best_estimator_
        # "XGBoost": grid_xgb.best_estimator_  # XGBoost model commented out
    }

    results = {}
    for model_name, model in models.items():
        train_sizes, train_scores, cv_scores = learning_curve(
            model, X_scaled, y, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=-1
        )
        results[model_name] = {
            "train_size": train_sizes,
            "train_mean": np.mean(train_scores, axis=1),
            "train_std": np.std(train_scores, axis=1),
            "cv_mean": np.mean(cv_scores, axis=1),
            "cv_std": np.std(cv_scores, axis=1)
        }

    width_cm = 17.8
    height_cm = 12.0
    width_in = width_cm / 2.54
    height_in = height_cm / 2.54
    fig, axs = plt.subplots(2, 2, figsize=(width_in, height_in), facecolor='white')
    axs = axs.ravel()

    for i, (mname, data_dict) in enumerate(results.items()):
        ax = axs[i]
        ax.fill_between(
            data_dict["train_size"],
            data_dict["train_mean"] - data_dict["train_std"],
            data_dict["train_mean"] + data_dict["train_std"],
            color="blue", alpha=0.2
        )
        ax.fill_between(
            data_dict["train_size"],
            data_dict["cv_mean"] - data_dict["cv_std"],
            data_dict["cv_mean"] + data_dict["cv_std"],
            color="red", alpha=0.2
        )
        ax.plot(
            data_dict["train_size"], data_dict["train_mean"],
            color="blue", marker="o", linestyle="-", label="Training"
        )
        ax.plot(
            data_dict["train_size"], data_dict["cv_mean"],
            color="red", marker="s", linestyle="--", label="Cross-validation"
        )
        ax.set_title(mname, fontsize=10)
        ax.set_xlabel("Training Size", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout(rect=[0.07, 0.03, 1, 0.97])
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

##############################################################################
# 2×2 t-SNE Subplot
##############################################################################
def plot_tsne_for_four_datasets(
    files_and_targets,
    subplot_titles=None,
    sampling_method=None,
    figsize=(12, 10),
    pdf_filename="tsne_fourplot.pdf"
):
    """
    Loads each CSV, trains an SVM (with GridSearchCV), and creates a 2×2 t-SNE figure.
    """
    from matplotlib.colors import ListedColormap

    # Overriding the label map for t-SNE plots
    custom_class_map = {0: "Monolayer", 1: "Bilayer", 2: "Multilayer"}
    markers = {0: 'o', 1: 's', 2: '^'}
    colors_binary = sns.color_palette("colorblind", 2)
    colors_ternary = sns.color_palette("colorblind", 3)

    if subplot_titles is None:
        subplot_titles = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4"]

    fig, axs = plt.subplots(2, 2, figsize=figsize, facecolor='white')
    axs = axs.ravel()
    labels = ["(a)", "(b)", "(c)", "(d)"]

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly']
    }

    for i, (file_path, target_col) in enumerate(files_and_targets):
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        X = df[numeric_cols]
        y = df[target_col]

        X_res, y_res = preprocess_with_sampling(X, y, sampling_method=sampling_method)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        svc = SVC(probability=True)
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=0)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X_test_scaled)

        x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
        y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(X_embedded)
        original_points = knn.kneighbors(grid_points, return_distance=False)
        grid_original = X_test_scaled[original_points.ravel()]

        proba = best_model.predict_proba(grid_original)
        num_classes = proba.shape[1]

        ax = axs[i]
        if num_classes == 2:
            mapping = {0: "Monolayer", 1: "Multilayer"}
        else:
            mapping = custom_class_map

        if num_classes == 2:
            grid_predictions = proba[:, 1].reshape(xx.shape)
            cmap_binary = ListedColormap(colors_binary)
            ctf = ax.contourf(xx, yy, grid_predictions,
                              levels=np.linspace(0, 1, 50),
                              alpha=0.8, cmap=cmap_binary)
            cb = fig.colorbar(ctf, ax=ax)
            cb.ax.tick_params(labelsize=10)
            label_for_class1 = mapping.get(1, "Class 1")
            cb.set_label(f"Probability({label_for_class1})", fontsize=12, fontweight='normal')
        else:
            class_argmax = np.argmax(proba, axis=1).reshape(xx.shape)
            my_colors = colors_ternary if num_classes == 3 else sns.color_palette("colorblind", num_classes)
            cmap_discrete = ListedColormap(my_colors)
            levels = np.arange(num_classes+1) - 0.5
            ctf = ax.contourf(xx, yy, class_argmax, alpha=0.8,
                              cmap=cmap_discrete, levels=levels)
            cbar = fig.colorbar(ctf, ax=ax, ticks=range(num_classes))
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Predicted Class", fontsize=12, fontweight='normal')
            cbar.ax.set_yticklabels([mapping.get(c, f"Class {c}") for c in range(num_classes)])

        unique_classes = np.unique(y_test)
        for cls_ in unique_classes:
            color_list = colors_binary if num_classes == 2 else colors_ternary
            c_index = cls_ if cls_ < len(color_list) else 0
            point_color = color_list[c_index]
            marker_j = markers.get(cls_, 'o')
            label_str = mapping.get(cls_, f"Class {cls_}")
            pts = (y_test == cls_)
            ax.scatter(
                X_embedded[pts, 0], X_embedded[pts, 1],
                c=point_color, marker=marker_j,
                edgecolor='k', alpha=0.9, s=80,
                label=label_str
            )

        ax.set_title(f"{labels[i]} {subplot_titles[i]}", fontsize=12, fontweight='normal')
        ax.set_xlabel("t-SNE Component 1", fontsize=12, fontweight='normal')
        ax.set_ylabel("t-SNE Component 2", fontsize=12, fontweight='normal')
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.legend(title="Classes", fontsize=10, title_fontsize=11, loc="best")

    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight", dpi=300)
    plt.show()

##############################################################################
# 4 Subplots of AUROC with Transparent Background Distributions
##############################################################################
def plot_auroc_and_distributions(
    datasets,
    titles,
    target_col,
    class_label_map=None,
    figsize=(10, 8),
    save_path_auroc=None,
    save_path_dist=None
):
    """
    Trains a calibrated SVM on each dataset, then produces:
      1) A 2×2 figure of AUROC curves.
      2) A 2×2 figure of probability distributions.
    """
    from sklearn.calibration import CalibratedClassifierCV

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly']
    }

    if class_label_map is None:
        class_label_map = {0: "Class 0", 1: "Class 1"}

    n_datasets = len(datasets)
    if n_datasets != 4:
        print("WARNING: This function is set up for exactly 4 datasets (2×2).")

    all_fpr = []
    all_tpr = []
    all_roc_auc = []
    all_class0_vals = []
    all_class1_vals = []

    for i, (data_path, method_name) in enumerate(zip(datasets, titles)):
        df = pd.read_csv(data_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        X = df[numeric_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svc = SVC(probability=True)
        grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=0)
        grid_search.fit(X_train_scaled, y_train)
        best_svm = grid_search.best_estimator_

        calibrated_clf = CalibratedClassifierCV(estimator=best_svm, method='sigmoid', cv='prefit')
        calibrated_clf.fit(X_train_scaled, y_train)

        y_proba = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc = auc(fpr, tpr)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_roc_auc.append(roc_auc)

        class0_vals = y_proba[y_test == 0]
        class1_vals = y_proba[y_test == 1]
        all_class0_vals.append(class0_vals)
        all_class1_vals.append(class1_vals)

    # Figure 1: AUROC curves
    fig_auroc, axes_auroc = plt.subplots(2, 2, figsize=figsize, facecolor='white')
    axes_auroc = axes_auroc.ravel()
    for i in range(n_datasets):
        ax = axes_auroc[i]
        ax.plot(all_fpr[i], all_tpr[i], color='blue', linewidth=2,
                label=f"AUROC = {all_roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.6)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(titles[i], fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    fig_auroc.patch.set_edgecolor('black')
    fig_auroc.patch.set_linewidth(2)
    plt.tight_layout()
    if save_path_auroc:
        plt.savefig(save_path_auroc, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # Figure 2: Probability Distributions
    fig_dist, axes_dist = plt.subplots(2, 2, figsize=figsize, facecolor='white')
    axes_dist = axes_dist.ravel()
    for i in range(n_datasets):
        ax = axes_dist[i]
        label0 = class_label_map.get(0, "Class 0")
        label1 = class_label_map.get(1, "Class 1")
        sns.histplot(all_class0_vals[i], color='orange', label=label0,
                     kde=False, stat='density', alpha=0.6, edgecolor='black', bins=15, ax=ax)
        sns.histplot(all_class1_vals[i], color='green', label=label1,
                     kde=False, stat='density', alpha=0.6, edgecolor='black', bins=15, ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("", fontsize=10)
        ax.set_title(titles[i], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
    fig_dist.patch.set_edgecolor('black')
    fig_dist.patch.set_linewidth(2)
    plt.tight_layout()
    if save_path_dist:
        plt.savefig(save_path_dist, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
