import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def perform_kruskal_dunn_test(data_dict, test_type="variance"):
    """
    Perform Kruskal-Wallis test and Dunn's pairwise comparison on categorical data.
    
    Parameters:
    - data_dict (dict): A dictionary where keys are category names and values are lists of values (either variances or entropies).
    - test_type (str): Specifies the type of test, either "variance" or "entropy". Default is "variance".
    
    Returns:
    - kruskal_results (dict): A dictionary with the Kruskal-Wallis statistic and p-value.
    - dunn_results (DataFrame): A DataFrame with Dunn's test pairwise comparison results (only if Kruskal-Wallis is significant).
    """
    # Group data by categories
    categories = list(data_dict.keys())
    grouped_data = [data_dict[cat] for cat in categories]
    
    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(*grouped_data)
    
    # Prepare Kruskal-Wallis results
    kruskal_results = {
        'statistic': stat,
        'p_value': p_value
    }
    
    # Print Kruskal-Wallis test results
    print(f"\nKruskal-Wallis H-test for {test_type} equality:")
    print(f"Test Statistic (H): {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"The result is statistically significant (p < 0.05). This suggests that at least one category has a different {test_type}.")
    else:
        print(f"The result is not statistically significant (p >= 0.05). This suggests that the {test_type}s across categories are similar.")
    
    # Perform Dunn's test for pairwise comparisons if Kruskal-Wallis test is significant
    if p_value < 0.05:
        print(f"\nSince the Kruskal-Wallis test is significant, Dunn's test for {test_type} can be performed.")
        
        # Create a DataFrame for Dunn's test since the method expects a DataFrame
        data = {
            'Value': [value for values in data_dict.values() for value in values],
            'Category': [cat for cat in data_dict for _ in data_dict[cat]]
        }
        dunn_df = pd.DataFrame(data)
        
        # Perform Dunn's test
        dunn_results = sp.posthoc_dunn(dunn_df, val_col='Value', group_col='Category')
        
        # Print Dunn's test results
        print("\nDunn's test pairwise comparisons (p-values):")
        
        # Only compare the upper triangle of the matrix (avoid redundant comparisons)
        for i in range(len(dunn_results.columns)):
            for j in range(i+1, len(dunn_results.columns)):
                category_1 = dunn_results.columns[i]
                category_2 = dunn_results.columns[j]
                p_val = dunn_results.loc[category_1, category_2]
                if p_val < 0.05:
                    print(f"Comparison between {category_1} and {category_2}: p-value = {p_val:.4f} (significant difference)")
                else:
                    print(f"Comparison between {category_1} and {category_2}: p-value = {p_val:.4f} (no significant difference)")
        
        return kruskal_results, dunn_results
    else:
        return kruskal_results, None
    

def plot_boxplot(data_dict, test_type="variance", output_dir=None, dpi=100):
    """
    Plots a boxplot for the provided data dictionary (either variance or entropy).
    
    Parameters:
    - data_dict (dict): A dictionary where keys are category names and values are lists of values (either variances or entropies).
    - test_type (str): Specifies the type of data ("variance" or "entropy"). This is used for the plot's title and label.
    - output_dir (str): Directory to save plot. If None, plots are not saved.
    - dpi (int): Dots per inch for the figure resolution. Defaults to 100.
    """
    # Convert dictionary to a DataFrame for seaborn
    categories = []
    values = []
    for category, points in data_dict.items():
        categories.extend([category] * len(points))
        values.extend(points)

    # Create the DataFrame
    data_df = pd.DataFrame({"Category": categories, "Value": values})

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Category", y="Value", data=data_df, showfliers=False, hue="Category", palette="Set2", legend=False) 

    # Overlay all data points
    sns.stripplot(x="Category", y="Value", data=data_df, size=1.5, alpha=0.5, color="gray", jitter=True)

    # Add title and labels
    plt.title(f"{test_type.capitalize()} (Epistemic Uncertainty)" if test_type == "variance" else f"{test_type.capitalize()} distribution")
    plt.ylabel(test_type.capitalize())
    plt.xlabel("Category")
    if output_dir:
        plt.savefig(f"{output_dir}/boxplot_{test_type}.png", dpi=dpi)
        plt.savefig(f"{output_dir}/boxplot_{test_type}.svg")
    plt.show()


def evaluate_test_performance(true_labels_dict, pred_labels_dict, test_keys, class_to_label, output_dir=None, dpi=100):
    """
    Evaluate aggregate performance metrics for test datasets, including combined confusion matrix.

    Parameters:
    - true_labels_dict (dict): Dictionary of true labels for all datasets.
    - pred_labels_dict (dict): Dictionary of predicted labels for all datasets.
    - test_keys (list): List of keys indicating test datasets.
    - class_to_label (dict): Dictionary mapping class names to label integers.
    - output_dir (str): Directory to save confusion matrix plot. If None, plots are not saved.
    - dpi (int): Dots per inch for the figure resolution. Defaults to 100.

    Returns:
    - aggregate_metrics (dict): Dictionary containing aggregate performance metrics for all test datasets.
    """
    aggregate_metrics = {}
    combined_true_labels = []
    combined_pred_labels = []

    # Ensure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Collect all true and predicted labels
    for key in test_keys:
        true_labels = true_labels_dict[key]
        pred_labels = pred_labels_dict[key]

        # Update combined labels
        combined_true_labels.extend(true_labels)
        combined_pred_labels.extend(pred_labels)

    # Aggregate metrics for the entire test dataset
    combined_unique_labels = np.unique(np.concatenate([combined_true_labels, combined_pred_labels]))
    combined_label_names = [k for k, v in class_to_label.items() if v in combined_unique_labels]

    # Compute aggregate metrics
    accuracy = accuracy_score(combined_true_labels, combined_pred_labels)
    precision = precision_score(combined_true_labels, combined_pred_labels, labels=combined_unique_labels, average="weighted", zero_division=0)
    recall = recall_score(combined_true_labels, combined_pred_labels, labels=combined_unique_labels, average="weighted", zero_division=0)
    f1 = f1_score(combined_true_labels, combined_pred_labels, labels=combined_unique_labels, average="weighted", zero_division=0)
    conf_matrix_combined = confusion_matrix(combined_true_labels, combined_pred_labels, labels=combined_unique_labels)

    # Store aggregate metrics
    aggregate_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Confusion Matrix": conf_matrix_combined,
    }

    # Plot combined confusion matrix for all data
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_combined, annot=True, fmt="d", cmap="Blues",
                xticklabels=combined_label_names,
                yticklabels=combined_label_names)
    plt.title("Confusion Matrix for All Test Data")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    if output_dir:
        plt.savefig(f"{output_dir}/combined_confusion_matrix.png", dpi=dpi)
        plt.savefig(f"{output_dir}/combined_confusion_matrix.svg")
    plt.show()

    # Print aggregate metrics
    print("\nAggregate Metrics for Test Data:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")

    return aggregate_metrics


def generate_ood_classification_report(pred_labels_dict, ood_keys, class_to_label):
    """
    Analyze and report the distribution of predicted classes for OOD data.

    Parameters:
    - pred_labels_dict (dict): Dictionary of predicted labels for all datasets.
    - ood_keys (list): List of keys indicating OOD datasets.
    - class_to_label (dict): Dictionary mapping class names to label integers.

    Returns:
    - ood_predictions (dict): Dictionary containing prediction counts for each class in OOD datasets.
    """
    ood_predictions = {}
    label_to_class = {v: k for k, v in class_to_label.items()}

    for key in ood_keys:
        pred_labels = pred_labels_dict[key]

        # Count predictions for each class
        unique, counts = np.unique(pred_labels, return_counts=True)
        prediction_counts = {label_to_class[label]: count for label, count in zip(unique, counts)}

        # Store predictions
        ood_predictions[key] = prediction_counts

        # Print report
        print(f"\nPrediction Counts for OOD Data ({key}):")
        for cls, count in prediction_counts.items():
            print(f"  {cls}: {count}")

    return ood_predictions


def plot_histograms(data_dict, data_type, dpi=100, output_dir='plots', auto_xlim=True):
    """
    Plots individual histograms for each category in the provided data dictionary with optional global x-axis limits.
    
    Parameters:
    - data_dict (dict): A dictionary where keys are category names and values are lists of values (variances or entropies).
    - data_type (str): Type of data to plot ("variance" or "entropy").
    - dpi (int): Dots per inch for the figure resolution. Defaults to 100.
    - output_dir (str): Directory to save the plots as png files. If None, saves to the current directory.
    - auto_xlim (bool): If True, calculates global x-axis limits from the data. Defaults to True.
    """
    if output_dir is None:
        output_dir = os.getcwd()  # Default to the current directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Calculate global x-axis limits if auto_xlim is enabled
    if auto_xlim:
        all_values = np.concatenate(list(data_dict.values()))
        xlim = (all_values.min(), all_values.max())
    else:
        xlim = None

    for name, values in data_dict.items():
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=30, alpha=0.75, color='blue', edgecolor='black')
        plt.title(f'Prediction {data_type.capitalize()} Histogram: {name}')
        plt.xlabel(data_type.capitalize())
        plt.ylabel('Frequency')
        if xlim:
            plt.xlim(xlim)  # Apply shared x-axis scale
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        if output_dir:
            output_file = os.path.join(output_dir, f'{name}_{data_type}_histogram.png')
            plt.savefig(output_file, dpi=dpi)
            print(f"Saved histogram for {name} to {output_file}")
        plt.show()