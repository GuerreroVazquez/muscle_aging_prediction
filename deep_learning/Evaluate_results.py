import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_results(results: pd.DataFrame, formula='absolute_error ~ Experiment'):
    model = ols(formula, data=results).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    return anova_table


def evaluate_error_experiment(results: pd.DataFrame, save=False, file_name="error_experiment.png"):
    return evaluate_results(results, formula='absolute_error ~ Experiment')


def evaluate_error_experiment(results: pd.DataFrame, groups="Experiment", value="absolute_error"):
    return evaluate_results(results, formula=f'{value} ~ {groups}')


def evaluate_Kruskal_Wallis(results: pd.DataFrame, groups="Experiment", value="absolute_error"):
    # Group absolute_error by Experiment
    experiment_groups = [group[value] for name, group in results.groupby(groups)]

    # Perform Kruskal-Wallis H-test
    h_statistic, p_value = kruskal(*experiment_groups)

    return h_statistic, p_value, f"h statistic: {h_statistic} p-value: {p_value}"


def evaluate_Dunnes_test(results: pd.DataFrame, groups="Experiment", value="absolute_error", p_adjust='holm'):
    # Perform Dunn test as post-hoc analysis
    dunn_results = sp.posthoc_dunn(results, val_col=value, group_col=groups, p_adjust=p_adjust)
    return dunn_results


def print_boxplot(results: pd.DataFrame, groups="Experiment", value="absolute_error", save=False, file_name="boxplot.png"):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=results, x=groups, y=value)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Set title and labels
    plt.title(f"Distribution of {value} by {groups}")
    plt.xlabel(groups)
    plt.ylabel(value)
    plt.axhline(y=10, color='r', linestyle='--')
    if save:
        plt.savefig(file_name)
    # Show plot
    plt.show()


def print_barplot(results: pd.DataFrame, groups="Experiment"):
    # plt.hist(training_data["Age"], bins=20,orientation='vertical')
    plt.bar(results[groups].value_counts().index, results[groups].value_counts())
    plt.xticks(rotation=45)


def plot_Dunes_result(dunn_results, groups="Experiment", save=False, file_name="dunn_test.png"):
    # Assuming dunn_results contains the results of the Dunn test

    # Generate heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(dunn_results, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True)

    # Set title and labels
    plt.title("Dunn Test Results - Adjusted p-values")
    plt.xlabel(groups)
    plt.ylabel(groups)
    if save:
        plt.savefig(file_name)
    # Show plot
    plt.show()


def evaluate_group_all_test(results, save=False, path=""):
    evaluate_error_experiment(results)
    evaluate_Kruskal_Wallis(results)
    evaluate_Dunnes_test(results)
    print_boxplot(results)
