import os
import feature_selection as fs
import pandas as pd
import model as m
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_age_distribution(ages, q_ranges=None, age_ranges=None):

    # Calculate quartiles
    if q_ranges:
        quartiles = np.percentile(ages, q_ranges)
        print(quartiles)
        # Define age groups based on quartiles
        quartile_ranges = [18] + quartiles.tolist() + [90]

        # Define age groups based on quartiles
    if age_ranges:
        quartile_ranges = age_ranges
    
    if age_ranges is None and q_ranges is None:
        raise ValueError('Either age_ranges or quartile_ranges must be provided')
    
    age_groups = [{'label': f'{int(quartile_ranges[i])}-{int(quartile_ranges[i+1])}', 
                    'range': (quartile_ranges[i], quartile_ranges[i+1]),
                    'count' : len([age for age in ages if (age > quartile_ranges[i]) and (age <= quartile_ranges[i+1])])
                    } for i in range(len(quartile_ranges)-1)]
    return age_groups
# Function to assign age group labels
def assign_age_group(age, age_groups):
    for group in age_groups:
        if group['range'][0] <= age < group['range'][1]:
            return group['label']
    return 'Unknown'

def get_feature_from_class(dataset, alpha=0.1):
    section_df = dataset.drop(columns=["Age"])
    section_df_age = dataset["Age"]
    section_df = m.remove_non_numeric_columns(section_df)
    features, coef = fs.ridge_L2(X=section_df, y=section_df_age, alpha=alpha)
    result = pd.DataFrame(list(features), coef, columns=["coef"])
    result["abs_coef"]=result["coef"].abs()
    result.sort_values(ascending=False, inplace=True, by="abs_coef")
    result
    return result

def get_feature_randomforest(dataset, n_bootstrap=100):
    section_df = dataset.drop(columns=["Age"])
    #section_df_age = dataset["Age"]
    # Define the category labels
    is_numeric = dataset["Age"].dtype.name in ["int64", "float64"]


    if is_numeric:
        labels = ['Young', 'MiddleAge', 'Old']
        # Define the age ranges for each category
        bins = [18, 35, 65, float('inf')]
        # Map the age numeric value to category label
        section_df_age = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    else:
        section_df_age = dataset["Age"]
    
    

    # Print the updated dataset
    section_df = m.remove_non_numeric_columns(section_df)
    coef = fs.random_forest(X=section_df, y=section_df_age, n_bootstrap=n_bootstrap)
    features = section_df.columns
    result = pd.DataFrame(list(features), coef, columns=["coef"])
    result.sort_values(ascending=False, inplace=True, by="coef")
    result
    return result


def get_feature_SMOTE(dataset, alpha=0.1, is_categorical=False, k=5):
    section_df = dataset.drop(columns=["Age"])
    if not is_categorical:
        labels = ['Young', 'MiddleAge', 'Old']
        # Define the age ranges for each category
        bins = [18, 35, 65, float('inf')]
        # Map the age numeric value to category label
        section_df_age = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    else:
        section_df_age = dataset["Age"]
    #section_df_age = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    section_df = m.remove_non_numeric_columns(section_df)
    features, coef = fs.smote(X=section_df, y=section_df_age,k=k)
    #result = pd.DataFrame(list(features), coef, columns=["coef"])
    ##result["abs_coef"]=result["coef"].abs()
    #result.sort_values(ascending=False, inplace=True, by="abs_coef")
    #result
    return features, coef


def get_feature_catBoost(dataset, is_categorical=True):
    section_df = dataset.drop(columns=["Age"])
    if not is_categorical:
        labels = ['Young', 'MiddleAge', 'Old']
        # Define the age ranges for each category
        bins = [18, 35, 65, float('inf')]
        # Map the age numeric value to category label
        section_df_age = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    else:
        section_df_age = dataset["Age"]
    #section_df_age = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    section_df = m.remove_non_numeric_columns(section_df)
    features, coef = fs.catBoost(X=section_df, y=section_df_age)
    #result = pd.DataFrame(list(features), coef, columns=["coef"])
    ##result["abs_coef"]=result["coef"].abs()
    #result.sort_values(ascending=False, inplace=True, by="abs_coef")
    #result
    result = pd.DataFrame(list(features), coef, columns=["coef"])
    result.sort_values(ascending=False, inplace=True, by="coef")
    result
    return result


def plot_fake_data(result):
    # Assuming your DataFrame is named df
    # Perform dimensionality reduction
    X = result.drop(columns=["Age", "Class"]).values  # Extracting only numeric columns
    y_age = result["Age"].values
    y_class = result["Class"].values

    # Dimensionality reduction with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Dimensionality reduction with t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=30)
    X_tsne = tsne.fit_transform(X)
    color_map = {'Young': 'blue', 'MiddleAge': 'green', 'Old': 'red'}

    # Plotting
    plt.figure(figsize=(14, 7))

    # Plotting PCA
    plt.subplot(1, 2, 1)
    for i, age in enumerate(y_age):
        if y_class[i] == 'modified':
            marker = 'o'  # Circle for fake
        else:
            marker = 's'  # Square for real
        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=color_map[age], marker=marker)
    plt.title('PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Plotting t-SNE
    plt.subplot(1, 2, 2)
    for i, age in enumerate(y_age):
        if y_class[i] == 'modified':
            marker = 'o'  # Circle for fake
        else:
            marker = 's'  # Square for real
        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=color_map[age], marker=marker)
    plt.title('t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    plt.show()


