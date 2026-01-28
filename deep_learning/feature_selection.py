### feature selections
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import model as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_feature_from_class(dataset, alpha=0.1):
    section_df = dataset.drop(columns=["Age"])
    section_df_age = dataset["Age"]
    section_df = m.remove_non_numeric_columns(section_df)
    features, coef = ridge_L2(X=section_df, y=section_df_age, alpha=alpha)
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
    coef = random_forest(X=section_df, y=section_df_age, n_bootstrap=n_bootstrap)
    features = section_df.columns
    result = pd.DataFrame(list(features), coef, columns=["coef"])
    result.sort_values(ascending=False, inplace=True, by="coef")
    result
    return result

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
                    'count' : len([age for age in ages if (age >= quartile_ranges[i]) and (age < quartile_ranges[i+1])])
                    } for i in range(len(quartile_ranges)-1)]
    return age_groups
# Function to assign age group labels
def assign_age_group(age, age_groups):
    for group in age_groups:
        if group['range'][0] <= age < group['range'][1]:
            return group['label']
    return 'Unknown'

def get_feature_SMOTE(dataset, alpha=0.1, is_categorical=False, k=5, strategy='auto'):
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
    features, coef = smote(X=section_df, y=section_df_age, k=k, strategy=strategy)
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
    features, coef = catBoost(X=section_df, y=section_df_age)
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



def ridge_L2(X,y, alpha=1.0):
    # Generate some sample data (you can replace this with your own data)
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize Ridge regression model
    ridge = Ridge(alpha=alpha)  # You can adjust the alpha parameter as needed

    # Train the model
    ridge.fit(X_scaled, y)
    coefficients = ridge.coef_
    features = X.columns
    # Use SelectFromModel to select features based on coefficients
    #sfm = SelectFromModel(estimator=ridge, threshold='median')
    #sfm.fit(X_scaled, y)

    # Get selected feature indices
    #selected_feature_indices = sfm.get_support(indices=True)
    #selected_feature_coefficients = ridge.coef_[selected_feature_indices]

    # Print selected feature indices
    #print("Selected Feature Indices:", selected_feature_indices)
    return coefficients, features




def smote(X, y, k=5, strategy='auto'):
    # Count class distribution before SMOTE
    print("Class distribution before SMOTE:", Counter(y))

    # Initialize SMOTE
    smote = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=strategy)

    # Perform SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Count class distribution after SMOTE
    print("Class distribution after SMOTE:", Counter(y_resampled))
    return X_resampled, y_resampled


def catBoost(X, y):
    # Initialize CatBoostClassifier
    regressor = CatBoostRegressor(iterations=100,  # Number of boosting iterations
                                learning_rate=0.1,  # Learning rate
                                depth=6,  # Depth of trees
                                random_state=42,  # Random seed for reproducibility
                                verbose=0)  # Verbosity level (0 - silent, 1 - verbose)

    # Fit the model
    regressor.fit(X, y)
    coefficients = regressor.get_feature_importance()
    features = X.columns


    return coefficients, features




def random_forest(X, y, n_bootstrap=100):
    # Initialize list to store feature importance
    feature_importance = [0] * X.shape[1]

    # Perform feature selection using bootstrapping
    for _ in range(n_bootstrap):
        # Create a bootstrap sample
        X_boot, y_boot = resample(X, y, replace=True, random_state=42)

        # Train a random forest classifier on the bootstrap sample
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_boot, y_boot)

        # Aggregate feature importance
        feature_importance += clf.feature_importances_


    # Select features based on importance threshold
    threshold = 0.05
    selected_features = [i for i, importance in enumerate(feature_importance) if importance > threshold]

    print("Selected Features:", selected_features)
    return feature_importance

