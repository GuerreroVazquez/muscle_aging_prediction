from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import model as m
import seaborn as sns
import divide_bins as divide
import numpy as np



def ridge_L2(X,y,z=None, alpha=1.0, test_size=0.2):

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y,z, test_size=test_size, random_state=42)
    # Initialize Ridge regression model
    ridge = Ridge(alpha=alpha)  # You can adjust the alpha parameter as needed

    # Train the model
    ridge.fit(X_train, y_train)    
    y_pred = ridge.predict(X_test)

    

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE from MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    result_metrics = {"Method": "Ridge Regression",  "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Group': z_test})
    return results, result_metrics, ridge

def eval_model(X_test, y_test, z_test, ridge_model):
    "Once we have the fit model, this is to easilly evaluate the metrics for regression"
    y_pred = ridge_model.predict(X_test)

    

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE from MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    result_metrics = {"Method": "Ridge Regression",  "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Group': z_test})
    return results, result_metrics


def lasso_L1(X, y,z=None, alpha=1.0):
    
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y,z, test_size=0.2, random_state=42)

    # Initialize Lasso regression model
    lasso = Lasso(alpha=alpha)  # You can adjust the alpha parameter as needed

    # Train the model
    lasso.fit(X_train, y_train)

    
    y_pred = lasso.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE from MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    result_metrics = {"Method": "LASSO Regression",  "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Group': z_test})
    return results, result_metrics, lasso

def random_forest(X, y,z=None, n_bootstrap=100):

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y,z, test_size=0.2, random_state=42)    # Initialize list to store feature importance
    feature_importance = [0] * X_train.shape[1]

    # Perform feature selection using bootstrapping
    for _ in range(n_bootstrap):
        # Create a bootstrap sample
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=42)

        # Train a random forest classifier on the bootstrap sample
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_boot, y_boot)

        # Aggregate feature importance
        feature_importance += clf.feature_importances_

    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE from MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    result_metrics = {"Method": "LASSO Regression",  "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Group': z_test})
    return results, result_metrics

def eval_regression_file(file_name, method="L2"):
    assert method in ["L1", "L2"]
    f = file_name
    eval={"File": f}
    if "micro" in f:
        eval["Technology"]="Microarray"
    else:
        eval["Technology"]= "RNAseq"
    
    if "z" in f:
        eval["Normalization"]= "Z-score"
    else:
        eval["Normalization"]="Two-step"
    
    if "reCombat" in f:
        eval["BatchCorrection"]= "reCombat"
    else:
        eval["BatchCorrection"]="Combat"
        
    dataset = pd.read_csv(file_name)
    section_df = dataset.drop(columns=["Age"])
    section_df_age = dataset["Age"]
    section_df = m.remove_non_numeric_columns(section_df)
    if method == "L2":
        result_df, result_metrics, ridge = ridge_L2(X=section_df, y=section_df_age, z=dataset["Experiment"])
    elif method == "L1":
        result_df, result_metrics = lasso_L1(X=section_df, y=section_df_age, z=dataset["Experiment"])
    age_ranges = divide.get_age_distribution(result_df["Actual"], age_ranges=[18, 25, 35, 45,50,55,60,70, 100])
    y_train_cat = result_df["Actual"].apply(lambda x: divide.assign_age_group(x, age_ranges))
    y_test_cat = result_df["Predicted"].apply(lambda x: divide.assign_age_group(x, age_ranges))

    cat_results = pd.DataFrame({"Actual": y_train_cat, "Predicted": y_test_cat})
    mcc = matthews_corrcoef(cat_results["Actual"], cat_results["Predicted"])
    result_metrics["MCC"]= mcc
    eval.update(result_metrics)

    return eval, result_df

def plot_results(results=None, X_test=None, y_test=None, z_test=None, y_pred=None):
    # Plotting the regression line
    if results is None:
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Group': z_test})
    sns.scatterplot(x='Actual', y='Predicted', hue='Group', data=results, palette='Set1', alpha=0.8, edgecolor='w')

    #plt.scatter(y_test, ridge.predict(X_test), hue=z, label='Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Ridge Regression')
    plt.legend()
    plt.show()
