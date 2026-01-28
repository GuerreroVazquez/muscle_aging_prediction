import model as m
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import Evaluate_results as er
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
import tensorflow as tf
import tensorflow as tf
import os
import json
from contextlib import redirect_stdout
import yaml
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_new_folder(base_path="Results"):
    # Check if the Results directory exists, if not, create it
    # get files that contain the word "result_RNAseq_SMOTE" and are directories
    results_dir = os.path.join(base_path, "Model_outputs")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Find existing result folders and determine the next number
    existing_folders = [name for name in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, name)) and "result_RNAseq_SMOTE" in name]
    if not existing_folders:
        next_number = 1
    else:
        # Sort existing folders to get the highest numbered folder
        existing_folders.sort()
        last_folder = existing_folders[-1]
        last_number = int(last_folder.replace("result_RNAseq_SMOTE", ""))
        next_number = last_number + 1

    # Generate the name for the new folder
    new_folder_name = "result_RNAseq_SMOTE{:02d}".format(next_number)
    new_folder_path = os.path.join(results_dir, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path)

    return new_folder_path


def save_performance_results(results, my_model):
    def analyze_restults(results, my_model):
        #results["Sex"] = my_model.z_test["Sex"].values
        #results["Status"] = my_model.z_test["Status"].values

        results["absolute_error"] = abs(results["Actual"]-results["Predicted"])
        results["big_absolute_error"] = np.where(results["absolute_error"] < 5, 0, results["absolute_error"])
        return results

    results = analyze_restults(results, my_model)
    with open(f"{m.output_directory}/model_summary.txt", "a") as f:
        # Redirect the stdout to the file
        with redirect_stdout(f):
            print("/n/t Actual age distribution:")
            print(results["Actual"].describe())
            print("/n/t Predicted age distribution:")
            print(results["Predicted"].describe())
            print("/n/t Predicted absolute error distribution:")
            print(results["absolute_error"].describe())
            print("/n/n/t /t Is the experiment acting the outcome?/n")
            
            #print("/n/t Error distribution absolute_error ~ Age anova_lm:")
            #print(er.evaluate_error_experiment(results, groups=["Age"]))
            #print("/n/t Krustal Wallis test:")
            #print (er.evaluate_Kruskal_Wallis(results,groups=["Age"]))
            #print("/n/t Dunnes test:")
            #print(er.evaluate_Dunnes_test(results, groups=["Age"]))
    #er.plot_Dunes_result(er.evaluate_Dunnes_test(results), save=True, file_name=f"{m.output_directory}/Dunnes_test.png")


def plot_distrobution_features(expression_df, age:list, n_features=None):
    def map_age(age):
        if age == 35:
            return 'Y'
        elif age > 65:
            return 'O'
        else:
            return 'M'
    x = expression_df
    if n_features:
        x=x.sample(n=n_features, axis='columns')
    x["Age"] = age.apply(lambda age: map_age(age))
    data = x
    melted_data = pd.melt(data,id_vars="Age",
                    var_name="features",
                    value_name='value')

    plt.figure(figsize=(10,10))
    sns.violinplot(x="features", y="value", hue="Age", data=melted_data,split=True, inner="quart")
    plt.xticks(rotation=90)


    #sns.boxplot(x="features", y="value", hue="Age", data=melted_data, showfliers=False)
    #plt.xticks(rotation=90)
    return melted_data



def map_to_category(age):
    if age < 35:
        return "Young"
    elif age > 65:
        return "Old"
    else:
        return "MiddleAge"
    
def color_feature_selction_TSNE(df):
    """
    make sure df has the expression and the column Age
    """
    feature_df = pd.DataFrame(df)
    feature_df_t = pd.DataFrame(df) 
    feature_df_t["Age"]=  [map_to_category(age) for age in df["Age"]] 
    tab20_colors = plt.cm.tab10.colors
    num_colors = len(np.unique(feature_df_t["Age"]))
    colors = [tab20_colors[i] for i in range(num_colors)]
    custom_cmap = ListedColormap(colors)
    ategory_map = {category: i for i, category in enumerate(feature_df_t["Age"])}
    category_map = {'Young': -1, 'MiddleAge': 0, 'Old': 1}
    numeric_age_order = [category_map[experiment] for experiment in feature_df_t["Age"]]
    X = feature_df.drop(columns=["Age"]).values 
    tSNE=TSNE(n_components=2)
    tSNE_result=tSNE.fit_transform(X)
    # Step 3: Visualize the clusters
    # Plot the first two principal components and color them according to the assigned clusters
    x=tSNE_result[:,0]
    y=tSNE_result[:,1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x,y, c=numeric_age_order, cmap=custom_cmap, s=50, alpha=0.5)
    plt.title('TSNE Adjusted')
    #plt.colorbar(label="Experiment", spacing ="uniform",  values=numeric_experiment_order)

    # Create legend with experiment names
    legend_handles = [plt.Line2D([0], [0], marker='o',  markerfacecolor=color, markersize=10, label=f'Experiment {i+1}') for i, color in enumerate(colors)]
    plt.legend(handles=legend_handles, labels=list(category_map.keys()), loc='best')

    #plt.legend(list(category_map.keys()))
    #plt.show()

    plt.savefig(f"{m.output_directory}/TSNE_Scatter_Plot_feature_selection.jpg")

def run_all(dataset_path, epochs=1500, optimizer='adam', loss='mse', metrics=['mae'], batch_size=50, validation_split=0.1, lasso_alpha=0.1, two_step_normalization=False):

    m.output_directory=create_new_folder()
    training_data=pd.read_csv(dataset_path)
    my_model = m.Model(training_data)
    normalized_X_train, normalized_X_test_fs = my_model.prepare_data(df=training_data, lasso_alpha=lasso_alpha, two_step=two_step_normalization)
    my_model.train(normalize_X_train=normalized_X_train, 
                    epochs=epochs, optimizer=optimizer,
                    loss=loss, metrics=metrics, batch_size=batch_size, validation_split=validation_split)
    m.plot_model(my_model.model, f'{m.output_directory}/model.png', show_shapes=True)
    results, error = my_model.test(normalized_X_test_fs, my_model.y_test)
    m.print_performance(my_model.history, results, my_model.z_test, save=True)

    conf_dic = {'dataset':dataset_path,
                'loss': loss,
                'optimizer': optimizer,
                'metrics': metrics,
                'epochs': epochs, 
                'batch_size': batch_size, 
                'validation_split': validation_split, 
                'lasso_alpha': lasso_alpha, 
                'two_step_normalization': two_step_normalization}
    with open(f"{m.output_directory}/model_summary.txt", "a") as f:
        # Redirect the stdout to the file
        with redirect_stdout(f):
            print("Error: ")
            print(error)

    save_performance_results(results, my_model)
    clean_df_z = pd.DataFrame(my_model.z_test.values)
    clean_df_z.columns = my_model.z_test.columns
    results_all_metadata = pd.concat([results,clean_df_z ], axis=1)
    with open(f"{m.output_directory}/configuration.yaml", "w") as yaml_file:
        yaml.dump(conf_dic, yaml_file, default_flow_style=False)
    with open(f"{m.output_directory}/model_architecture.yaml", "w") as yaml_file:
        yaml.dump(json.loads(my_model.model.to_json()), yaml_file, default_flow_style=False)
    with open(f"{m.output_directory}/selected_features.csv", "w") as f:
        f.write("\n".join(list(my_model.model_in_features)))

    feature_df = pd.DataFrame(normalized_X_train)
    feature_df = feature_df.dropna()
    feature_df
    feature_df["Age"]= my_model.y_train.values
    color_feature_selction_TSNE(feature_df)
    # Now check with the training data
        
        


def my_train_data(X_train, y_train, optimizer='adam', loss='mse', metrics=['mae'], epochs=1500, validation_split=0.1,
              batch_size=50):
    """
    Train the data
    
    :return: The trained model
    """

    n_features = X_train.shape[1]
    new_model = Sequential()

    # Add layers to the model 
   # new_model.add(Dense(8, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
   # new_model.add(Dense(16, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
   # new_model.add(Dropout(0.5))  # Adding dropout to reduce overfitting
    new_model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    new_model.add(Dropout(0.5))  # Adding dropout to reduce overfitting
    new_model.add(Dense(64, activation='softsign', kernel_initializer='he_normal'))
    new_model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    #new_model.add(Dropout(0.5))  # Adding dropout to reduce overfitting
    #new_model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
    #new_model.add(Dropout(0.5))
    #new_model.add(Dense(8, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    new_model.add(Dense(1))

    # Compile the model
    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    new_model_history = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split)
    return new_model, new_model_history
m.train_data=my_train_data