import os
import argparse
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
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
import regression as reg
import pickle
import yaml
import json


def read_configuration(file_path):
    with open(file_path, 'r') as file:
        configuration = yaml.safe_load(file)
    return configuration

# Read configuration from configuration.yaml
configuration = read_configuration('configuraton.yaml') 

class run_eval_regression:
    def __init__(self, input_file,  output_folder, method="L2") -> None:
        self.eval={"File": input_file}
        self.file = input_file
        self.base_name = input_file.split("/")[-1].split(".")[0]
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if self.output_folder[-1] == "/":
            self.output_folder = self.output_folder[:-1]
        self.age_ranges_list= configuration["age_ranges_list"]
        self.test_size = configuration["test_size"]
        self.define_technology()

    def define_technology(self):
        if "micro" in self.file:
            self.eval["Technology"]="Microarray"
        else:
            self.eval["Technology"]= "RNAseq"
        
        if "z" in self.file:
            self.eval["Normalization"]= "Z-score"
        else:
            self.eval["Normalization"]="Two-step"
        
        if "reCombat" in self.file:
            self.eval["BatchCorrection"]= "reCombat"
        else:
            self.eval["BatchCorrection"]="Combat"

    def eval_regression_file(self, method="L2"):
        assert method in ["L1", "L2"]
                    
        dataset = pd.read_csv(self.file)
        assert ("Age" in dataset.columns), "Age column not found in the input file."
        assert ("Experiment" in dataset.columns), "Experiment column not found in the input file."

        section_df = dataset.drop(columns=["Age"])
        section_df_age = dataset["Age"]
        section_df = m.remove_non_numeric_columns(section_df)
        if method == "L2":
            result_df, result_metrics, regression_model = reg.ridge_L2(X=section_df, y=section_df_age, z=dataset["Experiment"], test_size=self.test_size)
        elif method == "L1":
            result_df, result_metrics, regression_model = reg.lasso_L1(X=section_df, y=section_df_age, z=dataset["Experiment"], test_size=self.test_size)

        # save model
        model_file = f"{self.output_folder}/{self.base_name}_model.pkl"
        with open(model_file, 'wb') as file:
            pickle.dump(regression_model, file)
        

        age_ranges = divide.get_age_distribution(result_df["Actual"], age_ranges=self.age_ranges_list)
        y_train_cat = result_df["Actual"].apply(lambda x: divide.assign_age_group(x, age_ranges))
        y_test_cat = result_df["Predicted"].apply(lambda x: divide.assign_age_group(x, age_ranges))

        cat_results = pd.DataFrame({"Actual": y_train_cat, "Predicted": y_test_cat})
        mcc = matthews_corrcoef(cat_results["Actual"], cat_results["Predicted"])
        result_metrics["MCC"]= mcc
        # area under the curve
        auc = accuracy_score(cat_results["Actual"], cat_results["Predicted"])
        result_metrics["AUC"] = auc
        self.eval.update(result_metrics)
        #save eval and results_df
        with open(f"{self.output_folder}/{self.base_name}_regression_evaluation.jsn", 'w') as fp:
            json.dump(self.eval, fp)
        result_df.to_csv(f"{self.output_folder}/{self.base_name}_regression_results.csv", index=False)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a file.")

    # Add arguments for the input and output files
    parser.add_argument('-i', '--input_file', type=str, required=True, help="The path to the input file.")
    parser.add_argument('-o', '--output_folder', type=str, required=True, help="The path to the output file.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create an instance of the run_eval_regression class
    run_eval = run_eval_regression(args.input_file, args.output_folder)
    run_eval.eval_regression_file()


    
    print(f"The dataset has been divided and saved to the output folder {args.output_folder}.")

if __name__ == "__main__":
    main()
