import argparse
import feature_selection as fs
import pandas as pd
import model as m
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feature_selection as fs
import model as m
import yaml


def read_configuration(file_path):
    with open(file_path, 'r') as file:
        configuration = yaml.safe_load(file)
    return configuration

# Read configuration from configuration.yaml
configuration = read_configuration('configuraton.yaml') 

class DivideDataset:
    def __init__(self, input_file, output_folder):

        self.input_file = input_file
        self.base_name = input_file.split("/")[-1].split(".")[0]
        self.output_folder = output_folder
        if self.output_folder[-1] == "/":
            self.output_folder = self.output_folder[:-1]
        self.age_ranges_list= configuration["age_ranges_list"]
        self.standar_age_range= configuration["standar_age_range"]
        self.test_size = configuration["test_size"]
        self.min_samples_for_smote = configuration["min_samples_for_smote"]

    def divide_test_train(self, input_file):
        self.all_rna_seq_data = pd.read_csv(input_file)
        assert ("Age" in self.all_rna_seq_data.columns), "Age column not found in the input file."
        seq_X_train, seq_X_test, seq_y_train, _, seq_z_train, seq_z_test = m.divide_test_train_dataset(self.all_rna_seq_data, test_size=self.test_size)
        training_df = pd.concat([seq_X_train,  seq_z_train], axis=1)
        test_df = pd.concat([seq_X_test,  seq_z_test], axis=1)
        training_df.to_csv(f"{self.output_folder}/{self.base_name}_train", index=False)
        test_df.to_csv(f"{self.output_folder}/{self.base_name}_test", index=False)
        return seq_X_train, seq_y_train

    def apply_smote(self, seq_X_train, seq_y_train):
        dataset = pd.DataFrame(seq_X_train)
        dataset["Age"]=seq_y_train
        train_age_ranges = fs.get_age_distribution(seq_y_train, age_ranges=self.age_ranges_list)
        dataset["Age"] = self.all_rna_seq_data['Age'].apply(lambda x: fs.assign_age_group(x, train_age_ranges))

        age_group_counts = dataset["Age"].value_counts()
        not_enough_categories = age_group_counts[age_group_counts < self.min_samples_for_smote].index.tolist()
        # Separate the data into enough and not enough samples
        not_enough = dataset[dataset["Age"].isin(not_enough_categories)]
        dataset = dataset[~dataset["Age"].isin(not_enough_categories)]
        result = fs.get_feature_SMOTE(dataset, is_categorical=True, k=self.min_samples_for_smote)
        age_group = result[1]
        data = result[0]
        data["Age"] = age_group
        data = pd.concat([data, not_enough], axis=0)
        data.to_csv(f"{self.output_folder}/{self.base_name}_smote_{len(self.age_ranges_list)}_classes.csv", index=False)




def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a file.")

    # Add arguments for the input and output files
    parser.add_argument('-i', '--input_file', type=str, required=True, help="The path to the input file.")
    parser.add_argument('-o', '--output_folder', type=str, required=True, help="The path to the output file.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Create the DivideDataset object
    divide_dataset = DivideDataset(args.input_file, args.output_folder)
    seq_X_train, seq_y_train = divide_dataset.divide_test_train(args.input_file)
    divide_dataset.apply_smote(seq_X_train, seq_y_train)
    
    print(f"The dataset has been divided and saved to the output folder {args.output_folder}.")

if __name__ == "__main__":
    main()
