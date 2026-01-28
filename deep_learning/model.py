import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.utils import plot_model
import seaborn as sns

output_directory=""


def balance_classes(df, column_name="Age"):
    """
    Balance the classes in a dataframe by downsampling the majority classes
    
    :param df: The dataframe to balance
    :param column_name: The name of the column to balance by

    :return: Tuple containing the balanced dataframe and the dataframe with unselected rows
    """
    df['group'] = pd.cut(df[column_name], bins=[-np.inf, 35, 65, np.inf], labels=[1, 2, 3], right=False)
    min_group_size = df['group'].value_counts().min()  # Calculate the size of the smallest group
    sample_size = min_group_size  # + random.randint(-10, 10)  # Add some randomness to the sample size  Remove with stratification
    balanced_df = pd.DataFrame()
    unselected_df = pd.DataFrame()

    for value in [1, 2, 3]:
        subset = df[df['group'] == value]
        if len(subset) <= sample_size:
            balanced_df = pd.concat([balanced_df, subset])
        else:
            balanced_subset = subset.sample(sample_size)
            unselected_subset = subset.drop(balanced_subset.index)
            balanced_df = pd.concat([balanced_df, balanced_subset])
            unselected_df = pd.concat([unselected_df, unselected_subset])

    balanced_df = balanced_df.drop(columns=['group'])
    unselected_df = unselected_df.drop(columns=['group'])

    return balanced_df, unselected_df


def remove_non_numeric_columns(df: pd.DataFrame):
    """
    Remove non-numeric columns from a dataframe
    
    :param df: The dataframe to remove the columns from
    
    :return: The dataframe with the non-numeric columns removed
    """
    df = pd.DataFrame(df)
    return df.select_dtypes(include=[np.number])


def divide_test_train_dataset(balanced_df: pd.DataFrame, unselected_samples_df: pd.DataFrame = None, test_size=0.2):
    """
    Divide the dataset into a training and a test set
    
    :return: The training and test sets
    """

    X = remove_non_numeric_columns(balanced_df)
    if "Age" not in X:
        n_column = len(X.columns)
        if n_column > 10:
            some_features = list(X.columns[:3]).extend(list(X.columns[-3:]))
        else:
            some_features = list(X.columns)
        raise (f"Age NEEDS to be a column in the dataframe. The given dataframe has {n_column} columns, from which "
               f"none is 'Age', some of the columns are: {some_features}")
    X = X.drop(columns=['Age'])
    columns_a = set(X.columns)
    # Get the columns of dataframe B
    columns_b = set(balanced_df.columns)
    # Find the columns in dataframe B that are not in dataframe A
    columns_only_in_b = columns_b - columns_a
    # Convert the set to a list if needed
    columns_only_in_b = list(columns_only_in_b)  # ['Sample','Experiment', 'Age', 'Sex', 'Status']

    z = balanced_df[columns_only_in_b]
    y = balanced_df['Age']
    # divide by age groups
    y_class = pd.cut(balanced_df['Age'], bins=[-np.inf, 35, 65, np.inf], labels=[1, 2, 3], right=False)

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X, y, z, test_size=test_size, random_state=42,
                                                                         shuffle=True, stratify=y_class)

    if unselected_samples_df is not None:
        un_X = remove_non_numeric_columns(unselected_samples_df)  # unselected_samples_df
        un_y = unselected_samples_df['Age']
        y_test = pd.concat([y_test, un_y])
        X_test = pd.concat([X_test, un_X])
    return X_train, X_test, y_train, y_test, z_train, z_test


def normalize_train_data(X_train: pd.DataFrame):
    """
    Normalize the training data
    
    :param X_train: The training data
    """

    scaler = StandardScaler()
    normalize_X_train = scaler.fit_transform(X_train)
    return normalize_X_train, scaler


def normalize_train_data_row(X_train: pd.DataFrame, method: str = 'minmax'):
    """
    Normalize the training data by row
    
    :param X_train: The training data
    :param method: The normalization method, 'minmax' for Min-Max scaling or 'zscore' for Z-score scaling
    :return: Normalized training data and the scaler object
    """
    # Transpose the DataFrame
    X_train_transposed = X_train.T

    # Normalize by Min-Max scaling (scale to [0, 1])
    if method == 'minmax':
        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train_transposed)

    # Normalize by Z-score scaling (Z-scaler)
    elif method == 'zscore':
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train_transposed)

    # Transpose the normalized data back to its original shape
    X_train_normalized = X_train_normalized.T

    return X_train_normalized


def train_data(X_train, y_train, optimizer='adam', loss='mse', metrics=['mae'], epochs=1500, batch_size=50, validation_split=0.1):
    """
    Train the data
    
    :return: The trained model
    """

    n_features = X_train.shape[1]
    new_model = Sequential()

    # Add layers to the model 
    new_model.add(Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    new_model.add(Dropout(0.5))  # Adding dropout to reduce overfitting
    new_model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    new_model.add(Dropout(0.5))  # Adding dropout to reduce overfitting
    new_model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    new_model.add(Dense(1))

    # Compile the model
    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    new_model_history = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split)
    return new_model, new_model_history


def train_data_simplifided(X_train, y_train, optimizer='adam', loss='mse', metrics=['mae']):
    """
    Train the data
    
    :return: The trained model
    """
    n_features = X_train.shape[1]
    new_model = Sequential()

    # Add layers to the model
    new_model.add(Dense(32, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    new_model.add(Dropout(0.7))  # Adding dropout to reduce overfitting
    new_model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
    new_model.add(Dense(1))

    # Compile the model
    new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    new_model_history = new_model.fit(X_train, y_train, epochs=150, batch_size=50, verbose=0, validation_split=0.05,
                                      shuffle=True)
    # testing part is not integrated yet.
    return new_model, new_model_history


def print_loss(history, save: bool = False):
    """
    Print the loss of the model
    
    :param history: The history of the model
    """
    pyplot.title('Learning Curves')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    if save:
        pyplot.savefig(f'{output_directory}/history_loss.png')
    pyplot.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model
    
    :param model: The model to evaluate
    :param X_test: The test data
    :param y_test: The test labels
    """
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print('MAE: %.3f' % mae)
    print('Loss: %.3f' % loss)
    # Make predictions using the trained model
    predictions = model.predict(X_test)

    # If your target variable is a single numeric value (regression), you can use predictions directly
    # For example, if predictions is a 2D array, you can flatten it to a 1D array
    predictions_flat = np.ravel(predictions)

    results = pd.DataFrame(list(zip(y_test, predictions_flat)))
    results = results.set_axis(["Actual", "Predicted"], axis=1)

    return results


def plot_results(results: pd.DataFrame, labels: pd.DataFrame, print_label: str = 'Experiment', save: bool = False, save_output=None):
    """
    Plot the results
    """
    # Extract 'experiment' column from original data
    experiment_column = labels[print_label].values
    results[print_label] = experiment_column
    if save:
        results.to_csv(f'{output_directory}/results.csv')
    print(results)
    # Plot scatter plot with color according to 'Experiment' column in results DataFrame

    # Create scatter plot with hue='Experiment' to group by Experiment category
    sns.scatterplot(x='Actual', y='Predicted', hue=print_label, data=results, palette='Set1', alpha=0.8, edgecolor='w')

    plt.legend()
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().set_ylim(bottom=15, top=100)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    if save:
        plt.savefig(save_output)
    plt.show()

    # results.plot(kind='scatter', x='Actual', y='Predicted', s=32, alpha=.8)
    # plt.gca().spines[['top', 'right',]].set_visible(False)
    # plt.gca().set_ylim(bottom=15, top=100)
    # plt.show()


def extend_result_data(results: pd.DataFrame, z_data: pd.DataFrame):
    """
    Extend the results with the original data
    """
    results["Experiment"] = z_data["Experiment"].values
    results["Sample"] = z_data["Sample"].values
    results["Sex"] = z_data["Sex"].values
    results["Status"] = z_data["Status"].values

    results["absolute_error"] = abs(results["Actual"] - results["Predicted"])
    results["big_absolute_error"] = np.where(results["absolute_error"] < 5, 0, results["absolute_error"])
    return results


def get_feature_importance(model, X_train):
    """
    Get the feature importance of the model
    
    :param model: The model to get the feature importance of
    :param X_train: The training data
    """
    # Fit the model to the Lasso regressor

    # Get the feature importance
    feature_importance = SelectFromModel(lasso)
    feature_importance.fit(X_train, model.predict(X_train))

    return feature_importance, lasso


def save_selected_genes(genes: list, file_name: str):
    if output_directory:
        file_name = f"{output_directory}/{file_name}"
    with open(file_name, 'w') as file:
        for gene in genes:
            file.write(f'{gene}\n')


def get_results_statistics(results: pd.DataFrame):
    """
    Get the statistics of the results
    
    :param results: The results to get the statistics of
    """
    # Calculate the mean absolute error
    mae = np.mean(np.abs(results['Actual'] - results['Predicted']))

    # Calculate the mean squared error
    mse = np.mean((results['Actual'] - results['Predicted']) ** 2)

    # min and max
    min_value = results['Predicted'].min()
    max_value = results['Predicted'].max()

    print(f"mae: {mae}, mse: {mse}, min: {min_value}, max: {max_value}")


def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, alpha=0.1):
    """
    Select the most important features
    
    :param df: The dataframe to select the features from

    """
    if alpha == 1:
        return X_train, X_test, None
    lasso = Lasso(alpha=alpha)  # Adjust the alpha parameter
    selector = SelectFromModel(lasso)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector


def print_performance(history, results, z_test, save: bool = False, skip_experiment=False):
    """
    Print all the data
    """
    print_loss(history, save=save)
    if not skip_experiment:
        plot_results(results, z_test, save=save, print_label='Experiment')
        experiment_ratio = z_test[['Experiment']].value_counts()
        print(experiment_ratio)
    else:
        plot_results(results, z_test, save=save, print_label='Age')

    get_results_statistics(results)


class Model:
    def __init__(self, dataset_path: str = 'Data/trainign_dataset_51270_microarray_adjusted.csv'):
        self.coef_ = None
        self.selector_features = None
        self.scaler = None
        self.dataset_path = dataset_path
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.z_train = None
        self.z_test = None
        self.selector = None
        self.feature_importance = None
        self.selected_features = None

    def mean_square_threshold(self, y_true, y_pred, threshold=15):
        diff = abs(y_true - y_pred)
        squared_diff = tf.square(tf.where(diff < threshold, diff, tf.square(diff)))
        mean_square = tf.reduce_mean(squared_diff, axis=-1)

        return mean_square

    def feature_selection(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, alpha=0.1):
        """
        Select the most important features
        
        :param df: The dataframe to select the features from

        """
        if alpha == 1:
            return X_train, X_test
        lasso = Lasso(alpha=alpha)  # Adjust the alpha parameter
        self.selector_features = SelectFromModel(lasso)
        X_train_selected = self.selector_features.fit_transform(X_train, y_train)
        X_test_selected = self.selector_features.transform(X_test)

        return X_train_selected, X_test_selected

    def get_selected_features(self, all_genes: pd.Index):
        """
        Select the most important features
        :param selected_features: The selected features    
        """

        # let's look the genes
        selected_feature_indices = self.model_in_features

        # Get the names of the selected features
        first_filter_features = list(selected_feature_indices)
        #first_filter_features = all_genes[first_filter_features]
        return first_filter_features

    def two_step_normalization(self, data: pd.DataFrame):
        """
        Two step normalization from Ivan Izonin et al, 2022.  Two-Step Data Normalization Approach for Improving 
        Classification Accuracy in the Medical Diagnosis Domain.
        1. Max Abs Scaler
        2. L2 normalization
        3. Add norm feature
        4. L2 normalization
        5. Return the normalized dataset
        """
        dataset = data.values
        # Initialize the Max Abs Scaler (step 1)
        self.scaler = MaxAbsScaler()
        # Fit and transform the data Equation (1)
        dataset = self.scaler.fit_transform(dataset)
        # Calculate the norm of each vector in the dataset using Equation (2)
        norms = np.sqrt(np.sum(dataset ** 2, axis=1))
        # Apply Equation (3) to calculate the norm of each vector in the dataset
        dataset = np.divide(dataset, norms[:, np.newaxis])
        # add norm feature
        dataset = np.column_stack((dataset, norms))
        # let's repeat the scaler
        self.scaler_2 = StandardScaler()
        dataset = self.scaler_2.fit_transform(dataset)

        # dataset = np.delete(dataset, -1, axis=1)
        # transform back to dataframe and return
        columns = list(data.columns)
        columns.append('norm')
        return pd.DataFrame(dataset, index=data.index, columns=columns)

    def apply_two_step_normalization(self, data):
        """
        Apply the same normalization transformation to new data using the saved scaling parameters.
        """
        dataset = data.values
        if self.scaler is None:
            raise ValueError("Scaler has not been initialized. Please fit the scaler to training data first.")

        # Apply Max Abs Scaler
        scaled_data = self.scaler.transform(dataset)

        # Calculate the norm of each vector in the dataset using Equation (2)
        norms = np.sqrt(np.sum(scaled_data ** 2, axis=1))
        # Apply Equation (3) to calculate the norm of each vector in the dataset
        normalized_data = np.divide(scaled_data, norms[:, np.newaxis])
        # add norm feature
        normalized_data = np.column_stack((normalized_data, norms))
        # Calculate the norm of each vector in the dataset using Equation (2)
        norms = np.sqrt(np.sum(normalized_data ** 2, axis=1))
        # Apply Equation (3) to calculate the norm of each vector in the dataset, yes, again
        normalized_data = np.divide(normalized_data, norms[:, np.newaxis])
        normalized_data = self.scaler_2.transform(normalized_data)

        columns = list(data.columns)
        columns.append('norm')
        # transform back to dataframe and return
        return pd.DataFrame(normalized_data, index=data.index, columns=columns)

    def normalize_data(self, two_step: bool):
        columns = self.X_train.columns
        index_x = self.X_train.index
        index_y = self.X_test.index
        if two_step:
            self.X_train = self.two_step_normalization(self.X_train)
            self.X_test = self.apply_two_step_normalization(self.X_test)
        else:
            self.X_train, self.scaler = normalize_train_data(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        self.X_train = pd.DataFrame(self.X_train, columns=columns, index=index_x)
        self.X_test = pd.DataFrame(self.X_test, columns=columns, index=index_y)

    def set_selected_features(self):
        if self.selector_features is None:
            self.model_in_features = self.X_train.columns
        else:
            selected_feature_indices = self.selector_features.get_support()
            self.model_in_features = list(selected_feature_indices)
            self.model_in_features = self.X_train.columns[self.model_in_features]

    def prepare_data(self, df: pd.DataFrame, lasso_alpha=0.1, two_step=False):
        """
        Prepare the data
        this takes the original dataframe with all the features and lables
        It removes all non-numeric columns, balances the classes, and divides
          the dataset into a training and a test set.
        It also normalizes the training data and the test data prepared for futer use

        :param df: The dataframe to prepare
        :param lasso_alpha: the alpha for the lasso feature selection
        return: The normalized training data and the normalized test data

        """
        self.X_train, self.X_test, self.y_train, self.y_test, self.z_train, self.z_test = divide_test_train_dataset(df)
        self.normalize_data(two_step=two_step)
        X_train_fs, X_test_fs = self.feature_selection(self.X_train, self.y_train, X_test=self.X_test,
                                                       alpha=lasso_alpha)
        self.set_selected_features()
        return X_train_fs, X_test_fs

    def train(self, normalize_X_train, epochs=1500, optimizer='adam', loss='mse', metrics=None, validation_split=0.1,
              batch_size=50):
        if metrics is None:
            metrics = ['mae']
        self.model, self.history = train_data(normalize_X_train, self.y_train,
                                              epochs=epochs, optimizer=optimizer,
                                              loss=loss, metrics=metrics, validation_split=validation_split,
                                              batch_size=batch_size)
        # self.lasso_select, self.lasso = get_feature_importance(self.model, normalize_X_train)
        # self.feature_importance = self.lasso_select.get_support()

    def test(self, X_test, y_test):
        results = evaluate_model(self.model, X_test, y_test)
        error = self.model.evaluate(X_test, y_test, verbose=0)
        return results, error

    def run_model(self, X):

        if not self.model_in_features in X:
            raise ValueError(f"X should contain the following features: {self.model_in_features}")

        if isinstance(X, pd.DataFrame):
            X_selected_features = X[self.model_in_features]
        else:
            X_selected_features = pd.DataFrame([X], columns=self.model_in_features)
        self.scaler.transform(X_selected_features)
        predictions = self.model.predict(X_selected_features)
        predictions_flat = np.ravel(predictions)
        return predictions_flat

    def get_feature_weigths(self, final_genes: pd.Index):

        weigths = [x for x in self.lasso.coef_ if x != 0]
        df_feature_importance = pd.DataFrame({'Gene': final_genes, 'Weigth': weigths})
        return df_feature_importance.iloc[df_feature_importance['Weigth'].abs().argsort()[::-1]]


def main(dataset_path: str = 'Data/trainign_dataset_51270_microarray_adjusted.csv'):
    training_data = pd.read_csv(dataset_path)
    my_model = Model(dataset_path)
    normalized_X_train, normalized_X_test_fs = my_model.prepare_data(df=training_data)
    my_model.train(normalize_X_train=normalized_X_train,
                   epochs=1500, optimizer='adam',
                   loss='mse', metrics=['mae'])
    plot_model(my_model.model, 'model.png', show_shapes=True)
    results, error = my_model.test(normalized_X_test_fs, my_model.y_test)
    print_performance(my_model.history, results, my_model.z_test)
    print(error)
