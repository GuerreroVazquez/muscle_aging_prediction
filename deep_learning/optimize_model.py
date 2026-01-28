import numpy as np
import pyswarms as ps
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import model as m



dataset_path = 'Data/trainign_dataset_51270_microarray_adjusted.csv'
training_data=pd.read_csv(dataset_path)


def build_model(hidden_layer_configs, n_features=5127):
    """
    Build a Sequential model based on the provided hidden layer configurations
    
    :param hidden_layer_configs: List of tuples representing hidden layer configurations
    :param n_features: Number of input features
    :return: Constructed Sequential model
    """
    new_model = Sequential()

    for i, config in enumerate(hidden_layer_configs):
        # Add Dense layer
        new_model.add(Dense(config[0], activation='relu', kernel_initializer='he_normal', input_shape=(n_features,) if i == 0 else None))
        
        # Add Dropout layer
        if len(config) > 1 and i != len(hidden_layer_configs) - 1:
            new_model.add(Dropout(config[1]))  # Adding dropout to reduce overfitting

    # Add output layer
    new_model.add(Dense(1))

    return new_model


# Define the objective function
def objective_function(layer_layout):
    model=build_model(layer_layout)
    
    return np.sum(x**2)

# Set up the optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Create a swarm instance
num_particles = 10
dim = 3
optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dim, options=options)

# Perform optimization
best_cost, best_position = optimizer.optimize(objective_function, iters=100)

print("Best position:", best_position)
print("Best cost:", best_cost)