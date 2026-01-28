# train a generative adversarial network on a one-dimensional function
import pandas as pd
import numpy as np
from numpy import zeros
from numpy import ones
import argparse

from numpy.random import randn
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, LSTM

import deep_learning.model as m
import sys

dataset_path = ('/home/karen/Documents/GitHub/Identify_muscle_age_genes/deep_learning/Results/'
                'microarray_w_selected_genes_normalized.csv')


# create the discriminator

class GAN:
    def __init__(self, dataset_path=None, output_path=None, n_epochs=30000, latent_dim=300, n_batch=600, n_eval=200):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.n_epochs = n_epochs
        self.latent_dim = latent_dim
        self.n_batch = n_batch
        self.n_eval = n_eval
        self.training_data = pd.read_csv(dataset_path)
        self.training_data = m.remove_non_numeric_columns(self.training_data)
        self.o_real_data = np.array(self.training_data)
        self.LENGTH_INPUT = self.o_real_data.shape[1]
        #latent_dim = self.o_real_data.shape[0]
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator(latent_dim)
        self.gan_model = self.define_gan(self.generator, self.discriminator)

    # define the standalone discriminator model
    def define_discriminator(self):
        n_inputs = self.LENGTH_INPUT
        model = Sequential()
        model.add(Dense(n_inputs, activation='relu', input_dim=n_inputs))
        model.add(Dense(250, activation='relu', input_dim=n_inputs))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self, latent_dim):
        model = Sequential()
        model.add(Input(shape=(latent_dim, 1)))
        model.add(LSTM(150))
        model.add(Dense(self.LENGTH_INPUT, activation='linear'))
        model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, generator, discriminator):
        # make weights in the discriminator not trainable
        discriminator.trainable = False
        # connect them
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    # generate n real samples with class labels
    def generate_real_samples(self, n):
        amps = np.arange(0.1, 10, 0.1)
        bias = np.arange(0.1, 10, 0.1)
        freqs = np.linspace(1, 2, 1000)
        X2 = np.linspace(-5, 5, self.LENGTH_INPUT)
        X1 = []
        for x in range(n):
            noise = np.random.normal(size=len(X2))
            X1.append(
                np.random.choice(amps) * np.sin(X2 * np.random.choice(freqs)) + np.random.choice(bias) + 0.3 * noise)
        X1 = np.array(X1).reshape(n, self.LENGTH_INPUT)
        # generate class labels
        y = ones((n, 1))
        return X1, y

    def get_real_data(self, n):
        # n = int(o_real_data.shape[0] / 2)
        sample_indices = np.random.choice(self.o_real_data.shape[0], n, replace=False)
        x = self.o_real_data[sample_indices]
        y = ones((n, 1))
        return x, y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, latent_dim, n):
        # generate points in the latent space
        x_input = randn(latent_dim * n)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n, latent_dim)
        return x_input

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, latent_dim, n):
        # generate points in latent space
        x_input = self.generate_latent_points(latent_dim, n)
        # predict outputs
        X = self.generator.predict(x_input, verbose=0)
        # create class labels
        y = zeros((n, 1))
        # print(x_input)
        return X, y

    # train the generator and discriminator
    def train(self, latent_dim, n_epochs=10000, n_batch=128, n_eval=200):
        # determine half the size of one batch, for updating the discriminator
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # prepare real samples
            x_real, y_real = self.get_real_data(half_batch)
            # x_real, y_real = self.generate_real_samples(half_batch)
            # prepare fake examples
            x_fake, y_fake = self.generate_fake_samples(latent_dim, half_batch)
            # update discriminator
            self.discriminator.train_on_batch(x_real, y_real)
            self.discriminator.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = self.generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            self.gan_model.train_on_batch(x_gan, y_gan)
            # evaluate the model every n_eval epochs
            if (i + 1) % n_eval == 0:
                print(f"Currently running epoch {i} of {n_epochs}. Relax c:")

        return self.generate_fake_samples(latent_dim, latent_dim)


def main(dataset_path=None, n_epochs=30000, latent_dim=300, n_batch=600, n_eval=200, output_path=None):
    if dataset_path is None:
        print("No dataset path provided. Using default path.")
        dataset_path = '/home/karen/Documents/GitHub/Identify_muscle_age_genes/deep_learning/Results/microarray_w_selected_genes_normalized.csv'
    if output_path is None:
        output_path = "fake_microarray_data.csv"
    print("Dataset Path:", dataset_path)
    print("Number of Epochs:", n_epochs)
    print("Latent Dimension:", latent_dim)
    print("Batch Size:", n_batch)
    print("Evaluation Frequency:", n_eval)
    print("Output Path:", output_path)

    gan = GAN(dataset_path=dataset_path, output_path=output_path, latent_dim=latent_dim)
    gan.generate_real_samples = gan.get_real_data


    column_names = gan.training_data.columns
    print(f"Training data has shape {gan.training_data.shape}, with columns {column_names}.")


    # train model
    generated = gan.train(latent_dim=latent_dim, n_epochs=n_epochs, n_batch=n_batch, n_eval=n_eval)
    fake_data = generated[0]
    print(f"The fake data has size {fake_data.shape}. Proceed to save it to {output_path}.")
    pd.DataFrame(fake_data).to_csv(output_path, index=False, header=column_names)
    print("Done.")
    return fake_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--dataset_path', dest='dataset_path', default=None, help='Path to dataset')
    parser.add_argument('--n_epochs', dest='n_epochs', default=30000, type=int, help='Number of epochs')
    parser.add_argument('--latent_dim', dest='latent_dim', default=300, type=int, help='Latent dimension')
    parser.add_argument('--n_batch', dest='n_batch', default=600, type=int, help='Batch size')
    parser.add_argument('--n_eval', dest='n_eval', default=200, type=int, help='Evaluation frequency')
    parser.add_argument('--output_path', dest='output_path', default=None, help='Output path')

    args = parser.parse_args()

    # n_epochs=args.n_epochs
    # latent_dim = args.latent_dim
    # n_batch= args.n_batch
    # n_eval= args.n_eval
    main(dataset_path=args.dataset_path,
         n_epochs=args.n_epochs,
         latent_dim=args.latent_dim,
         n_batch=args.n_batch,
         n_eval=args.n_eval,
         output_path=args.output_path)
