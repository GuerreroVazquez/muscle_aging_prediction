import numpy as np
import pandas as pd
import pytest
import deep_learning.GAN as g
import math
from umap import UMAP
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

test_data_path = "test_data/"
rnaseq_dataset_path = test_data_path + "rnaseq_expr_abund.csv"

microarray_dataset_path = test_data_path + "microarray_lum.csv"


def get_data():
    data = {
        'NM_001091': [93.629994, -27.212588, -67.652035, -60.991420, -24.714860, -51.595196, 53.785246, -47.670191],
        'NM_033128': [260.535084, -705.217072, -665.770861, -633.805821, 183.682958, -315.515680, -94.480865,
                      -21.709424],
        'BC010094': [-57.238207, -7.066575, -38.439401, 20.322403, -130.814961, -124.092210, 778.997113, 41.611098],
        'NM_003468': [320.518072, -203.487417, -247.154541, -266.665389, -206.274682, -197.912895, -221.604627,
                      -186.299298],
        'BC039525': [14.139142, -131.217350, -139.895350, 65.277341, 61.868132, 173.132486, 70.546140, 77.364559],
        'BC033791': [-230.275607, 142.290881, 226.167062, 310.043292, -333.007724, -113.239020, 569.474403, -44.317485],

    }
    y = [47.97, 50.60, 50.82, 50.15, 51.35, 50.91, 51.00, 51.68]
    return pd.DataFrame(data), pd.DataFrame(y)


def test_main():
    dataset_path = rnaseq_dataset_path
    n_epochs = 30
    latent_dim = 14
    n_batch = 14
    n_eval = 200
    output_path = "fake_test_microarray_data.csv"

    data0 = g.main(dataset_path=dataset_path, n_epochs=1, latent_dim=latent_dim, n_batch=n_batch, n_eval=n_eval,
                   output_path=output_path)
    data1 = g.main(dataset_path=dataset_path, n_epochs=n_epochs, latent_dim=latent_dim, n_batch=n_batch, n_eval=n_eval,
                   output_path=output_path)
    data2 = g.main(dataset_path=dataset_path, n_epochs=n_epochs * 2, latent_dim=latent_dim, n_batch=n_batch,
                   n_eval=n_eval,
                   output_path=output_path)
    data_real = pd.read_csv(dataset_path)
    # Step 1: Compute UMAP embeddings
    data0 = data0[:, -10:]
    data1 = data1[:, -10:]
    data2 = data2[:, -10:]
    data_real.drop(columns=["Experiment", "Sample"], inplace=True)
    data_real = data_real.iloc[:, -10:]
    umap_model = UMAP(n_components=2)
    embedding0 = umap_model.fit_transform(data0)
    embedding1 = umap_model.fit_transform(data1)
    embedding2 = umap_model.transform(data2)
    embeddingReal = umap_model.transform(data_real)

    # Step 2: Normalize data
    scaler = StandardScaler()
    embedding0_normalized = scaler.fit_transform(embedding0)
    embedding1_normalized = scaler.fit_transform(embedding1)
    embedding2_normalized = scaler.transform(embedding2)
    embeddingReal_normalized = scaler.transform(embeddingReal)
    # Step 3: Compute distances
    distances_0_2 = distance.cdist(embedding0_normalized, embedding2_normalized, 'euclidean')

    distances_1_2 = distance.cdist(embedding1_normalized, embedding2_normalized, 'euclidean')

    distances_0_Real = distance.cdist(embedding0_normalized, embeddingReal_normalized, 'euclidean')
    distances_2_Real = distance.cdist(embedding2_normalized, embeddingReal_normalized, metric="euclidean")
    # Step 4: Aggregate distances
    mean_distance_0_2 = np.mean(distances_0_2)
    mean_distance_1_2 = np.mean(distances_1_2)
    assert mean_distance_0_2 < mean_distance_1_2, "It is not improving!!!"
    mean_distance_0_Real = np.mean(distances_0_Real)
    mean_distance_2_Real = np.mean(distances_2_Real)
    assert mean_distance_0_Real < mean_distance_2_Real, "It is not improving!!!"
