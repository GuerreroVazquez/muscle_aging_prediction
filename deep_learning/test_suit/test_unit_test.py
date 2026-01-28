import numpy as np
import pandas as pd
import pytest
from deep_learning.model import divide_test_train_dataset
import deep_learning.model as m
import math

test_data_path = "test_data/"
rnaseq_dataset_path = test_data_path + "rnaseq_expr_abund.csv"

microarray_dataset_path = test_data_path + "microarray_lum.csv"


def get_test():
    data = {
        'NM_001091': [93.629994, -27.212588, -67.652035, -60.991420, -24.714860, -51.595196, 53.785246, -47.670191],
        'NM_033128': [260.535084, -705.217072, -665.770861, -633.805821, 183.682958, -315.515680, -94.480865,
                      -21.709424],
        'BC010094': [-57.238207, -7.066575, -38.439401, 20.322403, -130.814961, -124.092210, 778.997113, 41.611098],
        'NM_003468': [320.518072, -203.487417, -247.154541, -266.665389, -206.274682, -197.912895, -221.604627,
                      -186.299298],
        'BC039525': [14.139142, -131.217350, -139.895350, 65.277341, 61.868132, 173.132486, 70.546140, 77.364559],
        'BC033791': [-230.275607, 142.290881, 226.167062, 310.043292, -333.007724, -113.239020, 569.474403, -44.317485]
    }
    y = [47.97, 50.60, 50.82, 50.15, 51.35, 50.91, 51.00, 51.68]
    return pd.DataFrame(data), pd.DataFrame(y)


def get_train():
    data = {
        'NM_001091': [-1.521648, -36.846692, 6.090480, 152.148247, -52.427773, -50.643680, 84.352706, -22.573949,
                      106.475469, 59.375401, -49.454285, -29.591379, 132.166400, -35.776237, 56.164037, 73.648146,
                      -61.467179, -55.282322, -45.648218, 47.362508, -48.978527, -15.556514, -28.520926, 40.582949,
                      118.964121, 128.479285, -64.916426, -40.771699],
        'NM_033128': [-46.193270, 268.016199, -225.741561, -138.007731, -10.827721, -385.566734, 1427.598947,
                      -599.120355, 242.852257, -428.413478, 375.473146, -131.886757, -453.577446, 561.142411,
                      1772.413133, 484.290285, -401.209193, 435.322594, -624.964433, -217.580244, 57.182992, 296.580678,
                      78.266355, -305.994170, -210.779173, 1131.752349, -540.631153, 52.422263],
        'BC010094': [-196.175023, 379.616027, -44.290698, 350.608586, -164.553200, -227.921338, 215.282119, 52.566712,
                     -216.716756, 66.261201, -0.094847, -204.765204, -110.148734, 137.472522, 565.238572, 169.467843,
                     -47.403082, 98.007517, -37.816924, -176.753749, 47.462395, -112.514147, 177.560041, -90.104974,
                     -211.363457, -44.788672, 97.883014, 379.491486],
        'NM_003468': [88.710694, 9.738231, -223.927346, -49.723384, -190.015645, -277.814440, -203.951962, -149.135786,
                      -266.200843, -239.721846, 525.846532, -206.274682, -215.565560, -137.522188, -243.438194,
                      -190.944737, -188.622017, 219.712059, -243.902739, -210.455575, -158.426664, 250.836522,
                      12.525488, -86.422359, -145.883975, -176.079328, -199.771069, 319.588998],
        'BC039525': [48.541218, 72.095783, 39.863209, 161.975063, -33.899783, 149.577921, 246.585542, 29.635569,
                     66.517060, -145.474064, -84.418140, -99.294708, -74.810354, 20.337713, -112.001780, -132.457066,
                     262.391918, -4.456571, -138.655635, -71.711069, 24.986641, 12.589499, -92.166354, -85.657853,
                     209.394116, 371.486758, -1.977145, 110.216981],
        'BC033791': [-315.452237, 536.964240, 198.858545, 847.111175, -145.749183, -41.716662, 523.960175, 151.393687,
                     -290.744506, -223.123369, 291.837631, 321.096778, 467.392511, 123.434976, -376.571341, 334.100843,
                     42.809762, 365.310570, -384.373779, -131.444707, -315.452237, -282.942074, -367.468495, 222.265852,
                     213.162997, -340.810162, -312.851420, 462.841108]
    }
    y = [51.63, 45.44, 52.22, 52.58, 50.19, 51.65, 50.72, 52.50, 56.08, 46.86, 53.36, 50.85, 48.59, 51.14, 50.88, 51.75,
         49.07, 54.84, 50.26, 49.82, 50.65, 53.01, 51.56, 50.60, 51.41, 52.24, 45.83, 50.80]
    return pd.DataFrame(data), pd.DataFrame(y)


# microarray_dataset_path has 23 columns, 4 non_numeric: Experiment,Sex,Status and Sample

def test_model_prediction_microarray():
    data_df = pd.read_csv(microarray_dataset_path)
    test_model = m.Model(microarray_dataset_path)
    # Test that the model can be created
    normalized_X_train, normalized_X_test_fs = test_model.prepare_data(df=data_df, lasso_alpha=0.1,
                                                                       two_step=False)
    assert len(normalized_X_train.columns) == 24 - 4, "The number of columns changed"

    assert test_model is not None
    # Add more tests as needed


def test_divide_test_train_dataset():
    data_df = pd.read_csv(microarray_dataset_path)
    n_samples = data_df.shape[0]
    n_non_numeric = 4
    n_features = data_df.shape[1] - n_non_numeric - 1
    X_train, X_test, y_train, y_test, z_train, z_test = divide_test_train_dataset(data_df)
    type(X_train)
    assert X_train.shape[1] == n_features, "The number of features is not the same"
    assert X_train.shape[0] + X_test.shape[0] == n_samples
    assert X_test.shape[1] == n_features
    assert y_test.shape[0] == X_test.shape[0] == z_test.shape[0]
    ages_test = y_test.describe()
    ages_train = y_train.describe()


def test_model_normalize_one_step():
    test_model = m.Model(microarray_dataset_path)
    test_model.X_test, test_model.y_test = get_test()
    test_x, test_y = get_test()
    train_x, train_y = get_train()
    test_model.X_train, test_model.y_train = get_train()
    test_model.normalize_data(two_step=False)

    assert isinstance(test_model.X_test, np.ndarray)
    assert not np.array_equal(test_model.X_test, test_x.values), "The normalization didn't change the test data"
    assert not np.array_equal(test_model.X_train, train_x.values), "The normalization didn't change the train data"

    assert math.isclose(test_model.X_train.mean(), b=0, abs_tol=0.001)
    assert math.isclose(test_model.X_train.std(), b=1, abs_tol=0.001)


def test_two_step_normalization():
    test_model = m.Model(microarray_dataset_path)
    test_x, _ = get_train()
    two_step_normalized = test_model.two_step_normalization(test_x)
    mean = two_step_normalized.mean()
    std = two_step_normalized.std()
    x = two_step_normalized.drop(columns=["norm"], inplace=False).std()
    all_mean = two_step_normalized.values.mean()
    add_std = two_step_normalized.values.mean()
    print(two_step_normalized)
    assert (two_step_normalized.isna().sum().sum()) < 1


def test_apply_two_step_normalization():
    test_model = m.Model(microarray_dataset_path)
    test_model.X_test, test_model.y_test = get_test()
    test_x, test_y = get_train()
    test_x_test, _ = get_test()
    two_step_normalized = test_model.two_step_normalization(test_x)
    appy_2_step_test = test_model.apply_two_step_normalization(test_x_test)
    print(appy_2_step_test)
    mean = appy_2_step_test.mean()
    std = appy_2_step_test.std()
    x = appy_2_step_test.drop(columns=["norm"], inplace=False).std()
    all_mean = appy_2_step_test.values.mean()
    add_std = appy_2_step_test.values.mean()
    print(two_step_normalized)
    assert appy_2_step_test.columns.equals(two_step_normalized.columns)
    assert (appy_2_step_test.isna().sum().sum()) < 1
    assert math.isclose(appy_2_step_test.values.mean(), b=0, abs_tol=0.2)
    assert math.isclose(appy_2_step_test.values.std(), b=1, abs_tol=0.2)


def test_model_normalize_two_step():
    test_model = m.Model(microarray_dataset_path)
    test_model.X_test, test_model.y_test = get_test()
    test_x, test_y = get_test()
    train_x, train_y = get_train()
    test_model.X_train, test_model.y_train = get_train()
    test_model.normalize_data(two_step=True)

    assert isinstance(test_model.X_test, pd.DataFrame)
    assert isinstance(test_model.X_train, pd.DataFrame)
    assert not np.array_equal(test_model.X_test, test_x.values), "The normalization didn't change the test data"
    assert not np.array_equal(test_model.X_train, train_x.values), "The normalization didn't change the train data"

    assert math.isclose(test_model.X_train.values.mean(), b=0, abs_tol=0.1)
    assert math.isclose(test_model.X_train.values.std(), b=1, abs_tol=0.1)

    assert (test_model.X_train.isna().sum().sum()) < 1
    assert (test_model.X_test.isna().sum().sum()) < 1


def test_feature_selection():
    test_model = m.Model(microarray_dataset_path)
    X_test, y_test = get_test()
    X_train, y_train = get_train()
    X_train_fs, X_test_fs, _ = test_model.feature_selection(X_train, y_train, X_test=X_test,
                                                alpha=0.1)
    assert X_train_fs
    assert X_test_fs
