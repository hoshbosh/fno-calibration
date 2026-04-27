import numpy as np

def main():

    # Loads toy surfaces, parameters, and grids from generate_heston.py
    
    surfaces = np.load("outputs/fake_surfaces.npy")
    parameters = np.load("outputs/fake_parameters.npy")
    log_moneyness_grid = np.load("outputs/log_moneyness_grid.npy")
    maturity_grid = np.load("outputs/maturity_grid.npy")

    # Inputs and targets: volatility surfaces and parameters

    X = surfaces
    y = parameters

    # Splits X and y into training and test sets

    # First shuffles order
    np.random.seed(0) # Random seed for testing
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # Splits into a fraction of training and testing sets
    train_fraction = 0.80
    n_train = int(train_fraction * len(X))

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    y_test = y[n_train:]

    # Flattens surface from 2D (16, 20) to 1D (320) for simple linear model

    flat_X_train = X_train.reshape(X_train.shape[0], -1)
    flat_X_test = X_test.reshape(X_test.shape[0], -1)

    # Adds a bias column of 1s (y_int in linear model) prediction = bias + weighted inputs

    ones_train = np.ones((flat_X_train.shape[0], 1))
    ones_test = np.ones((flat_X_test.shape[0], 1))

    X_train_design = np.concatenate([ones_train, flat_X_train], axis=1)
    X_test_design = np.concatenate([ones_test, flat_X_test], axis=1)

    # Example least squares regression to solve weights

    weights, residuals, rank, singular_values = np.linalg.lstsq(
        X_train_design,
        y_train,
        rcond=None
    )
    # Use the learned weights to predict parameters for train and test surfaces
    # Matrix math
    train_predictions = X_train_design @ weights
    test_predictions = X_test_design @ weights

    # Compares predicted training/test parameters to actual
    train_errors = train_predictions - y_train
    test_errors = test_predictions - y_test

    # Calculates Root Mean Squared Error for each parameter
    train_rmse = np.sqrt(np.mean(train_errors**2, axis=0))
    test_rmse = np.sqrt(np.mean(test_errors**2, axis=0))

    # Calculates normalized RMSE for each parameter to compare to pytorch
    parameter_std = np.std(y_test, axis=0)
    normalized_rmse = test_rmse / parameter_std

    # Prints RMSE results
    parameter_names = ["base_vol", "strike_curve", "maturity_slope"]

    print("\nTrain RMSE by parameter:")
    for name, error in zip(parameter_names, train_rmse):
        print(f"{name}: {error:.10f}")

    print("\nTest RMSE by parameter:")
    for name, error in zip(parameter_names, test_rmse):
        print(f"{name}: {error:.10f}")

    print("\nNormalized Test RMSE by parameter:")
    for name, error in zip(parameter_names, normalized_rmse):
        print(f"{name}: {error:.10f}")

if __name__ == "__main__":
    main()