from pathlib import Path

import numpy as np
import torch

# Random seeds for testing
np.random.seed(0)
torch.manual_seed(0)

# Clean path
outputs_dir = Path("outputs")

# Loads the .npy files for surfaces and parameters
surfaces = np.load(outputs_dir / "fake_surfaces.npy")
parameters = np.load(outputs_dir / "fake_parameters.npy")

# ML notation
X = surfaces
y = parameters

# Gets number of surfaces from X shape
num_examples = X.shape[0]

# Shuffles order of training and test sets
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

# Sets training fraction size and splits X and y to test and train
training_fraction = 0.8
train_size = int(training_fraction * num_examples)

X_train = X[:train_size]
y_train = y[:train_size]

X_test = X[train_size:]
y_test = y[train_size:]

# Flattens surface from 2D to 1D np arrays
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Convert NumPy arrays into PyTorch tensors
X_train_tensor = torch.tensor(X_train_flat, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_flat, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# In features and out features count (320 and 3)
surface_points = X_train_flat.shape[1]
parameter_count = y_train.shape[1]

# Picks model
model = torch.nn.Sequential(
    # Turns number of surface points into 64 learned features
    torch.nn.Linear(surface_points, 64),
    # Allows nonlinearity
    torch.nn.ReLU(),
    # Turns 64 learned features into 3 parameters
    torch.nn.Linear(64, parameter_count),
)

print(model)

# Sets error measurement as E^2
loss_function = torch.nn.MSELoss()

# Optimizer; made by AI
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Test one forward pass before training
predictions = model(X_train_tensor)
initial_loss = loss_function(predictions, y_train_tensor)
print("Initial loss:", initial_loss.item())

# Training model
epoch_count = 500

for epoch in range(epoch_count+1):
    # Make a prediction
    predictions = model(X_train_tensor)

    # Compare to true parameters
    loss = loss_function(predictions, y_train_tensor)

    # Clears old gradients
    optimizer.zero_grad()

    # Compute new gradients
    loss.backward()

    # Update model weights
    optimizer.step()

    # Print progress every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss: {loss.item()}")

# Evaluate model on test data
model.eval()

# Ignores gradients to save memory/ computation
with torch.no_grad():
    # Runs the 20% test surfaces through the model
    test_predictions = model(X_test_tensor)
    # Calculates the error; average RMSE for each parameter
    test_errors = test_predictions - y_test_tensor
    # Finds mean per column; different parameter each one
    test_mse_by_parameter = torch.mean(test_errors ** 2, dim=0)
    test_rmse_by_parameter = torch.sqrt(test_mse_by_parameter)


# Prints RMSE by parameter
print("Test RMSE by parameter:")
print("base_vol:", test_rmse_by_parameter[0].item())
print("strike_curve:", test_rmse_by_parameter[1].item())
print("maturity_slope:", test_rmse_by_parameter[2].item())


# Prints the normalized RMSE by parameter over the range

parameter_ranges = torch.tensor([0.30 - 0.15, 0.60 - 0.10, 0.05 - 0.00])
normalized_rmse = test_rmse_by_parameter / parameter_ranges

print("\nNormalized RMSE by parameter:")
print(f"base_vol: {normalized_rmse[0].item():.3f}")
print(f"strike_curve: {normalized_rmse[1].item():.3f}")
print(f"maturity_slope: {normalized_rmse[2].item():.3f}")
