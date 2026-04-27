import numpy as np
import matplotlib.pyplot as plt

# Random seed for debugging
np.random.seed(0)

# Defines surface size for strikes by maturities
n_strikes = 16 # number of strike price grid points
n_maturities = 20 # number of time to expire grid points

# Makes evenly spaced grid points for strikes and maturities
# log(moneyness) used instead of direct strike price; moneyness = stock price/ strike prices
log_moneyness_grid = np.linspace(-0.3, 0.3, n_strikes)
maturity_grid = np.linspace(0.1, 2.0, n_maturities)

def generate_surface(parameters):
    # Unpacks fake surface parameters
    base_vol = parameters[0]
    strike_curve = parameters[1]
    maturity_slope = parameters[2]

    # Creates empty volatility surface
    surface = np.zeros((n_strikes, n_maturities))

    # Next creates a simple fake volatility pattern toy formula
    # NOT REAL HESTON!

    # Loop through every strike row and maturity column in the surface
    # i/ j are array positions [i, j]
    for i, log_moneyness in enumerate(log_moneyness_grid):
        for j, maturity in enumerate(maturity_grid):
            # Strike effect creates example parabolic shape of curve (2nd order)
            strike_effect = strike_curve * log_moneyness**2
            # Maturity effect tilts surface as maturity (time) increases
            maturity_effect = maturity_slope * maturity

            implied_vol = base_vol + strike_effect + maturity_effect

            surface[i, j] = implied_vol

    return surface

def sample_parameters():
    # Randomly sample one fake parameter vector
    base_vol = np.random.uniform(0.15, 0.3)
    strike_curve = np.random.uniform(0.1, 0.6)
    maturity_slope = np.random.uniform(0.00, 0.05)

    return np.array([base_vol, strike_curve, maturity_slope])

def generate_dataset(n_example):
    # Stores generated surfaces and matching parameters
    parameter_list = []
    surface_list = []

    # Generates one matched pair of surfaces and parameters at a time
    for _ in range(n_example):
        params = sample_parameters()
        surface = generate_surface(params)

        parameter_list.append(params)
        surface_list.append(surface)

    # Converts lists into NumPy arrays 
    surfaces = np.stack(surface_list)
    parameters = np.stack(parameter_list)
    
    # Returns matched surface/ parameters arrays
    return surfaces, parameters

def plot_surface(surfaces, parameters, example_index):
    # Selects a particular surface and its matching parameters
    surface = surfaces[example_index]
    params = parameters[example_index]

    # Converts 1D strikes/ maturity grids into full 2D coordinate grids
    # Gives matplotlib an (x, y) for every volatility (z) value
    maturity_mesh, log_moneyness_mesh = np.meshgrid(
        maturity_grid,
        log_moneyness_grid
    )

    # Creates a 3D plot for surface
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        log_moneyness_mesh,
        maturity_mesh,
        surface,
        cmap="viridis"
    )

    # Labels for 3D plot
    ax.set_xlabel("Log moneyness")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Implied volatility")

    ax.set_title(
        f"Example {example_index}: "
        f"base_vol={params[0]:.3f}, "
        f"strike_curve={params[1]:.3f}, "
        f"maturity_slope={params[2]:.3f}"
    )
    plt.show()

def main():
    # Generates toy dataset size n
    n_dataset = 1000
    surfaces, parameters = generate_dataset(n_dataset)

    # Saves the dataset (surface/parameters) and grid to \outputs
    np.save("outputs/fake_surfaces.npy", surfaces)
    np.save("outputs/fake_parameters.npy", parameters)
    np.save("outputs/log_moneyness_grid.npy", log_moneyness_grid)
    np.save("outputs/maturity_grid.npy", maturity_grid)

    print("Generated surfaces:", surfaces.shape)
    print("Generated parameters:", parameters.shape)
    print("Surface 0 equals surface 1:", np.allclose(surfaces[0], surfaces[1]))

if __name__ == "__main__":
    main()