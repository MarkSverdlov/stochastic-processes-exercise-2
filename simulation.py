import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import time

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def simulate(t: float, n: int = 1000) -> pd.Series:
    locations = np.random.normal(0, np.sqrt(t / (n - 1)), size=(n - 1,)).cumsum()
    locations = np.r_[0, locations]
    return pd.Series(locations, index=np.arange(n) * t / (n - 1))


def simulate_fbm(t: float, H: float, n: int = 1000) -> pd.Series:
    dt = t / (n - 1)
    time_grid = np.arange(n) * dt
    cov_matrix = (time_grid[:, None] ** (2 * H) + time_grid[None, :] ** (2 * H) -  np.abs(time_grid[:, None] - time_grid[None, :]) ** (2 * H)) / 2
    L = sp.linalg.sqrtm(cov_matrix)
    increments = np.random.normal(size=n)
    locations = L @ increments
    return pd.Series(locations, index=time_grid)


@time_function
def count_cover(patch: pd.Series, epsilon: float) -> int:
    return np.ceil(patch.max(axis=1) / epsilon) - np.floor(patch.min(axis=1) / epsilon)


@time_function
def cut_into_patches(curve: pd.Series, epsilon: float) -> list[pd.Series]:
    column = curve.name if curve.name is not None else 0
    df = pd.DataFrame(curve)
    df['patch'] = np.floor(df.index / epsilon)
    return [group.values for _, group in df.groupby('patch')[column]]


@time_function
def calculate_box_dimension_from_patches(curve: pd.Series, epsilon: float) -> float:
    patches = cut_into_patches(curve, epsilon)
    max_length = max(len(patch) for patch in patches)
    patches = [np.pad(patch, (0, max_length - len(patch)), mode='mean') for patch in patches]
    vectorized_patches = np.stack(patches, axis=0)
    counts = count_cover(vectorized_patches, epsilon)
    return np.log(np.sum(counts)) / np.log(1 / epsilon)

def linear(t: float, n: int = 1000) -> pd.Series:
    return pd.Series(np.arange(n) * t / (n - 1), index=np.arange(n) * t / (n - 1))


if __name__ == "__main__":
    t = 1
    np.random.seed(42)
    N = 10
    simulation_resolution = int(1e6)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 3.375))
    for _ in range(N):
        locations = simulate(t, n=simulation_resolution)
        ax.plot(locations, label=f"Simulation {N - _}", markersize=0.5, linewidth=0.5)
    ax.set_title(f"10 Sample Paths of a Brownian Motion", fontsize=10)
    ax.set_xlabel("Time", fontsize=8)
    fig.tight_layout()
    fig.savefig("browinan_motion.png")


    epsilons = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    estimates = []
    for epsilon in epsilons:
        estimates.append(calculate_box_dimension_from_patches(locations, epsilon))
    fig, ax = plt.subplots(figsize=(6, 3.375))
    ax.plot(np.log(1/epsilons), estimates, marker='o', linestyle='-', markersize=4, label="Box Dimension Estimate")
    ax.axhline(1.5, linestyle='--', label='Theoretical Value (1.5)')
    ax.set_title("Box Dimension of Brownian Motion", fontsize=10)
    ax.set_xlabel("log(1/epsilon)", fontsize=8)
    ax.set_ylabel("Box Dimension Estimate", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig("box_dimension.png")


    # Simulate and plot fractional Brownian motion
    fbm_simulation_resolution = int(1e3)
    H = 0.7
    fig, axes = plt.subplots(figsize=(6, 3.375*2), nrows=2, sharex=True)
    fbm_sample_paths = []
    for _ in range(N):
        fbm_sample_paths.append(simulate_fbm(t, H, n=fbm_simulation_resolution))
        axes[0].plot(fbm_sample_paths[-1], label=f"Simulation {N - _}", markersize=0.5, linewidth=0.5)
    axes[0].set_title(f"10 Sample Paths of a Fractional Brownian Motion (H={H})", fontsize=10)
    axes[0].set_xlabel("Time", fontsize=8)

    fbm_sample_paths = pd.concat(fbm_sample_paths, axis=1)
    msd = np.mean(fbm_sample_paths ** 2, axis=1)
    axes[1].plot(msd, label='Mean Squared Displacement')
    theoretical_msd = pd.Series(msd.index ** (2 * H), index=msd.index)
    axes[1].plot(theoretical_msd, label='Theoretical MSD', linestyle='--')
    axes[1].set_title(f"Mean Squared Displacement of Fractional Brownian Motion (H={H})", fontsize=10)
    axes[1].set_xlabel("Time", fontsize=8)
    axes[1].set_ylabel("MSD", fontsize=8)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("fractional_brownian_motion.png")

