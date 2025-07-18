import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


@time_function
def count_cover(patch: pd.Series, epsilon: float) -> int:
    return np.ceil(patch.max(axis=1) / epsilon) - np.floor(patch.min(axis=1) / epsilon)


@time_function
def cut_into_patches(curve: pd.Series, epsilon: float) -> list[pd.Series]:
    column = curve.name if curve.name is not None else 0
    df = pd.DataFrame(curve)
    df['patch'] = np.floor(df.index / epsilon)
    return [group.values for _, group in df.groupby('patch')[column]]
    # t = curve.index[-1]
    # K = int(np.ceil(t / epsilon) + 1)
    # a = np.arange(K) * epsilon
    # b = (np.arange(K) + 1) * epsilon
    # return [curve[(curve.index >= a[i]) & (curve.index < b[i])] for i in range(K)]


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
    ax.set_title(f"10 Sample paths of a Brownian Motion")
    ax.set_xlabel("Time", fontsize=8)
    fig.savefig("browinan_motion.png")


    epsilons = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    estimates = []
    for epsilon in epsilons:
        estimates.append(calculate_box_dimension_from_patches(locations, epsilon))
    fig, ax = plt.subplots(figsize=(6, 3.375))
    ax.plot(np.log(1/epsilons), estimates, marker='o', linestyle='-', markersize=4, label="Box Dimension Estimate")
    ax.axhline(1.5, linestyle='--', label='Theoretical Value (1.5)')
    ax.set_title("Box Dimension of Brownian Motion")
    ax.set_xlabel("log(1/epsilon)", fontsize=8)
    ax.set_ylabel("Box Dimension Estimate", fontsize=8)
    ax.legend()
    fig.savefig("box_dimension.png")

