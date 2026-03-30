import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_synthetic_cbioportal_data(num_samples=1000, random_seed=42):
    """
    Generate synthetic data mimicking cBioPortal clinical and mutation data distributions.

    Args:
        num_samples (int): Number of synthetic samples to generate.
        random_seed (int): Random seed for reproducibility.

    Returns:
        X_scaled (np.ndarray): Scaled feature matrix with columns [mutationCount, age, sex].
        y (np.ndarray): Binary labels array for tumor status.
        scaler (StandardScaler): Fitted scaler object for feature scaling.
    """
    np.random.seed(random_seed)

    # Age distribution: normal with mean ~60, std ~12 (typical for cancer patients)
    age = np.random.normal(60, 12, num_samples).astype(int)
    age = np.clip(age, 18, 90)  # Age between 18 and 90

    # Sex distribution: roughly 50/50 male/female
    sex = np.random.choice([0, 1], num_samples)  # 0=Male, 1=Female

    # Mutation count: Poisson distribution with mean ~5 (adjusted for tumor mutation burden)
    mutation_count = np.random.poisson(5, num_samples)

    # Tumor status: binary label, simulate with some dependency on mutation count and age
    # Higher mutation count and older age increase tumor probability
    tumor_prob = 1 / (1 + np.exp(-0.1*(mutation_count - 5) - 0.05*(age - 60)))
    tumor_status = np.random.binomial(1, tumor_prob)

    X = pd.DataFrame({
        'mutationCount': mutation_count,
        'age': age,
        'sex': sex
    })

    y = tumor_status

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
