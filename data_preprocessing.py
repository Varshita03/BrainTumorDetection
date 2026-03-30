import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_cbioportal_data(clinical_path, mutation_path):
    """
    Preprocess cBioPortal clinical and mutation data files to extract features and labels for fusion model training.

    Args:
        clinical_path (str): Path to clinical data CSV/TSV file.
        mutation_path (str): Path to mutation data CSV/TSV file.

    Returns:
        X_scaled (np.ndarray): Scaled feature matrix with columns [mutationCount, age, sex].
        y (np.ndarray): Binary labels array for tumor status.
        scaler (StandardScaler): Fitted scaler object for feature scaling.
    """
    # Load clinical data
    clinical_df = pd.read_csv(clinical_path, sep=None, engine='python')
    # Load mutation data
    mutation_df = pd.read_csv(mutation_path, sep=None, engine='python')

    # Extract relevant clinical features: patient_id, age, sex, tumor_status
    # Adjust column names as per actual dataset
    clinical_df = clinical_df.rename(columns={
        'PATIENT_ID': 'patient_id',
        'AGE': 'age',
        'SEX': 'sex',
        'TUMOR_STATUS': 'tumor_status'  # Adjust if different
    })

    # Map sex to numeric: Female=1, Male=0 (adjust if needed)
    clinical_df['sex'] = clinical_df['sex'].map({'F': 1, 'M': 0})

    # Filter out rows with missing essential data
    clinical_df = clinical_df.dropna(subset=['patient_id', 'age', 'sex', 'tumor_status'])

    # Calculate mutation count per patient from mutation data
    mutation_counts = mutation_df.groupby('PATIENT_ID').size().reset_index(name='mutationCount')
    mutation_counts = mutation_counts.rename(columns={'PATIENT_ID': 'patient_id'})

    # Merge clinical and mutation counts on patient_id
    merged_df = pd.merge(clinical_df, mutation_counts, on='patient_id', how='inner')

    # Prepare features and labels
    X = merged_df[['mutationCount', 'age', 'sex']]
    y = merged_df['tumor_status'].apply(lambda x: 1 if x == 'Positive' else 0).values  # Adjust label mapping

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
