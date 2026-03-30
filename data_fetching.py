import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def fetch_cbioportal_data(study_id, clinical_endpoint, mutation_endpoint):
    """
    Fetch clinical and mutation data from cBioPortal API and preprocess for fusion model training.

    Args:
        study_id (str): cBioPortal study ID, e.g. 'gbm_tcga'
        clinical_endpoint (str): API endpoint for clinical data
        mutation_endpoint (str): API endpoint for mutation data

    Returns:
        X_scaled (np.ndarray): Scaled feature matrix with columns [mutationCount, age, sex].
        y (np.ndarray): Binary labels array for tumor status.
        scaler (StandardScaler): Fitted scaler object for feature scaling.
    """
    base_url = "https://www.cbioportal.org/api"

    # Correct clinical data endpoint
    clinical_url = f"{base_url}/studies/{study_id}/clinical-data"
    clinical_response = requests.get(clinical_url)
    clinical_response.raise_for_status()
    clinical_json = clinical_response.json()
    clinical_df = pd.json_normalize(clinical_json)

    # Correct mutation data endpoint
    mutation_url = f"{base_url}/studies/{study_id}/mutations"
    mutation_response = requests.get(mutation_url)
    mutation_response.raise_for_status()
    mutation_json = mutation_response.json()
    mutation_df = pd.json_normalize(mutation_json)

    # Extract relevant clinical features: patient_id, age, sex, tumor_status
    # Adjust column names as per actual API response
    clinical_df = clinical_df.rename(columns={
        'patientId': 'patient_id',
        'age': 'age',
        'gender': 'sex',
        'tumorStatus': 'tumor_status'  # Adjust if different
    })

    # Map sex to numeric: Female=1, Male=0 (adjust if needed)
    clinical_df['sex'] = clinical_df['sex'].map({'F': 1, 'M': 0})

    # Filter out rows with missing essential data
    clinical_df = clinical_df.dropna(subset=['patient_id', 'age', 'sex', 'tumor_status'])

    # Calculate mutation count per patient from mutation data
    mutation_counts = mutation_df.groupby('patientId').size().reset_index(name='mutationCount')
    mutation_counts = mutation_counts.rename(columns={'patientId': 'patient_id'})

    # Merge clinical and mutation counts on patient_id
    merged_df = pd.merge(clinical_df, mutation_counts, on='patient_id', how='inner')

    # Prepare features and labels
    X = merged_df[['mutationCount', 'age', 'sex']]
    y = merged_df['tumor_status'].apply(lambda x: 1 if x == 'Positive' else 0).values  # Adjust label mapping

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
