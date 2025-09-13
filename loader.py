import pandas as pd
import numpy as np
from pathlib import Path


def load_heart_disease_datasets(data_folder='data'):
    """
    Load and combine all heart disease datasets from the UCI repository

    Parameters:
    -----------
    data_folder : str
        Path to the folder containing the dataset files

    Returns:
    --------
    combined_df : pd.DataFrame
        Combined dataset from all locations
    individual_datasets : dict
        Dictionary containing individual datasets by location
    """

    # Define the datasets and their corresponding files
    dataset_files = {
        'cleveland': 'processed.cleveland.data',
        'hungarian': 'processed.hungarian.data',
        'switzerland': 'processed.switzerland.data',
        'va': 'processed.va.data'
    }

    # Column names for the heart disease dataset
    column_names = [
        'age',  # Age in years
        'sex',  # Sex (1 = male; 0 = female)
        'cp',  # Chest pain type (1-4)
        'trestbps',  # Resting blood pressure (in mm Hg)
        'chol',  # Serum cholestoral in mg/dl
        'fbs',  # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        'restecg',  # Resting electrocardiographic results (0-2)
        'thalach',  # Maximum heart rate achieved
        'exang',  # Exercise induced angina (1 = yes; 0 = no)
        'oldpeak',  # ST depression induced by exercise relative to rest
        'slope',  # Slope of the peak exercise ST segment (1-3)
        'ca',  # Number of major vessels (0-3) colored by flourosopy
        'thal',  # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        'target'  # Target variable (0 = no disease, 1-4 = disease severity)
    ]

    individual_datasets = {}
    all_dataframes = []

    print("Loading heart disease datasets...")

    for location, filename in dataset_files.items():
        file_path = Path(data_folder) / filename

        try:
            # Read the data file
            df = pd.read_csv(file_path, names=column_names, na_values='?')

            # Add location column
            df['location'] = location

            # Store individual dataset
            individual_datasets[location] = df.copy()

            print(f"✓ Loaded {location} dataset: {len(df)} samples")

            # Add to combined list
            all_dataframes.append(df)

        except FileNotFoundError:
            print(f"⚠️  {filename} not found in {data_folder}")
            continue
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
            continue

    # Combine all datasets
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n✅ Combined dataset created: {len(combined_df)} total samples")

        # Convert target to binary (0 = no disease, 1 = disease)
        combined_df['target_binary'] = (combined_df['target'] > 0).astype(int)

        return combined_df, individual_datasets
    else:
        print("❌ No datasets could be loaded")
        return None, None


def preprocess_heart_disease_data(df):
    """
    Preprocess the heart disease dataset

    Parameters:
    -----------
    df : pd.DataFrame
        Raw heart disease dataset

    Returns:
    --------
    X : pd.DataFrame
        Processed features
    y : pd.Series
        Target variable
    preprocessing_info : dict
        Information about preprocessing steps
    """

    print("Starting data preprocessing...")

    # Create a copy to avoid modifying original
    df_processed = df.copy()

    # Remove rows with missing target values
    df_processed = df_processed.dropna(subset=['target'])

    # Handle missing values in features
    missing_info = df_processed.isnull().sum()
    print(f"Missing values before processing:\n{missing_info[missing_info > 0]}")

    # Feature preprocessing
    preprocessing_steps = []

    # 1. Handle missing values
    # For 'ca' and 'thal' which commonly have missing values
    if 'ca' in df_processed.columns:
        median_ca = df_processed['ca'].median()
        # FIX: Assign the result back to the column
        df_processed['ca'] = df_processed['ca'].fillna(median_ca)
        preprocessing_steps.append(f"Filled 'ca' missing values with median: {median_ca}")

    if 'thal' in df_processed.columns:
        mode_thal = df_processed['thal'].mode()[0] if not df_processed['thal'].mode().empty else 3
        # FIX: Assign the result back to the column
        df_processed['thal'] = df_processed['thal'].fillna(mode_thal)
        preprocessing_steps.append(f"Filled 'thal' missing values with mode: {mode_thal}")

    # Fill other missing values with median for numerical, mode for categorical
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            # FIX: Assign the result back to the column
            df_processed[col] = df_processed[col].fillna(median_val)
            preprocessing_steps.append(f"Filled '{col}' with median: {median_val}")

    # 2. Feature engineering
    # Create age groups
    df_processed['age_group'] = pd.cut(df_processed['age'],
                                       bins=[0, 40, 55, 70, 100],
                                       labels=['Young', 'Middle_aged', 'Senior', 'Elderly'])
    df_processed['age_group'] = df_processed['age_group'].cat.codes

    # Create cholesterol categories
    if 'chol' in df_processed.columns:
        df_processed['chol_category'] = pd.cut(df_processed['chol'],
                                               bins=[0, 200, 240, 1000],
                                               labels=['Normal', 'Borderline', 'High'])
        df_processed['chol_category'] = df_processed['chol_category'].cat.codes

    # Create blood pressure categories
    if 'trestbps' in df_processed.columns:
        df_processed['bp_category'] = pd.cut(df_processed['trestbps'],
                                             bins=[0, 120, 140, 1000],
                                             labels=['Normal', 'Elevated', 'High'])
        df_processed['bp_category'] = df_processed['bp_category'].cat.codes

    preprocessing_steps.append("Created age_group, chol_category, bp_category features")

    # 3. Separate features and target
    # Use binary target if available, otherwise convert target
    if 'target_binary' in df_processed.columns:
        y = df_processed['target_binary']
    else:
        y = (df_processed['target'] > 0).astype(int)

    # Select feature columns (exclude target columns and location)
    feature_columns = [col for col in df_processed.columns
                       if col not in ['target', 'target_binary', 'location']]
    X = df_processed[feature_columns]

    # 4. Data quality checks
    print(f"\n📊 Dataset Statistics:")
    print(f"Total samples: {len(df_processed)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Target distribution:")
    print(y.value_counts())
    print(f"Missing values after processing: {X.isnull().sum().sum()}")

    preprocessing_info = {
        'steps': preprocessing_steps,
        'original_shape': df.shape,
        'processed_shape': X.shape,
        'target_distribution': y.value_counts().to_dict(),
        'feature_columns': feature_columns
    }

    print("✅ Data preprocessing completed!")

    return X, y, preprocessing_info


def save_processed_data(X, y, output_folder='data'):
    """
    Save processed data to CSV files

    Parameters:
    -----------
    X : pd.DataFrame
        Processed features
    y : pd.Series
        Target variable
    output_folder : str
        Output folder path
    """

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    # Save processed data
    X.to_csv(output_path / 'X_processed.csv', index=False)
    y.to_csv(output_path / 'y_processed.csv', index=False)

    print(f"✅ Saved processed data to {output_folder}/")
    print(f"   - X_processed.csv: {X.shape}")
    print(f"   - y_processed.csv: {y.shape}")


# Example usage
if __name__ == "__main__":
    # Load datasets from the current directory ('.') instead of a 'data' subfolder
    combined_df, individual_datasets = load_heart_disease_datasets(data_folder='.')

    if combined_df is not None:
        # Display dataset information
        print(f"\n📊 Combined Dataset Overview:")
        print(f"Shape: {combined_df.shape}")
        print(f"Columns: {list(combined_df.columns)}")

        print(f"\nDataset distribution by location:")
        print(combined_df['location'].value_counts())

        print(f"\nTarget distribution:")
        print(combined_df['target'].value_counts().sort_index())

        # Preprocess the data
        X, y, preprocessing_info = preprocess_heart_disease_data(combined_df)

        # Save processed data to the current directory
        save_processed_data(X, y, output_folder='.')

        # Display preprocessing information
        print(f"\n🔧 Preprocessing Summary:")
        for step in preprocessing_info['steps']:
            print(f"   - {step}")

    else:
        print("❌ Failed to load datasets. Please check your data files.")