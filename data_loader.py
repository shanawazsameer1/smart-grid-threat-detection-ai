import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(data_path, test_size=0.2, val_size=0.25):
    """
    Load and preprocess smart grid data
    
    Args:
        data_path: Path to the CSV file containing the data
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using synthetic data for demonstration...")
        df = generate_synthetic_data()
    
    return preprocess_data(df, test_size, val_size)

def preprocess_data(df, test_size=0.2, val_size=0.25, time_steps=None, n_features=None):
    """
    Preprocess the data for the model
    
    Args:
        df: DataFrame containing the data
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Assuming the last column is the label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].values

    # Coerce features to numeric and handle non-finite values
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop rows with any NaNs in features or missing labels
    valid_mask = X.notna().all(axis=1) & pd.Series(y).notna()
    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"Preprocessing: dropped {dropped} rows due to NaN/inf or missing label")
    X = X.loc[valid_mask]
    y = y[valid_mask.values]
    # Convert to numpy
    X = X.to_numpy(dtype=np.float64)

    # Determine time_steps and n_features
    total_feat = X.shape[1]

    def infer_from_column_names(columns):
        try:
            # Expect names like 'feature_{t}_{f}'
            ts, fs = [], []
            for c in columns[:-1]:
                if isinstance(c, str) and c.startswith('feature_'):
                    parts = c.split('_')
                    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
                        ts.append(int(parts[-2]))
                        fs.append(int(parts[-1]))
            if ts and fs:
                return max(ts) + 1, max(fs) + 1
        except Exception:
            pass
        return None, None

    if time_steps is None or n_features is None:
        # Try to infer from column names
        ts_guess, nf_guess = infer_from_column_names(list(df.columns))
        if ts_guess and nf_guess and ts_guess * nf_guess == total_feat:
            time_steps = ts_guess
            n_features = nf_guess
        else:
            # Try common default of 10 features if divisible
            if total_feat % 10 == 0:
                n_features = 10
                time_steps = total_feat // n_features
            else:
                # Find a reasonable factor pair
                found = False
                for nf in range(2, min(257, total_feat + 1)):
                    if total_feat % nf == 0:
                        ts = total_feat // nf
                        if 2 <= ts <= 2000:  # reasonable bounds
                            n_features = nf
                            time_steps = ts
                            found = True
                            break
                if not found:
                    raise ValueError(
                        f"Cannot factor feature columns ({total_feat}) into time_steps x n_features. "
                        f"Provide explicit values."
                    )

    if time_steps * n_features != total_feat:
        raise ValueError(
            f"Mismatch: total feature columns {total_feat} != time_steps({time_steps}) * n_features({n_features})."
        )

    # Reshape X for time series (time steps assumed in consecutive columns)
    n_samples = X.shape[0]
    X = X.reshape(n_samples, time_steps, n_features)
    
    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Split into train and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
    )
    
    # Normalize data
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Reshape back
    X_train = X_train_flat.reshape(X_train.shape)
    X_val = X_val_flat.reshape(X_val.shape)
    X_test = X_test_flat.reshape(X_test.shape)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def generate_synthetic_data(n_samples=1000, time_steps=24, n_features=10, n_classes=5):
    """
    Generate synthetic smart grid data for testing
    
    Args:
        n_samples: Number of samples to generate
        time_steps: Number of time steps per sample
        n_features: Number of features per time step
        n_classes: Number of attack classes (including normal)
        
    Returns:
        DataFrame with synthetic data
    """
    # Ensure at least 1 sample per class
    n_per_class = max(1, n_samples // n_classes)

    # Generate normal data
    normal_data = np.random.normal(0, 1, (n_per_class, time_steps, n_features))

    # Shape-agnostic anomaly patterns
    attack_data = []
    t = np.linspace(0, 2 * np.pi, time_steps, dtype=np.float64)
    time_wave = np.sin(t)[None, :, None]  # (1, T, 1)
    time_spike = (np.sign(np.sin(5 * t)) + 1) / 2  # (T,)
    time_spike = time_spike[None, :, None]

    # Feature masks
    if n_features > 0:
        feat_idx = np.arange(n_features)
        quarter = max(1, n_features // 4)
        half = max(1, n_features // 2)
        last_k = max(1, min(3, n_features))
        mask_quarter = (feat_idx < quarter).astype(np.float64)[None, None, :]
        mask_half = (feat_idx < half).astype(np.float64)[None, None, :]
        mask_last = (feat_idx >= n_features - last_k).astype(np.float64)[None, None, :]
    else:
        mask_quarter = mask_half = mask_last = 0.0

    for i in range(1, n_classes):
        attack = np.random.normal(0, 1, (n_per_class, time_steps, n_features))

        if i == 1:  # DDoS-like: periodic high load on first quarter features
            attack += 2.5 * time_wave * mask_quarter
        elif i == 2:  # Data Injection-like: smooth drift on half features
            attack += 1.5 * (t / (t.max() + 1e-9))[None, :, None] * mask_half
        elif i == 3:  # Command Injection-like: sharp toggles across features
            attack += 2.0 * time_spike * mask_half
        elif i == 4:  # Scanning-like: elevate last features uniformly
            attack += 1.2 * mask_last
        else:  # Fallback pattern
            attack += 0.8 * time_wave

        attack_data.append(attack)
    
    # Combine data
    X = np.vstack([normal_data] + attack_data)
    
    # Create labels
    y = np.hstack([
        np.zeros(n_per_class),
        np.concatenate([np.full(n_per_class, i) for i in range(1, n_classes)])
    ])
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Flatten X for DataFrame
    X_flat = X.reshape(X.shape[0], -1)
    
    # Create DataFrame
    columns = [f'feature_{i}_{j}' for i in range(time_steps) for j in range(n_features)]
    df = pd.DataFrame(X_flat, columns=columns)
    df['label'] = y
    
    return df
