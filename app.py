import os
import numpy as np
import pandas as pd
import torch
import streamlit as st

from utils.data_loader import load_data, preprocess_data, generate_synthetic_data
from utils.visualization import (
    plot_results,
    visualize_latent_space,
    plot_reconstruction_error,
    plot_confusion_matrix,
)
from model.autoencoder_gan import SmartGridSecurityModel

# Ensure required directories
os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

st.set_page_config(page_title="Smart Grid Security", layout="wide")
st.title("Smart Grid Security - Threat Detection UI")
ATTACK_NAMES = ["Normal", "DDoS", "Data Injection", "Command Injection", "Scanning"]

# Sidebar configuration
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox(
    "Mode",
    ["Demonstration (Synthetic)", "Train", "Test", "Anomaly Detection"],
)

# Common parameters
latent_dim = st.sidebar.number_input("Latent Dim", min_value=8, max_value=256, value=32, step=8)
batch_size = st.sidebar.number_input("Batch Size", min_value=8, max_value=512, value=32, step=8)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=200, value=10, step=1)
model_prefix = st.sidebar.text_input("Model Path Prefix", value="saved_models/smart_grid_model")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
st.sidebar.write(f"Device: {device}")

# Quick run options
st.sidebar.subheader("Run Options")
quick_run = st.sidebar.checkbox("Quick Run (fast demo)", value=True, help="Subsample data and reduce epochs for faster feedback")
skip_gan_full = st.sidebar.checkbox("Skip GAN and Full stages (fastest)", value=True, help="Only train Autoencoder and Classifier")
max_samples_per_split = st.sidebar.number_input(
    "Max samples per split (when Quick Run)", min_value=100, max_value=200000, value=2000, step=100,
    help="Applied to each of train/val/test when Quick Run is enabled"
)
skip_visuals = st.sidebar.checkbox("Skip heavy visualizations (t-SNE, plots)", value=True, help="Faster output")

# Data input
st.sidebar.subheader("Data Input")
use_file = st.sidebar.checkbox("Use CSV file", value=False)

data_path = None
uploaded_df = None
if use_file:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 
    default_path = st.sidebar.text_input("Or path to CSV on disk", value="dataset/data/smart_grid_data.csv")
    # Optional overrides for CSV reshaping
    st.sidebar.caption("Optional: set reshape parameters for CSV")
    ts_override = st.sidebar.number_input("time_steps (optional)", min_value=0, max_value=5000, value=0, step=1, help="Leave 0 to auto-infer")
    nf_override = st.sidebar.number_input("n_features (optional)", min_value=0, max_value=1024, value=0, step=1, help="Leave 0 to auto-infer")
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.sidebar.success(f"Uploaded data shape: {uploaded_df.shape}")
        except Exception as e:
            st.sidebar.error(f"Failed to read uploaded CSV: {e}")
    else:
        data_path = default_path
else:
    st.sidebar.caption("If not using a CSV, synthetic data will be generated.")

# Sidebar: direct download demo CSVs
with st.sidebar.expander("Download demo CSVs"):
    st.caption("Click to download ready-to-use demo datasets")
    demo_configs = [
        ("demo_6x13.csv", 2000, 6, 13, 5),
        ("demo_12x8.csv", 2000, 12, 8, 5),
        ("demo_24x10.csv", 2000, 24, 10, 5),
        ("demo_small_8x4.csv", 1000, 8, 4, 3),
    ]
    for fname, n, ts, nf, nc in demo_configs:
        df_demo = generate_synthetic_data(n_samples=n, time_steps=ts, n_features=nf, n_classes=nc)
        st.download_button(
            label=f"Download {fname} (ts={ts}, nf={nf})",
            data=df_demo.to_csv(index=False).encode("utf-8"),
            file_name=fname,
            mime="text/csv",
            key=f"dl_{fname}"
        )

# Helper: initialize model
@st.cache_resource(show_spinner=False)
def get_model(input_shape, latent_dim, num_classes, device):
    return SmartGridSecurityModel(
        input_shape=input_shape,
        latent_dim=latent_dim,
        num_classes=num_classes,
        device=device,
    )

# Helper: load or generate data
def get_splits_from_source(time_steps=None, n_features=None):
    if uploaded_df is not None:
        return preprocess_data(uploaded_df, time_steps=time_steps if time_steps else None, n_features=n_features if n_features else None)
    if use_file and data_path:
        # load_data internally calls preprocess_data without explicit shape; pass via direct preprocess if overrides provided
        if time_steps or n_features:
            try:
                df = pd.read_csv(data_path)
                return preprocess_data(df, time_steps=time_steps, n_features=n_features)
            except Exception:
                # fallback to load_data
                return load_data(data_path)
        return load_data(data_path)
    # synthetic fallback
    df = generate_synthetic_data(n_samples=2000, time_steps=24, n_features=10, n_classes=5)
    return preprocess_data(df)

def subsample_split(X, y, max_n):
    import numpy as np
    n = len(X)
    if n <= max_n:
        return X, y
    idx = np.random.choice(n, max_n, replace=False)
    return X[idx], y[idx]

col_run, col_info = st.columns([2, 1])

# Action buttons
run_clicked = col_run.button("Run", type="primary")

# Output area
log = st.empty()
placeholders = {
    "metrics": st.empty(),
    "images": st.container(),
    "tables": st.container(),
}

if run_clicked:
    try:
        # For Test/Anomaly with saved model, load model first to get expected input shape
        if mode in ("Test", "Anomaly Detection"):
            # Build a temporary model and load weights/config to obtain input_shape/num_classes
            temp_model = SmartGridSecurityModel(input_shape=(24, 10), latent_dim=latent_dim, num_classes=5, device=device)
            temp_model.load_model(model_prefix)
            expected_ts, expected_nf = temp_model.input_shape
            # If user provided overrides, honor them; else align to model config
            ts = int(ts_override) if use_file and 'ts_override' in locals() and ts_override > 0 else expected_ts
            nf = int(nf_override) if use_file and 'nf_override' in locals() and nf_override > 0 else expected_nf
            log.info(f"Preprocessing to match model shape: time_steps={ts}, n_features={nf}")
            X_train, X_val, X_test, y_train, y_val, y_test = get_splits_from_source(time_steps=ts, n_features=nf)
            # Validate shape; if mismatch, auto-train a quick model on this CSV and proceed
            cur_ts, cur_nf = X_train.shape[1], X_train.shape[2]
            if (cur_ts, cur_nf) != (expected_ts, expected_nf):
                st.warning(
                    f"Loaded model expects {(expected_ts, expected_nf)} but CSV is {(cur_ts, cur_nf)}.\n"
                    "Auto-training a quick model to match your CSV, then proceeding..."
                )
                # Build model for CSV shape
                num_classes = y_train.shape[1]
                model = get_model((cur_ts, cur_nf), latent_dim, num_classes, device)
                # Create tensors for quick training
                X_train_t_q = torch.tensor(X_train, dtype=torch.float32)
                X_val_t_q = torch.tensor(X_val, dtype=torch.float32)
                y_train_t_q = torch.tensor(y_train, dtype=torch.float32)
                y_val_t_q = torch.tensor(y_val, dtype=torch.float32)
                # Quick train
                q_epochs = max(1, min(epochs, 3))
                log.info("Quick training Autoencoder (mismatch)...")
                _ = model.train_autoencoder(X_train_t_q, X_val_t_q, epochs=q_epochs, batch_size=min(batch_size, 64))
                log.info("Quick training GAN (mismatch)...")
                _d, _g = model.train_gan(X_train_t_q, epochs=q_epochs, batch_size=min(batch_size, 64))
                log.info("Quick training Classifier (mismatch)...")
                _ = model.train_classifier(X_train_t_q, y_train_t_q, X_val_t_q, y_val_t_q, epochs=q_epochs, batch_size=min(batch_size, 64))
                # Optional very short fine-tune
                log.info("Quick fine-tuning full model (mismatch)...")
                _ = model.train_full_model(X_train_t_q, y_train_t_q, X_val_t_q, y_val_t_q, epochs=1, batch_size=min(batch_size, 64))
                # Save this model to prefix so Test/Anomaly use it
                model.save_model(model_prefix)
                st.success("Auto-training complete. Proceeding with detection using the new model.")
                num_classes = model.num_classes
            else:
                model = temp_model  # reuse loaded model
                num_classes = model.num_classes
        else:
            # Demonstration/Train path: preprocess first, then build model to match data
            log.info("Loading/preprocessing data...")
            # Apply optional overrides for CSV if provided
            ts = int(ts_override) if use_file and 'ts_override' in locals() and ts_override > 0 else None
            nf = int(nf_override) if use_file and 'nf_override' in locals() and nf_override > 0 else None
            X_train, X_val, X_test, y_train, y_val, y_test = get_splits_from_source(time_steps=ts, n_features=nf)
            num_classes = y_train.shape[1]
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = get_model(input_shape, latent_dim, num_classes, device)

        # Torch tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        # Ensure tensors are created after potential shape-aligned preprocessing

        if mode == "Demonstration (Synthetic)":
            # Optional subsampling for quick run
            if quick_run:
                X_train, y_train = subsample_split(X_train, y_train, max_samples_per_split)
                X_val, y_val = subsample_split(X_val, y_val, max_samples_per_split)
                X_test, y_test = subsample_split(X_test, y_test, max_samples_per_split)
                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                y_train_t = torch.tensor(y_train, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32)
                y_test_t = torch.tensor(y_test, dtype=torch.float32)

            log.info("Training Autoencoder...")
            ae_epochs = min(epochs, 2) if quick_run else epochs
            ae_history = model.train_autoencoder(X_train_t, X_val_t, epochs=ae_epochs, batch_size=min(batch_size, 64) if quick_run else batch_size)

            if not skip_gan_full:
                log.info("Training GAN...")
                d_losses, g_losses = model.train_gan(X_train_t, epochs=min(epochs, 2) if quick_run else epochs, batch_size=min(batch_size, 64) if quick_run else batch_size)
            else:
                d_losses, g_losses = [], []

            log.info("Training Classifier...")
            cls_history = model.train_classifier(
                X_train_t, y_train_t, X_val_t, y_val_t,
                epochs=min(epochs, 2) if quick_run else epochs,
                batch_size=min(batch_size, 64) if quick_run else batch_size
            )

            if not skip_gan_full:
                log.info("Training Full Model...")
                full_history = model.train_full_model(
                    X_train_t, y_train_t, X_val_t, y_val_t,
                    epochs=1 if quick_run else max(1, epochs // 2),
                    batch_size=min(batch_size, 64) if quick_run else batch_size
                )
            else:
                full_history = {
                    'loss': [], 'decoder_output_loss': [], 'classifier_output_loss': []
                }

            log.success("Training complete. Saving plots...")
            if not skip_gan_full:
                plot_results(ae_history, d_losses, g_losses, cls_history, full_history)

            with placeholders["images"]:
                st.image("results/training_results.png", caption="Training Curves", use_container_width=True)

            log.info("Evaluating...")
            results = model.evaluate(X_test_t, y_test_t)
            placeholders["metrics"].code(str(results["classification_report"]))
            st.write({k: v for k, v in results.items() if k != "classification_report"})

            if not skip_visuals:
                log.info("Visualizing latent space...")
                visualize_latent_space(model, X_test_t, y_test_t)
                with placeholders["images"]:
                    st.image("results/latent_space_visualization.png", caption="Latent Space", use_container_width=True)

            anomalies, mse, threshold = model.detect_anomalies(X_test_t)
            if not skip_visuals:
                plot_reconstruction_error(mse, threshold, y_test_t)
                with placeholders["images"]:
                    st.image("results/reconstruction_error.png", caption="Reconstruction Error", use_container_width=True)

            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(model.classify_attacks(X_test_t), axis=1)
            any_attack = bool(np.any(y_pred != 0))
            st.metric("Any attack detected", "Yes" if any_attack else "No")
            vals, cnts = np.unique(y_pred, return_counts=True)
            summary = {ATTACK_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)}
            st.write("Predicted threat counts:", summary)

            df_preview = pd.DataFrame({
                "true": [ATTACK_NAMES[int(i)] for i in y_true],
                "pred": [ATTACK_NAMES[int(i)] for i in y_pred],
                "anomaly": anomalies,
                "mse": mse,
            })
            with placeholders["tables"]:
                st.dataframe(df_preview.head(50))

            from sklearn.metrics import confusion_matrix
            if not skip_visuals:
                conf = confusion_matrix(y_true, y_pred)
                plot_confusion_matrix(conf, class_names=ATTACK_NAMES)
                with placeholders["images"]:
                    st.image("results/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

            # Export CSV
            try:
                df_out = df_preview.copy()
                df_out.to_csv("results/results.csv", index=False)
                with placeholders["tables"]:
                    st.dataframe(df_out.head(50))
                st.success("Exported results to results/results.csv")
            except Exception as e:
                st.warning(f"Could not export results: {e}")

        elif mode == "Train":
            log.info("Training pipeline starting...")
            # Optional subsampling for quick run
            if quick_run:
                X_train, y_train = subsample_split(X_train, y_train, max_samples_per_split)
                X_val, y_val = subsample_split(X_val, y_val, max_samples_per_split)
                X_test, y_test = subsample_split(X_test, y_test, max_samples_per_split)
                X_train_t = torch.tensor(X_train, dtype=torch.float32)
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                y_train_t = torch.tensor(y_train, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32)

            ae_history = model.train_autoencoder(
                X_train_t, X_val_t,
                epochs=min(epochs, 2) if quick_run else epochs,
                batch_size=min(batch_size, 64) if quick_run else batch_size
            )
            if not skip_gan_full:
                d_losses, g_losses = model.train_gan(
                    X_train_t,
                    epochs=min(epochs, 2) if quick_run else epochs,
                    batch_size=min(batch_size, 64) if quick_run else batch_size
                )
            else:
                d_losses, g_losses = [], []
            cls_history = model.train_classifier(
                X_train_t, y_train_t, X_val_t, y_val_t,
                epochs=min(epochs, 2) if quick_run else epochs,
                batch_size=min(batch_size, 64) if quick_run else batch_size
            )
            if not skip_gan_full:
                full_history = model.train_full_model(
                    X_train_t, y_train_t, X_val_t, y_val_t,
                    epochs=1 if quick_run else max(1, epochs // 2),
                    batch_size=min(batch_size, 64) if quick_run else batch_size
                )
            else:
                full_history = {
                    'loss': [], 'decoder_output_loss': [], 'classifier_output_loss': []
                }
            model.save_model(model_prefix)
            if not skip_gan_full:
                plot_results(ae_history, d_losses, g_losses, cls_history, full_history)
            with placeholders["images"]:
                st.image("results/training_results.png", caption="Training Curves", use_container_width=True)
            st.success(f"Model saved with prefix: {model_prefix}")

        elif mode == "Test":
            log.info("Loading model and evaluating...")
            # model already loaded to align shapes
            results = model.evaluate(X_test_t, y_test_t)
            placeholders["metrics"].code(str(results["classification_report"]))
            st.write({k: v for k, v in results.items() if k != "classification_report"})
            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(model.classify_attacks(X_test_t), axis=1)
            any_attack = bool(np.any(y_pred != 0))
            st.metric("Any attack detected", "Yes" if any_attack else "No")
            vals, cnts = np.unique(y_pred, return_counts=True)
            st.write("Predicted threat counts:", {ATTACK_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)})
            df_preview = pd.DataFrame({
                "true": [ATTACK_NAMES[int(i)] for i in y_true],
                "pred": [ATTACK_NAMES[int(i)] for i in y_pred],
            })
            with placeholders["tables"]:
                st.dataframe(df_preview.head(50))
            visualize_latent_space(model, X_test_t, y_test_t)
            with placeholders["images"]:
                st.image("results/latent_space_visualization.png", caption="Latent Space", use_container_width=True)

        elif mode == "Anomaly Detection":
            log.info("Loading model and detecting anomalies...")
            # model already loaded to align shapes
            anomalies, mse, threshold = model.detect_anomalies(X_test_t)
            n_anom = int(np.sum(anomalies))
            st.metric("Detected anomalies", n_anom)
            st.write({"threshold": float(threshold)})
            plot_reconstruction_error(mse, threshold, y_test_t)
            with placeholders["images"]:
                st.image("results/reconstruction_error.png", caption="Reconstruction Error", use_container_width=True)
            if n_anom > 0:
                y_pred = np.argmax(model.classify_attacks(X_test_t[anomalies]), axis=1)
                st.write("Anomalous class distribution:")
                vals, cnts = np.unique(y_pred, return_counts=True)
                st.write({ATTACK_NAMES[int(v)]: int(c) for v, c in zip(vals, cnts)})

        log.success("Done.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

with col_info:
    st.subheader("Instructions")
    st.markdown(
        """
        - Install deps: `pip install -r requirements.txt`
        - Run UI: `streamlit run app.py`
        - To use your own CSV, check "Use CSV file" and upload or provide a path.
        - Adjust epochs/batch size for faster demos.
        - Models are saved under `saved_models/` with the prefix you set.
        """
    )
