import os
import io
import numpy as np
import pandas as pd
import torch
import streamlit as st

from model.autoencoder_gan import SmartGridSecurityModel
from utils.data_loader import load_data, preprocess_data

st.set_page_config(page_title="Smart Grid Security", layout="wide")

# Sidebar controls
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox("Mode", ["train", "test", "anomaly_detection"], index=1)

default_data_path = "dataset/data/smart_grid_data.csv"
model_prefix = st.sidebar.text_input("Model path prefix", value="saved_models/smart_grid_model")
latent_dim = st.sidebar.number_input("Latent dim", min_value=8, max_value=256, value=32, step=8)
batch_size = st.sidebar.number_input("Batch size", min_value=8, max_value=512, value=32, step=8)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=200, value=10, step=1)

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"]) 
manual_data_path = st.sidebar.text_input("Or dataset path", value=default_data_path)

# Optional shape hints for uploaded CSVs
st.sidebar.markdown("Optional shape hints for uploaded CSV")
ts_hint = st.sidebar.number_input("time_steps (0 = auto)", min_value=0, max_value=5000, value=0, step=1)
nf_hint = st.sidebar.number_input("n_features (0 = auto)", min_value=0, max_value=1024, value=0, step=1)

st.title("Smart Grid Security - Frontend")
st.caption("Train, test, and perform anomaly detection with your PyTorch model")

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def prepare_splits(df: pd.DataFrame, time_steps: int | None, n_features: int | None):
    return preprocess_data(df, time_steps=(None if not time_steps else int(time_steps)), n_features=(None if not n_features else int(n_features)))

@st.cache_resource(show_spinner=False)
def build_model(input_shape, num_classes, latent):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    return SmartGridSecurityModel(input_shape=input_shape, latent_dim=latent, num_classes=num_classes, device=device)

# Load data
with st.spinner("Loading and preparing data..."):
    if uploaded_file is not None:
        try:
            df = read_uploaded_csv(uploaded_file.getvalue())
            X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df, ts_hint, nf_hint)
            st.success(f"Dataset uploaded. Shapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}\nTip: Set explicit time_steps and n_features in the sidebar to match your data layout.")
            st.stop()
    else:
        # Use load_data which falls back to synthetic data if file missing
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(manual_data_path)
        st.info(f"Using data from '{manual_data_path}' (or synthetic fallback).\ntrain: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1] if len(y_train.shape) > 1 else int(np.max(y_train)) + 1

# Build model
model = build_model(input_shape, num_classes, latent_dim)

col1, col2 = st.columns([2, 1])

if mode == "train":
    with col1:
        st.subheader("Training")
        if st.button("Start Training", type="primary"):
            # Guard against any stale/cached model shape: rebuild if needed
            try:
                expected_shape = (X_train.shape[1], X_train.shape[2])
                if getattr(model, 'input_shape', None) != expected_shape:
                    model = SmartGridSecurityModel(
                        input_shape=expected_shape,
                        latent_dim=int(latent_dim),
                        num_classes=num_classes,
                        device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                    )
            except Exception:
                pass
            with st.spinner("Training autoencoder..."):
                ae_hist = model.train_autoencoder(X_train, X_val, epochs=int(epochs), batch_size=int(batch_size))
            with st.spinner("Training GAN..."):
                d_losses, g_losses = model.train_gan(X_train, epochs=int(epochs), batch_size=int(batch_size))
            with st.spinner("Training classifier..."):
                cls_hist = model.train_classifier(X_train, y_train, X_val, y_val, epochs=int(epochs), batch_size=int(batch_size))
            with st.spinner("Training full model..."):
                full_hist = model.train_full_model(X_train, y_train, X_val, y_val, epochs=int(epochs), batch_size=int(batch_size))

            # Save
            os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
            model.save_model(model_prefix)
            st.success(f"Model saved to prefix: {model_prefix}")

            # Charts organized into tabs
            try:
                tab_ae, tab_gan, tab_cls, tab_full = st.tabs(["Autoencoder", "GAN", "Classifier", "Full Model"])
                with tab_ae:
                    if isinstance(ae_hist, dict) and "loss" in ae_hist and "val_loss" in ae_hist:
                        st.caption("Loss")
                        st.line_chart({"train_loss": ae_hist["loss"], "val_loss": ae_hist["val_loss"]})
                with tab_gan:
                    if d_losses and g_losses:
                        st.caption("Losses")
                        st.line_chart({"discriminator": d_losses, "generator": g_losses})
                with tab_cls:
                    if isinstance(cls_hist, dict):
                        if "loss" in cls_hist and "val_loss" in cls_hist:
                            st.caption("Loss")
                            st.line_chart({"train_loss": cls_hist["loss"], "val_loss": cls_hist["val_loss"]})
                        if "accuracy" in cls_hist and "val_accuracy" in cls_hist:
                            st.caption("Accuracy")
                            st.line_chart({"train_acc": cls_hist["accuracy"], "val_acc": cls_hist["val_accuracy"]})
                with tab_full:
                    if isinstance(full_hist, dict):
                        if "loss" in full_hist and "val_loss" in full_hist:
                            st.caption("Overall Loss")
                            st.line_chart({"train_loss": full_hist["loss"], "val_loss": full_hist["val_loss"]})
                        if "decoder_output_loss" in full_hist and "val_decoder_output_loss" in full_hist:
                            st.caption("Decoder Loss")
                            st.line_chart({"train": full_hist["decoder_output_loss"], "val": full_hist["val_decoder_output_loss"]})
                        if "discriminator_output_loss" in full_hist and "val_discriminator_output_loss" in full_hist:
                            st.caption("Discriminator Loss")
                            st.line_chart({"train": full_hist["discriminator_output_loss"], "val": full_hist["val_discriminator_output_loss"]})
                        if "classifier_output_loss" in full_hist and "val_classifier_output_loss" in full_hist:
                            st.caption("Classifier Loss")
                            st.line_chart({"train": full_hist["classifier_output_loss"], "val": full_hist["val_classifier_output_loss"]})
                        if "classifier_output_accuracy" in full_hist and "val_classifier_output_accuracy" in full_hist:
                            st.caption("Accuracy")
                            st.line_chart({"train_acc": full_hist["classifier_output_accuracy"], "val_acc": full_hist["val_classifier_output_accuracy"]})
            except Exception:
                pass

elif mode == "test":
    with col1:
        st.subheader("Testing / Evaluation")
        if st.button("Evaluate", type="primary"):
            try:
                model.load_model(model_prefix)
            except Exception as e:
                st.warning(f"Could not load existing model at '{model_prefix}'. Proceeding with current weights. Error: {e}")
            # If loaded model shape doesn't match current data, rebuild a compatible model
            try:
                expected_shape = (X_test.shape[1], X_test.shape[2])
                if getattr(model, 'input_shape', None) != expected_shape:
                    st.warning(
                        f"Model checkpoint expects input_shape={getattr(model, 'input_shape', None)}, "
                        f"but data is {expected_shape}. Using a fresh model for evaluation."
                    )
                    model = build_model(expected_shape, num_classes, latent_dim)
            except Exception:
                pass
            with st.spinner("Evaluating on test set..."):
                results = model.evaluate(X_test, y_test)
            try:
                mse = results.get('reconstruction_error')
                threshold = results.get('threshold')
                anomalies_present = False
                if mse is not None and threshold is not None:
                    anomalies_present = bool((mse > threshold).any())
                formatted = (
                    "Classification Report:\n"
                    f"{results['classification_report']}\n\n"
                    f"Anomaly Detection AUC: {results['anomaly_detection_auc']}\n"
                    f"Anomaly Present: {'Yes' if anomalies_present else 'No'}"
                )
                st.text(formatted)
            except Exception:
                # Fallback to original minimal output
                st.text(
                    "Classification Report:\n"
                    f"{results['classification_report']}\n\n"
                    f"Anomaly Detection AUC: {results.get('anomaly_detection_auc', '')}"
                )

elif mode == "anomaly_detection":
    with col1:
        st.subheader("Anomaly Detection")
        if st.button("Detect Anomalies", type="primary"):
            try:
                model.load_model(model_prefix)
            except Exception as e:
                st.warning(f"Could not load existing model at '{model_prefix}'. Proceeding with current weights. Error: {e}")
            # Guard against shape mismatch with checkpoint
            try:
                expected_shape = (X_test.shape[1], X_test.shape[2])
                if getattr(model, 'input_shape', None) != expected_shape:
                    st.warning(
                        f"Model checkpoint expects input_shape={getattr(model, 'input_shape', None)}, "
                        f"but data is {expected_shape}. Using a fresh model."
                    )
                    model = build_model(expected_shape, num_classes, latent_dim)
            except Exception:
                pass
            with st.spinner("Computing reconstruction errors..."):
                anomalies, mse, threshold = model.detect_anomalies(X_test)
            st.text(f"Threshold: {threshold:.4f}\nDetected anomalies: {int(np.sum(anomalies))} / {len(anomalies)}\nAnomaly Present: {'Yes' if int(np.sum(anomalies)) > 0 else 'No'}")
            # If any anomalies, try to classify
            if np.sum(anomalies) > 0:
                attack_probs = model.classify_attacks(X_test[anomalies])
                y_pred_labels = np.argmax(attack_probs, axis=1)
                unique, counts = np.unique(y_pred_labels, return_counts=True)
                st.write("Attack type distribution (predicted):")
                st.write(pd.DataFrame({"class": unique, "count": counts}))

with col2:
    st.subheader("Settings Summary")
    st.json({
        "mode": mode,
        "model_prefix": model_prefix,
        "latent_dim": int(latent_dim),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "input_shape": input_shape,
        "num_classes": int(num_classes),
    })
