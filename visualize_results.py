# visualize_results.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

# ============================================================
# 1. LOAD DATA (replace 'results.csv' with your output file)
# ============================================================
# If your main.py saves predictions, load them here
# Example: columns -> actual, predicted, loss, accuracy
try:
    results = pd.read_csv('results.csv')
except FileNotFoundError:
    print("⚠️ 'results.csv' not found. Please export model results from main.py first.")
    exit()

# ============================================================
# 2. CONFUSION MATRIX VISUALIZATION
# ============================================================
y_true = results['actual']
y_pred = results['predicted']

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Smart Grid Attack Detection")
plt.show()

# ============================================================
# 3. TRAINING LOSS AND ACCURACY GRAPH
# ============================================================
if 'epoch' in results.columns and 'loss' in results.columns and 'accuracy' in results.columns:
    plt.figure(figsize=(10,5))
    plt.plot(results['epoch'], results['loss'], label='Training Loss')
    plt.plot(results['epoch'], results['accuracy'], label='Training Accuracy')
    plt.title('Model Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================
# 4. ANOMALY DETECTION VISUALIZATION
# ============================================================
if 'anomaly_score' in results.columns:
    plt.figure(figsize=(10,5))
    sns.histplot(results['anomaly_score'], bins=30, kde=True, color='purple')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.show()

    # Highlight anomalies above a threshold (say > 0.7)
    threshold = 0.7
    anomalies = results[results['anomaly_score'] > threshold]
    print(f"🔍 Detected {len(anomalies)} anomalies (score > {threshold})")
    print(anomalies.head())

# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================
print("\n🧾 Classification Report:")
print(classification_report(y_true, y_pred))
