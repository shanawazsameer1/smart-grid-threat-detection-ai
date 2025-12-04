import os
from pathlib import Path
import pandas as pd

# Reuse the project's synthetic generator to ensure schema matches preprocess_data
from utils.data_loader import generate_synthetic_data

# Output directory
OUT_DIR = Path('dataset/demo')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Demo configurations: (filename, n_samples, time_steps, n_features, n_classes)
CONFIGS = [
    ("demo_6x13.csv",      2000, 6, 13, 5),   # 78 features
    ("demo_12x8.csv",      2000, 12, 8, 5),   # 96 features
    ("demo_24x10.csv",     2000, 24, 10, 5),  # 240 features
    ("demo_small_8x4.csv", 1000, 8, 4, 3),    # 32 features, fewer classes
]


def main():
    print(f"Writing demo datasets to: {OUT_DIR.resolve()}")
    for fname, n_samples, ts, nf, n_classes in CONFIGS:
        print(f" - Generating {fname} with shape (time_steps={ts}, n_features={nf}), samples={n_samples}, classes={n_classes}")
        df = generate_synthetic_data(n_samples=n_samples, time_steps=ts, n_features=nf, n_classes=n_classes)
        # Ensure label column is last and integer-coded
        if 'label' in df.columns:
            # Make sure label is int
            df['label'] = df['label'].astype(int)
        # Save
        out_path = OUT_DIR / fname
        df.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
