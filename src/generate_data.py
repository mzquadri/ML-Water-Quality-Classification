"""
Generate synthetic water quality dataset for classification.

Features based on WHO water quality parameters:
- pH, Hardness, Solids (TDS), Chloramines, Sulfate
- Conductivity, Organic Carbon, Trihalomethanes, Turbidity
- Target: Potability (1 = safe to drink, 0 = not safe)
"""

import numpy as np
import pandas as pd
import os


def generate_water_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic water quality data with realistic distributions."""
    rng = np.random.RandomState(seed)

    n = n_samples
    # Imbalanced: ~40% potable, ~60% not potable
    potability = rng.binomial(1, 0.40, n)

    # pH (6.5-8.5 is WHO standard)
    ph = np.where(
        potability == 1,
        rng.normal(7.2, 0.6, n),
        rng.normal(6.8, 1.2, n),
    )
    ph = np.clip(ph, 2, 14)

    # Hardness (mg/L) — potable tends to be moderate
    hardness = np.where(
        potability == 1,
        rng.normal(180, 40, n),
        rng.normal(210, 60, n),
    )
    hardness = np.clip(hardness, 50, 400)

    # Total Dissolved Solids (mg/L) — <500 is WHO standard
    solids = np.where(
        potability == 1,
        rng.normal(18000, 5000, n),
        rng.normal(22000, 8000, n),
    )
    solids = np.clip(solids, 300, 60000)

    # Chloramines (ppm) — 0-4 is safe
    chloramines = np.where(
        potability == 1,
        rng.normal(7.0, 1.2, n),
        rng.normal(7.5, 1.8, n),
    )
    chloramines = np.clip(chloramines, 1, 13)

    # Sulfate (mg/L) — <250 is WHO
    sulfate = np.where(
        potability == 1,
        rng.normal(320, 40, n),
        rng.normal(340, 55, n),
    )
    sulfate = np.clip(sulfate, 100, 500)

    # Conductivity (μS/cm)
    conductivity = np.where(
        potability == 1,
        rng.normal(400, 70, n),
        rng.normal(430, 90, n),
    )
    conductivity = np.clip(conductivity, 180, 800)

    # Organic Carbon (mg/L)
    organic_carbon = np.where(
        potability == 1,
        rng.normal(13, 3, n),
        rng.normal(15, 4, n),
    )
    organic_carbon = np.clip(organic_carbon, 2, 30)

    # Trihalomethanes (μg/L) — <80 is EPA standard
    trihalomethanes = np.where(
        potability == 1,
        rng.normal(60, 15, n),
        rng.normal(68, 20, n),
    )
    trihalomethanes = np.clip(trihalomethanes, 5, 130)

    # Turbidity (NTU) — <5 is WHO
    turbidity = np.where(
        potability == 1,
        rng.normal(3.5, 0.8, n),
        rng.normal(4.0, 1.0, n),
    )
    turbidity = np.clip(turbidity, 1, 7)

    # Add some missing values (realistic — 5-10% per feature)
    df = pd.DataFrame(
        {
            "ph": ph,
            "hardness": np.round(hardness, 2),
            "solids": np.round(solids, 2),
            "chloramines": np.round(chloramines, 2),
            "sulfate": np.round(sulfate, 2),
            "conductivity": np.round(conductivity, 2),
            "organic_carbon": np.round(organic_carbon, 2),
            "trihalomethanes": np.round(trihalomethanes, 2),
            "turbidity": np.round(turbidity, 2),
            "potability": potability,
        }
    )

    # Introduce 5-8% missing values in some columns
    for col in ["ph", "sulfate", "trihalomethanes"]:
        mask = rng.random(n) < 0.07
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_water_data(n_samples=5000)
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "water_quality.csv"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Generated {len(df)} samples")
    print(f"Class distribution:\n{df['potability'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\n{df.describe().round(2)}")
