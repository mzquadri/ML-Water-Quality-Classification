"""Generate a synthetic classification dataset with water-flavoured column names.

This is not water quality data and the label is not a safety judgement. The label
is drawn first, as a coin flip, and the nine features are then drawn from one of
two normal distributions selected by it. Nothing is measured and no standard is
applied.

The column names and the units in the comments below borrow the vocabulary of
water chemistry so the dataset reads as a plausible tabular problem. They do not
describe the values produced. Total dissolved solids are drawn around 18,000 to
22,000 in both classes, against a WHO drinking-water guideline of 1,000, and
sulfate around 320 to 340 against a guideline of 250. The numbers are not near
any real threshold and no threshold is used to assign the label.

Because the generating process is fully specified here, the best accuracy any
classifier could reach on this data is computable. src/ceiling.py does that.
"""

import os

import numpy as np
import pandas as pd


def generate_water_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic water quality data with realistic distributions."""
    rng = np.random.RandomState(seed)

    n = n_samples
    # The label. Drawn from nothing: no feature exists yet, so this is not a
    # rule over the data and cannot be recovered exactly from it.
    potability = rng.binomial(1, 0.40, n)

    # A pH-like column. Both classes overlap heavily.
    ph = np.where(
        potability == 1,
        rng.normal(7.2, 0.6, n),
        rng.normal(6.8, 1.2, n),
    )
    ph = np.clip(ph, 2, 14)

    # A hardness-like column, nominally mg/L.
    hardness = np.where(
        potability == 1,
        rng.normal(180, 40, n),
        rng.normal(210, 60, n),
    )
    hardness = np.clip(hardness, 50, 400)

    # A dissolved-solids-like column. The values are far above any
    # drinking-water guideline, in both classes alike.
    solids = np.where(
        potability == 1,
        rng.normal(18000, 5000, n),
        rng.normal(22000, 8000, n),
    )
    solids = np.clip(solids, 300, 60000)

    # A chloramine-like column, nominally ppm.
    chloramines = np.where(
        potability == 1,
        rng.normal(7.0, 1.2, n),
        rng.normal(7.5, 1.8, n),
    )
    chloramines = np.clip(chloramines, 1, 13)

    # A sulfate-like column, nominally mg/L.
    sulfate = np.where(
        potability == 1,
        rng.normal(320, 40, n),
        rng.normal(340, 55, n),
    )
    sulfate = np.clip(sulfate, 100, 500)

    # A conductivity-like column, nominally uS/cm.
    conductivity = np.where(
        potability == 1,
        rng.normal(400, 70, n),
        rng.normal(430, 90, n),
    )
    conductivity = np.clip(conductivity, 180, 800)

    # An organic-carbon-like column, nominally mg/L.
    organic_carbon = np.where(
        potability == 1,
        rng.normal(13, 3, n),
        rng.normal(15, 4, n),
    )
    organic_carbon = np.clip(organic_carbon, 2, 30)

    # A trihalomethane-like column, nominally ug/L.
    trihalomethanes = np.where(
        potability == 1,
        rng.normal(60, 15, n),
        rng.normal(68, 20, n),
    )
    trihalomethanes = np.clip(trihalomethanes, 5, 130)

    # A turbidity-like column, nominally NTU.
    turbidity = np.where(
        potability == 1,
        rng.normal(3.5, 0.8, n),
        rng.normal(4.0, 1.0, n),
    )
    turbidity = np.clip(turbidity, 1, 7)

    # Missing values are injected after the draw, so they are missing
    # completely at random and carry no information about the label.
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

    # Seven percent of three columns, chosen at random.
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
