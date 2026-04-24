import pandas as pd
import pickle
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def df_summary(df, label=""):
    if label:
        print(f"\n  [{label}]")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Headers: {list(df.columns)}")


# ── cohorts ──────────────────────────────────────────────────────────────────
section("cohorts/")
for f in sorted((DATA / "cohorts").glob("*.csv.gz")):
    df = pd.read_csv(f, compression="gzip", nrows=0)
    full = pd.read_csv(f, compression="gzip")
    df_summary(full, f.name)

# ── folds/mimic_cohort_aplasia_45_days/fold_0.pkl ────────────────────────────
section("folds/mimic_cohort_aplasia_45_days/fold_0.pkl")
with open(DATA / "folds/mimic_cohort_aplasia_45_days/fold_0.pkl", "rb") as fh:
    fold = pickle.load(fh)

if isinstance(fold, dict):
    for key, val in fold.items():
        print(f"\n  Key: '{key}'")
        if isinstance(val, pd.DataFrame):
            print(f"  Shape : {val.shape}")
            print(f"  Dtypes:\n{val.dtypes.to_string()}")
        elif hasattr(val, "shape"):
            print(f"  Shape : {val.shape}  dtype: {val.dtype}")
        else:
            print(f"  Type  : {type(val)}")
elif isinstance(fold, pd.DataFrame):
    print(f"\n  Shape : {fold.shape}")
    print(f"  Dtypes:\n{fold.dtypes.to_string()}")
else:
    print(f"\n  Type : {type(fold)}")

# ── preprocessed_files/aggregated — aplasia only ─────────────────────────────
section("preprocessed_files/aggregated/ (aplasia)")
agg_file = DATA / "preprocessed_files/aggregated/mimic_cohort_aplasia_45_days_agg.csv"
df = pd.read_csv(agg_file)
df_summary(df, agg_file.name)

# ── preprocessed_files/concatenated — aplasia only ───────────────────────────
section("preprocessed_files/concatenated/ (aplasia)")
concat_file = DATA / "preprocessed_files/concatenated/mimic_cohort_aplasia_45_days_concat.csv"
df = pd.read_csv(concat_file)
df_summary(df, concat_file.name)

# ── raw_files — aplasia only ─────────────────────────────────────────────────
section("raw_files/ (aplasia)")
raw_file = DATA / "raw_files/mimic_cohort_aplasia_45_days_admissions_labs_14_days_to_ts.csv.gz"
df = pd.read_csv(raw_file, compression="gzip")
df_summary(df, raw_file.name)

# ── top_features ─────────────────────────────────────────────────────────────
section("top_features/mimic_top100_features.pkl")
with open(DATA / "top_features/mimic_top100_features.pkl", "rb") as fh:
    features = pickle.load(fh)

if isinstance(features, pd.DataFrame):
    print(f"\n  Shape : {features.shape}")
    print(f"  Dtypes:\n{features.dtypes.to_string()}")
elif hasattr(features, "shape"):
    print(f"\n  Shape : {features.shape}  dtype: {features.dtype}")
else:
    print(f"\n  Type  : {type(features)}")
    if hasattr(features, "__len__"):
        print(f"  Length: {len(features)}")
    if isinstance(features, (list, tuple)) and len(features) > 0:
        print(f"  First element type: {type(features[0])}")
        print(f"  Sample: {features[:5]}")