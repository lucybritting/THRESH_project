import pandas as pd
from pathlib import Path


FILE_DIR = Path(__file__).parent.parent / "output/range_analysis/cohort_aplasia_with_ranges.csv.gz"
OUTPUT_DIR = Path(__file__).parent.parent / "output/cohort_ranges_first_500_rows.csv"

df = pd.read_csv(FILE_DIR)
print("Columns: ", list(df.columns))
print("Rows: ", df.shape)
print("First 5 rows: ")
print(df.head(100))

df.head(500).to_csv(OUTPUT_DIR, index=False)