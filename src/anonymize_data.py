"""
Create an anonymised copy of the dataset that is safe to publish.

The copy keeps every row (in the same order) and every column EXCEPT the
personal ones (Student_ID, First_Name, Last_Name, Email). The normal
cleaning pipeline runs on it unchanged, so all results are identical.

It is used by the deployed dashboard, where the raw Kaggle file is not
available. Locally the raw file is still preferred when it exists.

Run:  python src/anonymize_data.py
"""

from config import ANONYMIZED_CSV, PII_COLUMNS
from data_loader import DataError, find_csv, load_raw_data


def main():
    source = find_csv()
    if source == ANONYMIZED_CSV:
        raise DataError("The raw Kaggle CSV is needed to create the anonymised copy.")
    raw = load_raw_data(source)
    anonymized = raw.drop(columns=[c for c in PII_COLUMNS if c in raw])
    anonymized.to_csv(ANONYMIZED_CSV, index=False)
    print(f"Saved {len(anonymized)} rows without {PII_COLUMNS} to {ANONYMIZED_CSV}")


if __name__ == "__main__":
    try:
        main()
    except DataError as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)
