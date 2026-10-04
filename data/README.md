# Dataset

The raw dataset is not included in this repository.

`students_anonymized.csv` is a copy with the personal columns (Student_ID,
First_Name, Last_Name, Email) removed. The app uses it automatically when the
raw file is missing (for example on Streamlit Community Cloud), and all results
are identical. It is created with `python src/anonymize_data.py`.

To use the original file:

1. Download it from <https://www.kaggle.com/datasets/ganeshkumarofficial/student-dataset>
2. Unzip it and place `dcs_student_data.csv` in this folder:

```
data/dcs_student_data.csv
```
