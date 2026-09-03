import pandas as pd


def test_patient_split_uniqueness():
    df = pd.DataFrame({"patient_id": ["a", "a", "b"], "split": ["train", "train", "test"]})
    assert df.groupby("patient_id")["split"].nunique().max() == 1
