from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

def test_submission(df_test,test_features_dir,models,data_dir,filename='benchmark_test_output.csv'):

    data_dir = Path(data_dir)

    features_train = []

    # load the data from `df_test` (~ 1 minute)
    for sample in tqdm(df_test["Sample ID"].values,desc='Loading test data'):
        _features = np.load(test_features_dir / sample)
        coordinates, features = _features[:, :3], _features[:, 3:]
        features_train.append(features)

    # convert to numpy arrays
    features_train = np.array(features_train)

    preds_test = 0
    # loop over the classifiers
    for model in models:
        preds_test += model.predict(features_train, batch_size=1).flatten()
    # and take the average (ensembling technique)
    preds_test = preds_test / len(models)

    submission = pd.DataFrame(
        {"Sample ID": df_test["Sample ID"].values, "Target": preds_test}
    ).sort_values(
        "Sample ID"
    )  # extra step to sort the sample IDs

    # sanity checks
    assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
    assert submission.shape == (149, 2), "Your submission file must be of shape (149, 2)"
    assert list(submission.columns) == [
        "Sample ID",
        "Target",
    ], "Your submission file must have columns `Sample ID` and `Target`"

    # save the submission as a csv file
    submission.to_csv(data_dir / filename, index=None)