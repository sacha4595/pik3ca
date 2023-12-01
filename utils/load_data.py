from pathlib import Path
import pandas as pd
import numpy as np
import tqdm


def load_data(data_dir:str):

    # Data path to the root folder
    data_dir = Path(data_dir)

    # Load training and validation sets
    train_features_dir = data_dir / "train_input" / "moco_features"
    test_features_dir = data_dir / "test_input" / "moco_features"
    df_train = pd.read_csv(data_dir  / "supplementary_data" / "train_metadata.csv")
    df_test = pd.read_csv(data_dir  / "supplementary_data" / "test_metadata.csv")

    # Merge features and metadata
    y_train = pd.read_csv(data_dir  / "train_output.csv")
    df_train = df_train.merge(y_train, on="Sample ID")

    return df_train, df_test, train_features_dir, test_features_dir

def data_processing(df_train, train_features_dir):

    features_train = []
    centers_train = []
    patients_train = []
    coordinates_train = []
    y_train = []

    for sample, label, center, patient in tqdm.tqdm(
        df_train[["Sample ID", "Target", "Center ID", "Patient ID"]].values,
        desc = 'Processing training data'
    ):
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        coordinates, features = _features[:, :3], _features[:, 3:]  # Ks

        y_train.append(label)
        features_train.append(features)
        centers_train.append(center)
        patients_train.append(patient)
        coordinates_train.append(coordinates)

    # convert to numpy arrays
    y_train = np.array(y_train)
    features_train = np.array(features_train)
    centers_train = np.array(centers_train)
    patients_train = np.array(patients_train)
    coordinates_train = np.array(coordinates_train)

    return features_train, centers_train, patients_train, coordinates_train, y_train