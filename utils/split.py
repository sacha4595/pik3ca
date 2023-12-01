import numpy as np
from sklearn.model_selection import train_test_split

def train_test_split_patients_(X,y, patients_, df_train, test_size=0.2,random_state=42):
    """
    Split the data into train and test sets, ensuring that no patient is in both sets.
    """
    # Get the list of unique patients
    patients = np.unique(patients_)
    # Shuffle the patients
    np.random.seed(random_state)
    np.random.shuffle(patients)
    # Split the patients into train and test sets
    patients_train, patients_test = train_test_split(patients, test_size=test_size, random_state=random_state)
    df_train,df_val = train_test_split(df_train, test_size=test_size, random_state=42)
    # Get the indexes of the patients in the train and test sets
    idx_train = np.where(np.isin(patients_, patients_train))[0]
    idx_test = np.where(np.isin(patients_, patients_test))[0]
    # Split the data into train and test sets
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    return X_train, X_test, y_train, y_test, df_train, df_val