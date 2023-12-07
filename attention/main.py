import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import datetime
import numpy as np

from model import create_model, AttentionPooling, FeatureSelectionModule, LearningModule
from config import *
from utils.load_data import load_data, data_processing
from utils.data_generator import DataGenerator,DataGeneratorCenters, DataGeneratorInterMixUp, DataGeneratorIntraMixUp
from utils.submission import test_submission
from utils.split import train_test_split_patients_

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

from keras.regularizers import L2
from keras.callbacks import EarlyStopping
from keras.models import save_model, load_model

if __name__ == '__main__':

    # Load the data
    df_train, df_test, train_features_dir, test_features_dir = load_data(STORAGE_PATH)
    features_, centers_, patients_, coordinates_, y_train = data_processing(df_train, train_features_dir)

    # Create the model
    input_dim = (features_.shape[1], features_.shape[2])
    model_config = {
        'input_dim': input_dim,
        'output_dim': 1,
        'attention_units': 16,
        'do_feature_selection': False,
        'feature_selection_units': 16,
        'V_init': 'uniform',
        'w_init': 'uniform',
        'learning_rate': 1e-4,
        'use_gated_attention': True,
    }
    model_ = create_model(**model_config)
    model_.summary()

    # Split train and validation sets
    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split_patients_(features_, y_train, patients_, df_train, test_size=0.2, random_state=42)

    # Create the data generators
    train_generator = DataGeneratorInterMixUp(
        X_train,
        y_train,
        batch_size=16,
        same_class=False,
    )
    validation_generator = DataGenerator(
        X_val,
        y_val,
        batch_size=1,
    )

    # Set up callbacks
    early_stopping = EarlyStopping(monitor='val_auc', patience=100, restore_best_weights=True, mode='max')

    # Train the model
    model=model_
    model.fit(
        x=train_generator,
        validation_data=validation_generator,
        epochs=500,
        verbose=1,
        callbacks=[early_stopping],
    )

    # Save the model
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = SAVED_MODELS_PATH / date
    save_model(model, save_path / 'model.hdf5', save_format='hdf5')

    # Load the model
    model = load_model(
        save_path / 'model.hdf5',
        custom_objects={
            'AttentionPooling': AttentionPooling,
            'FeatureSelectionModule': FeatureSelectionModule,
            'LearningModule': LearningModule,
            'L2': L2,
        },
    )

    # Evaluate the model
    print('Evaluating the model...')
    test_submission(df_test, test_features_dir, [model], SAVED_TEST_OUTPUT_PATH, filename='test_output_attention_{}.csv'.format(date))