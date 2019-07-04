import tensorflow as tf
from keras import callbacks
from keras.utils import Sequence
import keras.backend as K
import pandas as pd
import re
import glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

with pd.HDFStore('preprocessing/preprocessing.h5') as preprocessing:
    FEATURE_MEAN = preprocessing['feature_mean'].transpose().values
    FEATURE_STD = preprocessing['feature_std'].transpose().values
    EXTRA_FEATURE_MEAN = preprocessing['extra_feature_mean'].transpose().values
    EXTRA_FEATURE_STD = preprocessing['extra_feature_std'].transpose().values
    TRAINING_FILE = preprocessing['training_file']
    TEST_FILE = preprocessing['test_file']


_EXTRA_FEATURE_REGEX = re.compile(r"(?<=:\s)[0-9\.]+")
_STORE_KEYS = ['feature', 'extra_feature', 'relative_radius', 'orbit']

METRICS = {'relative_radius': 'mae'}


def read_file(filepath):
    feature = pd.read_csv(filepath, sep='\t',
                          comment='#',
                          header=None, dtype=float)
    extra_feature = pd.Series(np.fromregex(filepath,
                                           _EXTRA_FEATURE_REGEX,
                                           dtype=[('val', 'float')])['val'])
    return feature, extra_feature


def read_batch(files):
    feature, extra_feature = zip(
        *(read_file(f) for f in files))
    return np.squeeze(np.stack(feature)), np.stack(extra_feature)


def create_train_val_generator(batch_size=128):

    train_files, val_files = split_files_into_train_val()
    train_generator = TrainGenerator(train_files, batch_size=batch_size)
    val_generator = TrainGenerator(val_files, batch_size=batch_size)
    return train_generator, val_generator


def split_files_into_train_val(training_file=TRAINING_FILE,
                               validation_ratio=0.2):
    planet_train, planet_val = train_test_split(
        training_file.index.unique(0).values, test_size=validation_ratio)
    return training_file.loc[planet_train], training_file.loc[planet_val]


class TrainGenerator(Sequence):
    def __init__(self, files, batch_size=128):
        self.randomized_files = files.sample(frac=1)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.randomized_files.shape[0]/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_mask = np.arange(self.batch_size * idx,
                               min(self.batch_size * (idx + 1),
                                   self.randomized_files.shape[0]))
        batch_stores = self.randomized_files['store'][batch_mask]

        batch = read_batch_store(batch_stores, is_training=True)
        normalized_batch_feature, normalized_batch_extra_features = normalize_features(
            batch['feature'], batch['extra_feature'])
        return ({'feature': normalized_batch_feature,
                 'extra_feature': normalized_batch_extra_features},
                {'relative_radius': batch['relative_radius'],
                 'orbit': batch['orbit']})


def read_batch_store(stores, is_training=True):
    values = pd.DataFrame((read_store(s, is_training=is_training)
                           for s in stores))
    return {col: np.squeeze(np.stack(values[col])) for col in values.columns}


def read_store(storepath, is_training=True):
    with pd.HDFStore(storepath) as store:
        if is_training:
            return {key: store[key] for key in _STORE_KEYS}
        else:
            return {key: store[key] for key in _STORE_KEYS[:2]}


def normalize_features(batch_flux, batch_extra_features):
    normalized_batch_flux = (
        batch_flux - np.expand_dims(FEATURE_MEAN, -1)) / np.expand_dims(FEATURE_STD, -1)
    normalized_batch_extra_features = (
        batch_extra_features - EXTRA_FEATURE_MEAN) / EXTRA_FEATURE_STD
    return normalized_batch_flux, normalized_batch_extra_features


class TestGenerator(Sequence):
    def __init__(self, files, batch_size=128):
        self.files = files
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.files.shape[0]/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_mask = np.arange(self.batch_size * idx,
                               min(self.batch_size * (idx + 1),
                                   self.files.shape[0]))
        batch_stores = self.files['store'][batch_mask]

        batch = read_batch_store(batch_stores, is_training=False)
        normalized_batch_feature, normalized_batch_extra_features = normalize_features(
            batch['feature'], batch['extra_feature'])
        return {'feature': normalized_batch_feature,
                'extra_feature': normalized_batch_extra_features}


def create_callbacks(model_name):
    timestamped = timestamp(model_name)
    return [
        callbacks.TensorBoard(log_dir="./logs/%s" %
                              timestamped, update_freq=1024),
        callbacks.ModelCheckpoint('model_checkpoints/%s.hdf5' % timestamped),
        callbacks.CSVLogger('history/%s.csv' % timestamped),
    ]


def timestamp(model_name):
    return "%s_%s" \
        % (datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), model_name)
