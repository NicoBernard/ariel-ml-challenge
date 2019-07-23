import tensorflow as tf
from keras import callbacks, layers
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
    TRAINING_FILE = preprocessing['training_file']
    VALIDATION_FILE = preprocessing['validation_file']
    TEST_FILE = preprocessing['test_file']
    MEAN = {
        'feature': np.expand_dims(preprocessing['feature_mean'].transpose().values, -1),
        'extra_feature': preprocessing['extra_feature_mean'].transpose().values,
    }
    STD = {
        'feature': np.expand_dims(preprocessing['feature_std'].transpose().values, -1),
        'extra_feature': preprocessing['extra_feature_std'].transpose().values,
    }


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


def create_train_val_generator(model, batch_size=16):

    train_generator = TrainGenerator(
        TRAINING_FILE, model, batch_size=batch_size)
    val_generator = TrainGenerator(
        VALIDATION_FILE, model, batch_size=batch_size)
    return train_generator, val_generator


def split_files_into_train_val(training_file=TRAINING_FILE,
                               validation_ratio=0.2):
    planet_train, planet_val = train_test_split(
        training_file.index.unique(0).values, test_size=validation_ratio)
    return training_file.loc[planet_train], training_file.loc[planet_val]


class Generator(Sequence):

    def __init__(self, files, input_names, output_names=[], batch_size=16):
        self.files = files['pickle'].unique()
        self._pickle_reader = CachedPickleReader()
        self._input_names = input_names
        self._output_names = output_names
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.files.shape[0]/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_mask = np.arange(self.batch_size * idx,
                               min(self.batch_size * (idx + 1),
                                   self.files.shape[0]))
        batch_files = self.files[batch_mask]

        batch = self._get_batch(batch_files)

        batch_input = {k: v for k, v in batch.items()
                       if k in self._input_names}
        batch_output = {k: v for k, v in batch.items()
                        if k in self._output_names}
        normalized_batch_input = normalize_features(batch_input)

        if self._output_names:
            return (normalized_batch_input, batch_output)
        else:
            return normalized_batch_input

    def _get_batch(self, batch_files):
        filewise = pd.DataFrame((self._pickle_reader.read(f)
                                 for f in batch_files))
        return {col: np.squeeze(np.stack(filewise[col])) for col in filewise.columns}


class CachedPickleReader(object):

    def __init__(self):
        self._data = {}

    def read(self, pickle):
        if pickle not in self._data:
            self._data[pickle] = read_pickle(pickle)

        return self._data[pickle]


def read_pickle(file):
    df = pd.read_pickle(file)
    return {col: np.squeeze(np.stack(df[col])) for col in df.columns}


def normalize_features(batch):
    return {key: (batch[key] - MEAN[key])/STD[key] for key in batch.keys()}


class TrainGenerator(Generator):
    def __init__(self, files, model, batch_size=16):
        Generator.__init__(self, files,
                           model.input_names,
                           output_names=model.output_names,
                           batch_size=batch_size)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.files = np.random.permutation(self.files)

    def _get_batch(self, batch_files):
        planetwise_batch = super()._get_batch(batch_files)
        observationwise_batch = {key: batch.reshape((-1, *batch.shape[2:]))
                                 for key, batch in planetwise_batch.items()}
        return observationwise_batch


class TestGenerator(Generator):
    def __init__(self, files, model, batch_size=16):
        Generator.__init__(self, files,
                           model.input_names,
                           output_names=[],
                           batch_size=batch_size)


def create_callbacks(model_name):
    timestamped = timestamp(model_name)
    return [
        callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, verbose=True, patience=2),
        callbacks.EarlyStopping(
            monitor='loss', patience=2,
            restore_best_weights=True, verbose=True),
        callbacks.TensorBoard(log_dir="./logs/%s" %
                              timestamped, update_freq=1024),
        callbacks.ModelCheckpoint('model_checkpoints/%s.hdf5' % timestamped),
        callbacks.CSVLogger('history/%s.csv' % timestamped),
    ]


def timestamp(model_name):
    return "%s_%s" \
        % (datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), model_name)


def create_channel(x, filters, kernel_size, pool_size):
    conv1 = layers.SeparableConv1D(
        filters, kernel_size, activation='relu',
        data_format='channels_first')(x)
    pool = layers.MaxPool1D(
        pool_size, data_format='channels_first')(conv1)
    return pool


def create_multichannel_cell(x, filters, channel_kernel_size, pool_size):
    cells = [create_channel(x, filters, kernel_size, pool_size)
             for kernel_size in channel_kernel_size]
    return layers.Concatenate()(cells)
