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


def create_train_val_generator(model, batch_size=128, random_scaling=None):

    train_generator = TrainGenerator(
        TRAINING_FILE, model, batch_size=batch_size, random_reverse=True, random_scaling=random_scaling)
    val_generator = TrainGenerator(
        VALIDATION_FILE, model, batch_size=batch_size)
    return train_generator, val_generator


def split_files_into_train_val(training_file=TRAINING_FILE,
                               validation_ratio=0.2):
    planet_train, planet_val = train_test_split(
        training_file.index.unique(0).values, test_size=validation_ratio)
    return training_file.loc[planet_train], training_file.loc[planet_val]


class Generator(Sequence):
    def __init__(self, files, input_names,
                 output_names=[], batch_size=128,
                 random_reverse=False,
                 random_scaling=None):
        self.files = files
        self._input_names = input_names
        self._output_names = output_names
        self.batch_size = batch_size
        self.random_reverse = random_reverse
        self.random_scaling = random_scaling

    def __len__(self):
        return int(np.ceil(self.files.shape[0]/float(self.batch_size)))

    def __getitem__(self, idx):
        batch_mask = np.arange(self.batch_size * idx,
                               min(self.batch_size * (idx + 1),
                                   self.files.shape[0]))
        batch_stores = self.files['store'][batch_mask]

        batch = read_batch_store(
            batch_stores, self._input_names + self._output_names)

        batch_input = {k: v for k, v in batch.items()
                       if k in self._input_names}
        batch_output = {k: v for k, v in batch.items()
                        if k in self._output_names}

        if self.random_scaling:
            batch_input['feature'] = _apply_random_scaling(
                batch_input['feature'], self.random_scaling)

        if self.random_reverse:
            batch_input['feature'] = _apply_random_reverse(
                batch_input['feature'])

        normalized_batch_input = normalize_features(batch_input)

        if self._output_names:
            return (normalized_batch_input, batch_output)
        else:
            return normalized_batch_input


def read_batch_store(stores, requested_keys):
    values = pd.DataFrame((read_store(s, requested_keys)
                           for s in stores))
    return {col: np.squeeze(np.stack(values[col])) for col in values.columns}


def read_store(storepath, requested_keys):
    with pd.HDFStore(storepath) as store:
        return {key: store[key] for key in requested_keys}


def normalize_features(batch):
    return {key: (batch[key] - MEAN[key])/STD[key] for key in batch.keys()}


def _apply_random_reverse(batch_feature):
    reverse = np.random.choice([0, 1])
    return batch_feature[:, :, ::-1] if reverse else batch_feature


def _apply_random_scaling(batch_feature, scaling_range):
    scale_factor = np.random.rand(
    )*(scaling_range[1] - scaling_range[0]) + scaling_range[0]
    mean_mask = mean_mask = np.concatenate(
        [np.arange(20, dtype=int), np.arange(280, 300, dtype=int)])
    feature_mean = batch_feature[:, :, mean_mask].mean(axis=-1, keepdims=True)
    return (batch_feature - feature_mean)*scale_factor + feature_mean


class TrainGenerator(Generator):
    def __init__(self, files, model, **kwargs):
        Generator.__init__(self, files,
                           model.input_names,
                           output_names=model.output_names,
                           **kwargs)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.files = self.files.sample(frac=1)


class TestGenerator(Generator):
    def __init__(self, files, model, batch_size=128):
        Generator.__init__(self, files,
                           model.input_names,
                           output_names=[],
                           batch_size=batch_size)


def create_callbacks(model_name):
    timestamped = timestamp(model_name)
    return [
        callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3),
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
