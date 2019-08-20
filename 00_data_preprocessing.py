# %% [markdown]
# # Preprocessings
# This notebook contains all the small preprocessing used for training.

# %%
import ariel
import tarfile
import tensorflow as tf
import pandas as pd
import re
import glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import os

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Converting data in a more manageable format

# %%

tar = tarfile.open("database.tar")
tar.extractall(path="data/")
tar.close()


# %%
noisy_test_file = "data/noisy_test/0005_01_01.txt"
feature_regex = re.compile(r"(?<=:\s)\d+\.\d+")
feature_bis = np.fromregex(
    noisy_test_file, feature_regex, dtype=[('val', np.float64)])
feature_bis['val']


# %%
np.loadtxt(noisy_test_file)


# %%
extra_feature_regex = re.compile(r"(?<=:\s)\d+\.\d+")


def read_file(filepath):
    feature = pd.read_csv(filepath, sep='\t', comment='#',
                          header=None, dtype='float')
    extra_feature = np.fromregex(filepath, extra_feature_regex, dtype=[
                                 ('val', 'float')])['val']
    return feature, extra_feature


# %%
feature, extra_feature = read_file("data/noisy_train/0001_01_02.txt")
feature


# %%
data.std(axis=1).shape

# %% [markdown]
# ## Extracting mean and sqrt

# %%
noisy_train_list = glob.glob("data/noisy_train/*.txt")

# %%
noisy_test_list = glob.glob("data/noisy_test/*.txt")
test_files = pd.DataFrame.from_dict({'noisy': noisy_test_list})

with pd.HDFStore('preprocessing/preprocessing.h5') as store:
    store['test_file'] = test_files

# %%
train_file_number = len(noisy_train_list)


# %%
feature_mean_by_file = np.zeros((train_file_number, 55))
feature_std_by_file = np.zeros((train_file_number, 55))
extra_feature_by_file = np.zeros((train_file_number, 6))


# %%
for iFile, file in enumerate(noisy_train_list):
    print("reading file %d/%d" % (iFile + 1, train_file_number), end='\r')
    feature, extra_feature = read_file(file)
    feature_mean_by_file[iFile] = feature.mean(axis=1)
    feature_std_by_file[iFile] = feature.std(axis=1)
    extra_feature_by_file[iFile] = extra_feature


# %%
feature_mean = feature_mean_by_file.mean(axis=0)
feature_std = feature_std_by_file.mean(axis=0)


# %%
feature_mean.shape


# %%
extra_feature_mean = extra_feature_by_file.mean(axis=0)
extra_feature_std = extra_feature_by_file.std(axis=0)


# %%
with pd.HDFStore('preprocessing/preprocessing.h5') as store:
    store['feature_mean_by_file'] = pd.DataFrame(
        feature_mean_by_file, index=noisy_train_list)
    store['feature_std_by_file'] = pd.DataFrame(
        feature_std_by_file, index=noisy_train_list)
    store['feature_mean'] = pd.DataFrame(feature_mean)
    store['feature_std'] = pd.DataFrame(feature_std)

    store['extra_feature_by_file'] = pd.DataFrame(
        extra_feature_by_file, index=noisy_train_list)
    store['extra_feature_mean'] = pd.DataFrame(extra_feature_mean)
    store['extra_feature_std'] = pd.DataFrame(extra_feature_std)

# %% [markdown]
# ## Creating data generators

# %%
selected_files = ariel.TRAIN_FILES[:2]
feature, extra_features = list(
    zip(*(ariel.read_file(f) for f in selected_files)))
np.stack(feature).shape


# %%
def read_batch(files):
    feature, extra_feature = zip(*(ariel.read_file(f) for f in selected_files))
    return np.stack(feature), np.stack(extra_feature)


# %%
batch_feat, batch_extra = ariel.read_batch(selected_files)


# %%
print(batch_feat.shape)
print(batch_extra.shape)


# %%
raw_index = pd.read_csv('data/noisy_train.txt',
                        header=None,
                        squeeze=True).str.extract(r'(?P<planet>\d+)_(?P<spot>\d+)_(?P<photon>\d+)')
index = pd.MultiIndex.from_frame(raw_index)


# %%
planet_train, planet_val = train_test_split(
    index.unique(0).values, test_size=0.2)


# %%
training_files = pd.DataFrame.from_dict(
    {'noisy': glob.glob("data/noisy_train/*.txt"),
     'params': glob.glob("data/params_train/*.txt")})
training_files.index = index


# %%
planet_train


# %%
training_files.loc[planet_train]


# %%
with pd.HDFStore('preprocessing/preprocessing.h5') as store:
    store['training_file'] = training_files


# %%
train_files, validation_files = ariel.split_files_into_train_val()


# %%
def generator(files, batch_size=32):

    randomized_files = files.sample(frac=1)
    i_batch = 0
    while True:

        batch_mask = np.arange(batch_size*i_batch, batch_size*(i_batch+1))
        batch_noisy_files = randomized_files['noisy'][batch_mask]
        batch_params_files = randomized_files['noisy'][batch_mask]
        batch_flux, batch_extra_features = ariel.read_batch(batch_noisy_files)
        batch_relative_radius, batch_orbit = ariel.read_batch(
            batch_params_files)
        normalized_batch_flux = (
            batch_flux - np.expand_dims(ariel.FEATURE_MEAN, -1))/np.expand_dims(ariel.FEATURE_STD, -1)
        normalized_batch_extra_features = (
            batch_extra_features - ariel.EXTRA_FEATURE_MEAN)/ariel.EXTRA_FEATURE_STD
        yield (normalized_batch_flux, normalized_batch_extra_features), (batch_relative_radius, batch_orbit)

        i_batch += 1


# %%
batch_size = 32
i_batch = 0


# %%
batch_mask = np.arange(batch_size*i_batch, batch_size*(i_batch+1))
batch_mask


# %%
files = train_files
randomized_files = files.sample(frac=1)


# %%
batch_noisy_files = randomized_files['noisy'][batch_mask]
batch_noisy_files


# %%
batch_params_files = randomized_files['noisy'][batch_mask]
batch_params_files


# %%
batch_flux, batch_extra_features = ariel.read_batch(batch_noisy_files)


# %%
batch_flux.shape


# %%
normalized_batch_flux = (
    batch_flux - np.expand_dims(ariel.FEATURE_MEAN, -1))/np.expand_dims(ariel.FEATURE_STD, -1)


# %%
normalized_batch_extra_features = (
    batch_extra_features - ariel.EXTRA_FEATURE_MEAN)/ariel.EXTRA_FEATURE_STD


# %%
gen = ariel.generator(train_files)
toto = next(gen)

# %%

train_file_number = ariel.TRAINING_FILE.shape[0]
iFile = 0
for observation, noisy, params in ariel.TRAINING_FILE.itertuples():
    store_name = "preprocessing/%s.hdf5" % "_".join(observation)
    print("reading file %d/%d" % (iFile + 1, train_file_number), end='\r')
    with pd.HDFStore(store_name) as store:
        store['feature'], store['extra_feature'] = ariel.read_file(noisy)
        store['relative_radius'], store['orbit'] = ariel.read_file(params)

    iFile += 1


# %%

test_file_number = ariel.TEST_FILE.shape[0]
iFile = 0
for _, noisy in ariel.TEST_FILE.itertuples():
    store_name = "preprocessing/test/%s.hdf5" % Path(noisy).stem
    print("progress: %.2f%%" % ((iFile + 1)*100 / test_file_number), end='\r')
    with pd.HDFStore(store_name) as store:
        store['feature'], store['extra_feature'] = ariel.read_file(noisy)

    iFile += 1

# %%

training_file = ariel.TRAINING_FILE
store_file = training_file.noisy \
    .str.replace('data/noisy_train', 'preprocessing/training') \
    .str.replace('txt', 'hdf5')
updated_training_file = training_file.assign(store=store_file)

# %%

test_file = ariel.TEST_FILE
store_file = test_file.noisy \
    .str.replace('data/noisy_test', 'preprocessing/test') \
    .str.replace('txt', 'hdf5')
updated_test_file = test_file.assign(store=store_file)

# %%
with pd.HDFStore('preprocessing/preprocessing.h5') as preprocessing:
    preprocessing['training_file'] = updated_training_file
    preprocessing['test_file'] = updated_test_file


# %%
train, val = ariel.split_files_into_train_val()

# %%

updated_val = val.assign(store=val.store.str.replace('training', 'validation'))


# %%
for old, new in zip(val.store, updated_val.store):
    os.rename(old, new)


# %%
with pd.HDFStore('preprocessing/preprocessing.h5') as preprocessing:
    preprocessing['training_file'] = train
    preprocessing['validation_file'] = updated_val


# %%
raw_index = pd.read_csv('data/noisy_test.txt',
                        header=None,
                        squeeze=True).str.extract(r'(?P<planet>\d+)_(?P<spot>\d+)_(?P<photon>\d+)')
index = pd.MultiIndex.from_frame(raw_index)


# %%
updated_test_file = ariel.TEST_FILE.set_index(index)

# %%
with pd.HDFStore('preprocessing/preprocessing.h5') as preprocessing:
    preprocessing['test_file'] = updated_test_file


# %%
training_file = ariel.TRAINING_FILE
planet_idx = training_file.index.unique(0).values

# %%
stacked = pd.DataFrame(observations, index=planet_store.index)

# %%
for i_planet, idx in enumerate(planet_idx):

    print("progress: %.2f%%" % (i_planet*100 / len(planet_idx)), end='\r')
    planet_store = training_file.store[idx]
    observations = [ariel.read_store(planet_store[obs_idx], ariel._STORE_KEYS)
                    for obs_idx in planet_store.index]
    stacked = pd.DataFrame(observations, index=planet_store.index)
    stacked.to_pickle('preprocessing/training/%s.pkl' % idx)

# %%
valid_file = ariel.VALIDATION_FILE
planet_idx = valid_file.index.unique(0).values

for i_planet, idx in enumerate(planet_idx):

    print("progress: %.2f%%" % (i_planet*100 / len(planet_idx)), end='\r')
    planet_store = valid_file.store[idx]
    observations = [ariel.read_store(planet_store[obs_idx], ariel._STORE_KEYS)
                    for obs_idx in planet_store.index]
    stacked = pd.DataFrame(observations, index=planet_store.index)
    stacked.to_pickle('preprocessing/validation/%s.pkl' % idx)


# %%
test_file = ariel.TEST_FILE
planet_idx = test_file.index.unique(0).values

for i_planet, idx in enumerate(planet_idx):

    print("progress: %.2f%%" % (i_planet*100 / len(planet_idx)), end='\r')
    planet_store = test_file.store[idx]
    observations = [ariel.read_store(planet_store[obs_idx], ariel._STORE_KEYS[:2])
                    for obs_idx in planet_store.index]
    stacked = pd.DataFrame(observations, index=planet_store.index)
    stacked.to_pickle('preprocessing/test/%s.pkl' % idx)

# %%

updated_training_file = ariel.TRAINING_FILE
updated_training_file['pickle'] = 'preprocessing/training/' + \
    updated_training_file.index.get_level_values(0).astype(str) + '.pkl'

# %%
updated_test_file = ariel.TEST_FILE
updated_test_file['pickle'] = 'preprocessing/test/' + \
    updated_test_file.index.get_level_values(0).astype(str) + '.pkl'

# %%
updated_val_file = ariel.VALIDATION_FILE
updated_val_file['pickle'] = 'preprocessing/validation/' + \
    updated_val_file.index.get_level_values(0).astype(str) + '.pkl'


# %%
with pd.HDFStore('preprocessing/preprocessing.h5') as preprocessing:
    preprocessing['training_file'] = updated_training_file
    preprocessing['validation_file'] = updated_val_file
    preprocessing['test_file'] = updated_test_file

# %%
