# %%
import ariel
import numpy as np
from keras import layers, callbacks, Input, Model, Sequential
from keras.models import load_model

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%

flux = Input(shape=(55, 300), name='feature')

conv1 = layers.Conv1D(32, 5, activation='relu',
                      data_format='channels_first', name='conv1')(flux)
averagePool1 = layers.AveragePooling1D(
    4, data_format='channels_first', name='averagePool1')(conv1)

conv2 = layers.Conv1D(64, 5, activation='relu',
                      data_format='channels_first', name='conv2')(averagePool1)
averagePool2 = layers.AveragePooling1D(
    4, data_format='channels_first', name='averagePool2')(conv2)

flattened = layers.Flatten(name='flatten')(averagePool2)
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(flattened)

model_name = 'model3'
model = Model(name=model_name,
              inputs=[flux],
              outputs=[relative_radius])

# %%
model.summary()

# %%
model.compile('rmsprop',
              loss='mse',
              metrics=ariel.METRICS)

# %%
batch_size = 128
train_generator, val_generator = ariel.create_train_val_generator(
    batch_size=batch_size)

# %%
history = model.fit_generator(train_generator,
                              epochs=3, callbacks=ariel.create_callbacks(model_name),
                              validation_data=val_generator,
                              use_multiprocessing=True, workers=4,
                              )

# %%

model = load_model("model_checkpoints/2019-07-03T12-02-07_model3.hdf5")

# %%
model.compile('rmsprop', loss='mse',
              metrics=ariel.METRICS)


# %%
