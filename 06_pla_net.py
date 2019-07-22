# %%
import ariel
import numpy as np
from keras import layers, callbacks, optimizers, Input, Model, Sequential
from keras.models import load_model
import keras.backend as K

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
K.clear_session()

flux = Input(shape=(55, 300))

cell1 = ariel.create_multichannel_cell(flux, 32, [3, 5, 7], 4)
cell2 = ariel.create_multichannel_cell(cell1, 64, [3, 5, 7], 4)
cell3 = ariel.create_multichannel_cell(cell2, 128, [3, 5, 7], 4)

flattened = layers.Flatten(name='flatten')(cell3)
dense1 = layers.Dense(256, activation='relu')(flattened)
dense2 = layers.Dense(128, activation='relu')(dense1)
relative_radius = layers.Dense(
    55, activation=None)(dense2)

observation_model = Model(name='3multichannel-2dense',
                          inputs=[flux],
                          outputs=[relative_radius])

feature = Input(shape=(100, 55, 300), name='feature')
multi_pred = layers.TimeDistributed(observation_model)(feature)
mean_pred = layers.Lambda(lambda x: K.mean(
    x, axis=1, keepdims=True))(multi_pred)
relative_radius = layers.Lambda(lambda x: K.repeat_elements(
    x, 100, 1), name='relative_radius')(mean_pred)

model_name = 'planet'
model = Model(name=model_name,
              inputs=[feature],
              outputs=[relative_radius])

# %%
model.summary()

# %%
model.compile('rmsprop',
              loss='mae',
              metrics=ariel.METRICS)

# %%
batch_size = 8
train_generator, val_generator = ariel \
    .create_train_val_generator(model,
                                batch_size=batch_size)

# %%
history = model.fit_generator(train_generator,
                              epochs=5, callbacks=ariel.create_callbacks(model.name),
                              validation_data=val_generator,
                              )


# %%
