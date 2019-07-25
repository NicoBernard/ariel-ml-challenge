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

flux = Input(shape=(55, 300), name='feature')

cell1 = ariel.create_multichannel_cell(flux, 32, [3, 5, 7], 4)
cell2 = ariel.create_multichannel_cell(cell1, 64, [3, 5, 7], 4)
cell3 = ariel.create_multichannel_cell(cell2, 128, [3, 5, 7], 4)

flattened = layers.Flatten(name='flatten')(cell3)
dense1 = layers.Dense(256, activation='relu')(flattened)
dense2 = layers.Dense(128, activation='relu')(dense1)
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(dense2)

model_name = '3multichannel-2dense'
model = Model(name=model_name,
              inputs=[flux],
              outputs=[relative_radius])

# %%
model.summary()

# %%
model.compile('rmsprop',
              loss='mae',
              metrics=ariel.METRICS)

# %%
batch_size = 128
train_generator, val_generator = ariel \
    .create_observationwise_generators(model,
                                       batch_size=batch_size)

# %%
history = model.fit_generator(train_generator,
                              epochs=10, callbacks=ariel.create_callbacks(model.name),
                              validation_data=val_generator,
                              )


# %%
