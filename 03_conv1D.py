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

feature = Input(shape=(55, 300), name='feature')
extra_feature = Input(shape=(6,), name='extra_feature')
repeated_extra_feature = layers.Lambda(lambda x: K.repeat_elements(
    K.expand_dims(x, -1), 300, -1), name='repeated_extra_feature')(extra_feature)

merged = layers.Concatenate(axis=1)([feature, repeated_extra_feature])

cell1 = ariel.create_multichannel_cell(merged, 32, [3, 5, 7], 4)
cell2 = ariel.create_multichannel_cell(cell1, 64, [3, 5, 7], 4)
cell3 = ariel.create_multichannel_cell(cell2, 128, [3, 5, 7], 4)

flattened = layers.Flatten(name='flatten')(cell3)
dense1 = layers.Dense(256, activation='relu')(flattened)
dense2 = layers.Dense(128, activation='relu')(dense1)
relative_radius = layers.Dense(
    55, activation='sigmoid', name='relative_radius')(dense2)

model_name = 'sigmoid-activation'
model = Model(name=model_name,
              inputs=[feature, extra_feature],
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
                              epochs=20, callbacks=ariel.create_callbacks(model.name),
                              validation_data=val_generator,
                              verbose=True,
                              )


# %%
