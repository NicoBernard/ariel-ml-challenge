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

observation_model = load_model(
    'model_checkpoints/2019-07-24T07-34-22_3multichannel-2dense.hdf5')

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
_, val_generator = ariel \
    .create_planetwise_generators(model,
                                  batch_size=batch_size)

# %%
model.evaluate_generator(val_generator, verbose=True)


# %%
