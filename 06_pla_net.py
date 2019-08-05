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
    'model_checkpoints/2019-07-25T07-37-28_3multichannel-2dense.hdf5')

feature = Input(shape=(100, 55, 300), name='feature')
reshaped_feature = layers.Reshape((100, 55*300))(feature)
extra_feature = Input(shape=(100, 6,), name='extra_feature')

merged = layers.Concatenate(axis=2)([reshaped_feature, extra_feature])

obs_model_from_slice = layers.Lambda(
    lambda x: observation_model([K.reshape(x[:, :-6], (-1, 55, 300)), x[:, -6:]]))

multi_pred = layers.TimeDistributed(obs_model_from_slice)(merged)
mean_pred = layers.Lambda(lambda x: K.mean(
    x, axis=1, keepdims=True))(multi_pred)
relative_radius = layers.Lambda(lambda x: K.repeat_elements(
    x, 100, 1), name='relative_radius')(mean_pred)

model_name = 'extra_feat'
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
batch_size = 8
_, val_generator = ariel \
    .create_planetwise_generators(model,
                                  batch_size=batch_size)

# %%
model.evaluate_generator(val_generator, verbose=True)

# %%
test_gen = ariel.create_test_generator(model)
predictions = model.predict_generator(test_gen,
                                      verbose=True)


# %%
filename = 'upload/%s.txt' % ariel.timestamp(model.name)
np.savetxt(filename, predictions.reshape((62900, 55)), fmt='%.13f')


# %%
