# %%
import ariel
import numpy as np
from keras import layers, callbacks, Input, Model, Sequential
from keras.models import load_model

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Vanilla model

flux = Input(shape=(55, 300), name='flux')
flattened_flux = layers.Flatten()(flux)
extra_features = Input(shape=(6,), name='extra_feature')
layer1 = layers.Concatenate()([flattened_flux, extra_features])

layer2 = layers.Dense(32, activation='relu')(layer1)
orbit = layers.Dense(2, activation=None, name='orbit')(layer2)
layer3 = layers.Concatenate()([layer2, orbit])
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(layer3)

model_name = 'model1'
model1 = Model(name=model_name,
               inputs=[flux, extra_features],
               outputs=[relative_radius, orbit])

# %%
model1.summary()

# %% []
# Evaluate losses magnitude
model1.compile('rmsprop', loss='mse')

orbit_loss, radius_loss, _ = model1.evaluate(*train_generator[9])
print(np.ceil(np.log10(orbit_loss)))

# %%
model1.compile('rmsprop',
               loss='mse', loss_weights={'orbit': 1e-20, 'relative_radius': 1},
               metrics={'relative_radius': 'mse'})

# %%
batch_size = 128
train_generator, val_generator = ariel.create_train_val_generator(
    batch_size=batch_size)

# %%
history1 = model1.fit_generator(train_generator,
                                epochs=5, callbacks=ariel.create_callbacks(model_name),
                                validation_data=val_generator,
                                use_multiprocessing=True, workers=4,
                                )
