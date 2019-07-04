# %%
import ariel
import numpy as np
from keras import layers, callbacks, Input, Model, Sequential
from keras.models import load_model

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Vanilla model

flux = Input(shape=(55, 300), name='feature')
flattened_flux = layers.Flatten()(flux)
layer1 = layers.Dense(32, activation='relu')(flattened_flux)
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(layer1)

model_name = 'model2'
model2 = Model(name=model_name,
               inputs=[flux],
               outputs=[relative_radius])

# %%
model2.summary()

# %%
model2.compile('rmsprop',
               loss='mse',
               metrics=ariel.METRICS)

# %%
batch_size = 128
train_generator, val_generator = ariel.create_train_val_generator(
    batch_size=batch_size)

# %%
history2 = model2.fit_generator(train_generator,
                                epochs=5, callbacks=ariel.create_callbacks(model_name),
                                validation_data=val_generator,
                                use_multiprocessing=True, workers=4,
                                )
