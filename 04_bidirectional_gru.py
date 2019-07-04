# %%
import ariel
import numpy as np
from keras import layers, callbacks, Input, Model, Sequential
from keras.models import load_model
from keras import backend as K


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
K.clear_session()

flux = Input(shape=(55, 300), name='feature')
transposed = layers.Permute((2, 1))(flux)

bidirectional_1 = layers.Bidirectional(
    layers.GRU(64), name='bidirectional_1')(transposed)
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(bidirectional_1)

model_name = 'bidir_gru
'
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
