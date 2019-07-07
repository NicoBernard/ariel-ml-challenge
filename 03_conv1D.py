# %%
import ariel
import numpy as np
from keras import layers, callbacks, optimizers, Input, Model, Sequential
from keras.models import load_model
import keras.backend as K

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%


def create_cell(x, filters, kernel_size, pool_size):
    conv = layers.SeparableConv1D(
        filters, kernel_size, activation='relu',
        data_format='channels_first')(x)
    pool = layers.MaxPool1D(
        4, data_format='channels_first')(conv)
    return pool


# %%
K.clear_session()

flux = Input(shape=(55, 300), name='feature')

cell1 = create_cell(flux, 32, 5, 4)
cell2 = create_cell(cell1, 64, 5, 4)
cell3 = create_cell(cell2, 128, 5, 4)

flattened = layers.Flatten(name='flatten')(cell3)
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(flattened)

model_name = 'sepconv1d-3layers-maxpool'
model = Model(name=model_name,
              inputs=[flux],
              outputs=[relative_radius])

# %%
model.summary()

# %%
model.compile(optimizers.RMSprop(lr=0.01),
              loss='mae',
              metrics=ariel.METRICS)

# %%
batch_size = 256
train_generator, val_generator = ariel \
    .create_train_val_generator(model,
                                batch_size=batch_size)

# %%
history = model.fit_generator(train_generator,
                              epochs=10, callbacks=ariel.create_callbacks(model_name),
                              validation_data=val_generator,
                              use_multiprocessing=True, workers=4,
                              )

# %%


def create_conv_average_layer(filters, kernel_size, pool_size,
                              layer_idx, pool_type=layers.AveragePooling1D):
