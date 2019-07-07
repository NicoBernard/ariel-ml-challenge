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
    conv1 = layers.SeparableConv1D(
        filters, kernel_size, activation='relu',
        data_format='channels_first')(x)
    conv2 = layers.SeparableConv1D(
        filters, kernel_size, activation='relu',
        data_format='channels_first')(conv1)
    pool = layers.MaxPool1D(
        pool_size, data_format='channels_first')(conv2)
    return pool


def create_multichannel_cell(x, filters, channel_kernel_size, pool_size):
    cells = [create_cell(x, filters, kernel_size, pool_size)
             for kernel_size in channel_kernel_size]
    return layers.Concatenate()(cells)


# %%
K.clear_session()

flux = Input(shape=(55, 300), name='feature')

cell1 = create_multichannel_cell(flux, 32, [1, 3, 5], 2)
cell2 = create_multichannel_cell(cell1, 64, [1, 3, 5], 2)
cell3 = create_multichannel_cell(cell2, 128, [1, 3, 5], 2)

flattened = layers.Flatten(name='flatten')(cell3)
relative_radius = layers.Dense(
    55, activation=None, name='relative_radius')(flattened)

model_name = 'multichannel-3layers'
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
