import numpy as np
from keras import layers
import random

# %%

hyperparameters = {
    'n_conv_layer': np.arange(2, 6),
    'first_conv_kernels': [16, 32, 64, 128],
    'kernel_size': [3, 5, 7, 9],
    'activation': ['relu', 'elu']
    'pool_type': [layers.AveragePooling1D, layers.MaxPooling1D],
    'pool_size': [2, 4, 8],
    'loss': ['mse', 'mae'],
    'batch_size': [32, 64, 128, 256, 512],
}

# %%
drawings = {k: np.random.choice(v, size=(100,))
            for k, v in hyperparameters.items()}
