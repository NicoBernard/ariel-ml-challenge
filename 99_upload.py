
# %%
import ariel
from keras.models import load_model
from datetime import datetime
import numpy as np

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Make prediction on noisy_test files

model = load_model('model_checkpoints/2019-07-02_model3.hdf5')

# %%
test_gen = ariel.TestGenerator(ariel.TEST_FILE)
predictions = model.predict_generator(test_gen,
                                      use_multiprocessing=True,
                                      workers=4,
                                      verbose=True)

# %% [markdown]
# ## Write prediction to time-stamped file

# %%
filename = 'upload/%s.txt' % ariel.timestamp(model.name)
np.savetxt(filename, predictions, fmt='%.13f')

# %%
