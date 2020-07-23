## Just to 
import os
os.environ["OMP_NUM_THREADS"] = "1"
# %%
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except (ModuleNotFoundError,NameError):
    print('Running outisde of a ipython kernel')
# # %%

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

except (ModuleNotFoundError,NameError):
    print('Running outisde of a ipython kernel')
# %%


import matplotlib.pyplot as plt
#import scipy as sp
import numpy as np
#from scipy import constants
#from scipy import interpolate
#import pickle
from pprint import pprint

#mu = 1.
#nm = 1e-3
#mm = 1e3
#m = 1e6

np.set_printoptions(precision=4)
#eV2nm = constants.c * constants.h / constants.elementary_charge * 1e9
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.facecolor'] = 'w'

#import ipywidgets as widgets
#from ipywidgets import (interact, interactive, fixed, interact_manual,
#                        interactive_output)
#from IPython import display as dpl
#from IPython.display import Latex


# %%
pprint('Loaded typical imports v 0.0.1')
