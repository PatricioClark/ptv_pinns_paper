# TF2.0 LES Channel test

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN

from   dom import *
from   mod import *
import numpy as np
import time
import random

# Get parameters
params = Run()
params.add_params(f"data/dset.py")

# NN params
layers = [4]+[params.hu]*params.layers+[4]

# Load data
X_data, Y_data, lambda_data, lambda_phys = generate_data(params)

# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
means[-1] = params.P
stds      = Y_data.std(0)
stds[-1]  = params.sig_p
onorm = [means, stds]

# Optimizer scheduler
if params.depochs:
    dsteps = params.depochs*len(X_data)/params.mbsize
    params.lr = keras.optimizers.schedules.ExponentialDecay(params.lr,
                                                            dsteps, 
                                                            params.drate)
# Initialize model
from equations import NS3D as Eqs
eq_params = ([0.0025,
              params.nu])
eq_params = [np.float32(p) for p in eq_params]
PINN = PhysicsInformedNN(layers,
                         dest=params.paths.dest,
                         norm_in=inorm,
                         norm_out=onorm,
                         optimizer=keras.optimizers.Adam(params.lr),
                         eq_params=eq_params)

# Validation method
PINN.validation = dns_validation(PINN, params)

# Train
PINN.train(X_data, Y_data,
           Eqs,
           epochs=params.epochs,
           batch_size=params.mbsize,
           alpha=0.1,
           lambda_data=lambda_data,
           lambda_phys=lambda_phys,
           print_freq=10,
           valid_freq=10,
           save_freq=10,
           data_mask=(True,True,True,False))
