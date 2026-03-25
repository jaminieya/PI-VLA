import sys
import os
sys.path.append('.')

# Use Agg for headless; use default (X11) when MPLBACKEND not set
# Run with MPLBACKEND=Agg or --headless for headless servers
if os.environ.get('MPLBACKEND') == 'Agg' or os.environ.get('MATPLOTLIB_HEADLESS'):
    import matplotlib
    matplotlib.use('Agg')

from models.metric_arm import model_train_metric as md#_newout_sqrtlog _newout_log2
from os import path
import numpy as np
modelPath = './Experiments/UR5'         
dataPath = './datasets/arm/UR5'

model    = md.Model(modelPath, dataPath, 6, [-1.2, 0.4-0.5*np.pi, 1.4, 0.2-0.5*np.pi,-0.5,0.9],device='cuda:0')

model.train()