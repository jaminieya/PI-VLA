import sys
import os
sys.path.append('.')

# Use Agg for headless; use default (X11) when MPLBACKEND not set
if os.environ.get('MPLBACKEND') == 'Agg' or os.environ.get('MATPLOTLIB_HEADLESS'):
    import matplotlib
    matplotlib.use('Agg')

from models.metric import model_train_metric as md
from os import path
from glob import glob

modelPath = './Experiments/Gib'
dataPath = './datasets/gibson/Auburn'
#dataPath = './datasets/gibson/Spotswood'  # use after preprocessing that scene



#model    = md.Model(modelPath, dataPath, 3, [0, 0.3,-0.03],device='cuda:0')
model    = md.Model(modelPath, dataPath, 3, [-0.15, 0.1,0.1],device='cuda:0')

model.train()


