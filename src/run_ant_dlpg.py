import tensorflow as tf
from antTrainEnv_class import antTrainEnv_dlpg_class
from util import gpu_sess
print ("TF version is [%s]."%(tf.__version__))

# Tuning params
headingCoef = 1e-4 # Heading penalty 
tMax = 3
nAnchor = 20
nIter4update = 1e2
maxEpoch  = 1000
batchSize = 100
nPrevConsider = 10
nPrevBestQ2Add = 50
name = 'ant_dlpg_headingCoef%.0e_tMax%d_nAnchor%d_nIter4update%d_batchSize%d'%\
    (headingCoef,tMax,nAnchor,nIter4update,batchSize)
print ("Name:[%s]"%(name))

# Instantiate class
tf.reset_default_graph() # Reset Graph
AntEnv = antTrainEnv_dlpg_class(_name=name,_headingCoef=headingCoef,
                                _tMax=tMax,_nAnchor=nAnchor,_maxRepeat=3,
                                _hypGainPrior=1/2,_hypLenPrior=1/4,
                                _hypGainPost=1/2,_hypLenPost=1/2,
                                _levBtw=0.9,_pGain=0.01,
                                _zDim=16,_hDims=[128,128],_vaeActv=tf.nn.tanh,_vaeOutActv=None,_vaeQactv=tf.nn.tanh,
                                _PLOT_GRP=True,_SAVE_TXT=True)

# Train
SAVE_VID = True
MAKE_GIF = False # Probably unnecessary 
PLOT_GRP = True 
PLOT_EVERY = 10
SAVE_NET_EVERY = 10
sess = gpu_sess()
print ("Start training...")
AntEnv.train_dlpg(_sess=sess,_seed=0,_maxEpoch=maxEpoch,_batchSize=batchSize,_nIter4update=nIter4update,
                  _nPrevConsider=nPrevConsider,_nPrevBestQ2Add=nPrevBestQ2Add,
                  _SAVE_VID=SAVE_VID,_MAKE_GIF=MAKE_GIF,_PLOT_GRP=PLOT_GRP,_PLOT_EVERY=PLOT_EVERY,
                  _DO_RENDER=(SAVE_VID|MAKE_GIF),_SAVE_NET_EVERY=SAVE_NET_EVERY)