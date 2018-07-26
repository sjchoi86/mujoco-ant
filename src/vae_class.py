import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim # I lkie slim 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.utils import shuffle
from util import gpu_sess,plot_imgs

class vae_class(object):
    def __init__(self,_name='VAE',_xDim=784,_zDim=10,_hDims=[64,64],_cDim=0,
                 _actv=tf.nn.relu,_outActv=tf.nn.sigmoid,_qActv=tf.nn.tanh,_bn=slim.batch_norm,
                 _entRegCoef=0,_klMinTh=0.0,
                 _optimizer=tf.train.AdamOptimizer,
                 _optm_param={'lr':0.001,'beta1':0.9,'beta2':0.999,'epsilon':1e-9},
                 _VERBOSE=True):
        self.name  = _name # Name
        self.xDim  = _xDim # Dimension of input
        self.zDim  = _zDim # Dimension of latent vector
        self.hDims = _hDims # Dimention of hidden layer(s)
        self.cDim  = _cDim # Dimention of conditional vector 
        self.actv  = _actv # Activation function 
        self.outActv = _outActv
        self.qActv = _qActv
        self.bn    = _bn # Batch norm (slim.batch_norm / None)
        self.entRegCoef = _entRegCoef # Entropy regularizer
        self.klMinTh = _klMinTh # KL-divergence minimum threshold 
        self.optimizer = _optimizer # Optimizer
        self.optm_param = _optm_param # Optimizer parameters
        self.VERBOSE = _VERBOSE
        if self.VERBOSE:
            print ("[%s] xdim:[%d] zdim:[%d] hdim:%s cdim:[%d]"\
                % (self.name,self.xDim,self.zDim,self.hDims,self.cDim))
        # Make model 
        self._build_model()
        self._build_graph()
        self._check_params()

    def _build_model(self):
        # Placeholders
        self.x  = tf.placeholder(tf.float32, shape=[None,self.xDim], name="x") # This will be inputs
        self.z  = tf.placeholder(tf.float32, shape=[None,self.zDim], name="z") # Latent vectors
        self.c  = tf.placeholder(tf.float32, shape=[None,self.cDim], name="c") # Conditioning vectors
        self.q  = tf.placeholder(tf.float32, shape=[None], name="q") # Weighting vectors 
        self.lr = tf.placeholder(tf.float32,name='lr') # Learning rate
        self.kp = tf.placeholder(tf.float32,name='kp') # Keep prob.
        self.klWeight = tf.placeholder(tf.float32) # KL weight heuristics 
        self.isTraining = tf.placeholder(dtype=tf.bool,shape=[]) # Training flag
        # Build graph
        self.bnInit = {'beta':tf.constant_initializer(0.),'gamma':tf.constant_initializer(1.)}
        self.bnParams = {'is_training':self.isTraining,'decay':0.9,'epsilon':1e-5,
                           'param_initializers':self.bnInit,'updates_collections':None}
        with tf.variable_scope(self.name,reuse=False) as scope:
            with slim.arg_scope([slim.fully_connected],activation_fn=self.actv,
                               weights_initializer=tf.random_normal_initializer(stddev=0.1),
                               biases_initializer=tf.constant_initializer(value=0.0),
                               normalizer_fn=self.bn,normalizer_params=self.bnParams,
                               weights_regularizer=None):
                _net = self.x
                self.N = tf.shape(self.x)[0] # Number of current inputs 
                # Encoder 
                for hIdx in range(len(self.hDims)): # Loop over hidden layers
                    _hDim = self.hDims[hIdx]
                    _net = slim.fully_connected(_net,_hDim,scope='enc_lin'+str(hIdx))
                    _net = slim.dropout(_net,keep_prob=self.kp,is_training=self.isTraining
                                        ,scope='enc_dr'+str(hIdx))
                # Latent vector z (NO ACTIVATION!)
                self.zMuEncoded = slim.fully_connected(_net,self.zDim,scope='zMuEncoded',activation_fn=None)
                self.zLogVarEncoded = slim.fully_connected(_net,self.zDim,scope='zLogVarEncoded',activation_fn=None)
                # Define z sampler (reparametrization trick)
                self.eps = tf.random_normal(shape=(self.N,self.zDim),mean=0.,stddev=1.,dtype=tf.float32)
                self.zSample = self.zMuEncoded+tf.sqrt(tf.exp(self.zLogVarEncoded))*self.eps
                # Concatenate the condition vector to the sampled latent vector
                if self.cDim != 0:
                    self.zEncoded = tf.concat([self.zSample,self.c],axis=1)
                else:
                    self.zEncoded = self.zSample
                # Decoder 
                _net = self.zEncoded
                for hIdx in range(len(self.hDims)): # Loop over hidden layers
                    _hDim = self.hDims[len(self.hDims)-hIdx-1]
                    _net = slim.fully_connected(_net,_hDim,scope='dec_lin'+str(hIdx))
                    _net = slim.dropout(_net,keep_prob=self.kp,is_training=self.isTraining
                                        ,scope='dec_dr'+str(hIdx))
                # Reconstruct output 
                self.xRecon = slim.fully_connected(_net,self.xDim,scope='xRecon',activation_fn=self.outActv)
        
        # self.zGiven => self.xGivenZ
        with tf.variable_scope(self.name,reuse=True) as scope:
            with slim.arg_scope([slim.fully_connected],activation_fn=self.actv,
                               weights_initializer=tf.random_normal_initializer(stddev=0.1),
                               biases_initializer=tf.constant_initializer(value=0.0),
                               normalizer_fn=self.bn,normalizer_params=self.bnParams,
                               weights_regularizer=None):
                # Start from given z, instead of sampled z
                if self.cDim != 0:
                    self.zGiven = tf.concat([self.z,self.c],axis=1)
                else:
                    self.zGiven = self.z
                # Decoder 
                _net = self.zGiven
                for hIdx in range(len(self.hDims)): # Loop over hidden layers
                    _hDim = self.hDims[len(self.hDims)-hIdx-1]
                    _net = slim.fully_connected(_net,_hDim,scope='dec_lin'+str(hIdx))
                # Reconstruct output 
                self.xGivenZ = slim.fully_connected(_net,self.xDim,scope='xRecon',activation_fn=self.outActv)
        
        


    def _build_graph(self):
        # Original VAE losses
        # Recon loss
        # qVal = tf.nn.sigmoid(self.q) 
        # qVal = tf.nn.softplus(self.q)
        # qVal = tf.nn.tanh(self.q)
        
        if self.qActv is None:
            qVal = (self.q)
        else:
            qVal = self.qActv(self.q)

        self._reconLoss = 0.5*tf.norm(self.xRecon-self.x,ord=1,axis=1)
        self._reconLossWeighted = qVal*self._reconLoss
        self._reconLossWeighted = tf.clip_by_value(t=self._reconLossWeighted,\
            clip_value_min=-1,clip_value_max=1e6) # <== Clip loss
        self.reconLossWeighted = tf.reduce_mean(self._reconLossWeighted)
        # KL loss
        self._klLoss = 0.5*tf.reduce_sum(tf.exp(self.zLogVarEncoded)+self.zMuEncoded**2-1.-self.zLogVarEncoded,1)
        self._klLoss = self.klWeight*self._klLoss
        self._klLoss = tf.clip_by_value(self._klLoss,self.klMinTh,np.inf)
        self._klLossWeighted = qVal*self._klLoss
        self._klLossWeighted = tf.clip_by_value(t=self._klLossWeighted,\
            clip_value_min=-1,clip_value_max=1e6) # <== Clip loss
        self.klLossWeighted = tf.reduce_mean(self._klLossWeighted)

        # Weight decay
        self.l2RegCoef = 1e-5
        _g_vars = tf.trainable_variables()
        self.c_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        self.l2Reg = self.l2RegCoef*tf.reduce_sum(tf.stack([tf.nn.l2_loss(v) for v in self.c_vars])) # [1]
        
        # Entropy regularizer (Zsampler => self.xSample)
        with tf.variable_scope(self.name,reuse=True) as scope:
            with slim.arg_scope([slim.fully_connected],activation_fn=self.actv,
                               weights_initializer=tf.random_normal_initializer(stddev=0.1),
                               biases_initializer=tf.constant_initializer(value=0.0),
                               normalizer_fn=self.bn,normalizer_params=self.bnParams,
                               weights_regularizer=None):
                # Start from given z, instead of sampled z
                _zSample = tf.random_normal(shape=(self.N,self.zDim),mean=0.,stddev=1.,dtype=tf.float32)
                if self.cDim != 0: _zSample = tf.concat([_zSample,self.c],axis=1)
                # Decoder 
                _net = _zSample
                for hIdx in range(len(self.hDims)): # Loop over hidden layers
                    _hDim = self.hDims[len(self.hDims)-hIdx-1]
                    _net = slim.fully_connected(_net,_hDim,scope='dec_lin'+str(hIdx))
                # Reconstruct output 
                self.xSample = slim.fully_connected(_net,self.xDim,scope='xRecon',activation_fn=self.outActv)
        self._entReg = self.entRegCoef*tf.norm(self.xSample,ord=1,axis=1) # Shape?
        self.entReg = tf.reduce_mean(self._entReg)

        
        # Total loss
        self.totalLoss = self.reconLossWeighted + self.klLossWeighted + self.l2Reg + self.entReg
        # Solver
        if self.optimizer == tf.train.AdamOptimizer:
            self._optm = tf.train.AdamOptimizer(
                learning_rate=self.lr,beta1=self.optm_param['beta1'],
                beta2=self.optm_param['beta2'],epsilon=self.optm_param['epsilon'])
        elif self.optimizer == tf.train.GradientDescentOptimizer:
            self._optm = tf.train.GradientDescentOptimizer(
                learning_rate=self.lr)
        tvars = tf.trainable_variables()
        grads_and_vars = self._optm.compute_gradients(self.totalLoss,tvars)
        clipped = [(tf.clip_by_value(grad, -1.0, 1.0), tvar) # gradient clipping
                    for grad, tvar in grads_and_vars]
        self.optm = self._optm.apply_gradients(clipped,name="minimize_cost")
            
    # Check parameters
    def _check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        if self.VERBOSE:
            print ("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print (" [%02d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))

    # Train
    def train(self,_sess,_X,_Y,_C,_Q,_maxIter,_batchSize,_PRINT_EVERY=100,_PLOT_EVERY=100,
        _imgSz=(28,28),_figsize=(15,2),_nR=1,_nC=10,
        _LR_SCHEDULE=True,_KL_SCHEDULE=False,
        _INIT_VAR=True):
        # X: inputs [N x D]
        # C: condition vectors [N x 1]
        # Q: weighting vectors [N x 1]
        self.sess = _sess
        # Initialize variables
        if _INIT_VAR:
            self.sess.run(tf.global_variables_initializer())
        # 
        _maxIter,_batchSize,_PRINT_EVERY,_PLOT_EVERY =\
            (int)(_maxIter),(int)(_batchSize),(int)(_PRINT_EVERY),(int)(_PLOT_EVERY)
        # Train
        _Xtrain = _X
        nX = _X.shape[0]
        for _iter in range(_maxIter):
            randIdx = np.random.permutation(nX)[:_batchSize] # Random indices every iteration
            xBatch = _X[randIdx,:] # X batch
            if _LR_SCHEDULE:
                if _iter < _maxIter*0.5: lrVal = self.optm_param['lr']
                elif _iter < _maxIter*0.75: lrVal = self.optm_param['lr']*0.5
                else: lrVal = self.optm_param['lr']*0.5*0.5
            else:
                lrVal = self.optm_param['lr']*1.0

            # KLD schedule
            if _KL_SCHEDULE:
                klWeight = 0.5+0.5*((float)(_iter+1)/_maxIter)
            else:
                klWeight = 1.0
            # Q batch
            if _Q is None:
                qBatch = np.ones(shape=(_batchSize))
            else:
                qBatch = _Q[randIdx]
            if _C is None: # Original VAE (without conditioning)
                feeds = {self.x:xBatch,self.q:qBatch,self.lr:lrVal,
                    self.klWeight:klWeight,self.isTraining:True,self.kp:0.9}
            else: # Conditional VAE
                cBatch = _C[randIdx,:]
                feeds = {self.x:xBatch,self.c:cBatch,self.q:qBatch,self.lr:lrVal,
                    self.klWeight:klWeight,self.isTraining:True,self.kp:0.9}
            # Train
            opers = [self.optm,self.totalLoss,self.reconLossWeighted,self.klLossWeighted,
                    self.l2Reg,self.entReg]
            _,totalLossVal,reconLossWeightedVal,klLossWeightedVal,l2RegVal,entRegVal =\
                 self.sess.run(opers,feed_dict=feeds)
            # Print 
            if _PRINT_EVERY != 0:
                if ((_iter%_PRINT_EVERY)==0) | (_iter==(_maxIter-1)):
                    print ("[%04d/%d][%.1f%%] Loss: %.2f(recon:%.2f+kl:%.2f+l2:%.2f+ent:%.2f)"%
                        (_iter,_maxIter,100.*_iter/_maxIter,totalLossVal,reconLossWeightedVal,
                        klLossWeightedVal,l2RegVal,entRegVal))
            # Plot
            if _PLOT_EVERY != 0:
                if ((_iter%_PLOT_EVERY)==0) | (_iter==(_maxIter-1)):
                    self.test(self.sess,_nR=_nR,_nC=_nC,_C=_C,_X=_X,_Y=_Y,
                        _imgSz=(28,28),_figsize=(15,2))

    # Sample one 
    def sample(self,_sess=None,_c=None,_seed=None):
        if _sess is not None:
            self.sess = _sess
        if _seed is not None:
            np.random.seed(seed=_seed)
        zRandn = 1.*np.random.randn(1,self.zDim)
        if _c is None:
            feeds = {self.z:zRandn,self.isTraining:False,self.kp:1.0}
        else:
            feeds = {self.z:zRandn,self.c:_c,self.isTraining:False,self.kp:1.0}
        sampledX = self.sess.run(self.xGivenZ,feed_dict=feeds)
        return sampledX

    # Test            
    def test(self,_sess,_nR,_nC,_C=None,_X=None,_Y=None,
        _imgSz=(28,28),_figsize=(15,2),_seed=None):
        self.sess = _sess
        # Plot sampled images 
        if _seed is not None:
            np.random.seed(seed=_seed)
        zRandn = 1.*np.random.randn(_nR*_nC,self.zDim)
        if _C is None: # Original VAE
            # Plot sampled images
            feeds = {self.z:zRandn,self.isTraining:False,self.kp:1.0}
            sampledImages = self.sess.run(self.xGivenZ,feed_dict=feeds)
            plot_imgs(_imgs=sampledImages,_imgSz=_imgSz,
                _nR=_nR,_nC=_nC,_figsize=_figsize,_title='Sampled Images')
        else: # Conditional VAE (assume contional vectors are one-hot coded.)
            cMtx = np.eye(_nC,self.cDim)
            feeds = {self.z:zRandn,self.c:cMtx,self.kp:1.0,self.isTraining:False,self.kp:1.0}
            sampledImages = self.sess.run(self.xGivenZ,feed_dict=feeds)
            _titles = []
            for _i in range(_nC): _titles.append('c:'+str(_i))
            plot_imgs(_imgs=sampledImages,_imgSz=_imgSz,
                _nR=_nR,_nC=_nC,_figsize=_figsize,_title='Sampled Images',_titles=_titles)
        # Plot encoded z
        if _X is not None: 
            _batchSize = 1000
            randIdx = np.random.permutation(_X.shape[0])[:_batchSize]
            xBatch = _X[randIdx,:]
            feeds = {self.x:xBatch,self.kp:1.0,self.isTraining:False}
            zMuVal,zSampleVal = self.sess.run([self.zMuEncoded,self.zSample],feed_dict=feeds)
            fig = plt.figure(figsize=(12,4));fig.suptitle('Latent Space', size=15)
            gs = gridspec.GridSpec(1,2); gs.update(wspace=0.05, hspace=0.05)
            ax = plt.subplot(gs[0]); plt.axis('equal')
            if _Y is None: # Just one color 
                cVal = np.ones(shape=(_batchSize))  
            else: # Encoded vectors with colors from labels 
                yBatch = _Y[randIdx,:]
                cVal = np.argmax(yBatch,1) # Color based on labels (mostly for CVAE)
            plt.scatter(zMuVal[:,0], zMuVal[:, 1],c=cVal,cmap='jet_r')
            plt.colorbar(); plt.grid(True); plt.title('zMuVal')
            ax = plt.subplot(gs[1]); plt.axis('equal')
            plt.scatter(zSampleVal[:,0], zSampleVal[:, 1],c=cVal,cmap='jet_r')
            plt.colorbar(); plt.grid(True); plt.title('zSampleVal')
            plt.show()

    # Save 
    def save(self,_sess,_savename=None,_VERBOSE=True):
        """ Save name """
        directory = 'nets'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if _savename==None:
            _savename='nets/net_%s.npz'%(self.name)
        """ Get global variables """
        self.g_wnames,self.g_wvals,self.g_wshapes = [],[],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = _sess.run(curr_wvar)
            curr_wval_sqz  = curr_wval.squeeze()
            self.g_wnames.append(curr_wname)
            self.g_wvals.append(curr_wval_sqz)
            self.g_wshapes.append(curr_wval.shape)
        """ Save """
        np.savez(_savename,g_wnames=self.g_wnames,g_wvals=self.g_wvals,g_wshapes=self.g_wshapes)
        if _VERBOSE:
            print ("[%s] Saved. Size is [%.4f]MB" % 
                   (_savename,os.path.getsize(_savename)/1000./1000.))

    # Restore
    def restore(self,_sess,_loadname=None,_VERBOSE=True):
        if _loadname==None:
            _loadname='nets/net_%s.npz'%(self.name)
        l = np.load(_loadname)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        for widx,wname in enumerate(g_wnames):
            curr_wvar  = [v for v in tf.global_variables() if v.name==wname][0]
            _sess.run(tf.assign(curr_wvar,g_wvals[widx].reshape(g_wshapes[widx])))
        if _VERBOSE:
            print ("Weight restored from [%s] Size is [%.4f]MB" % 
                   (_loadname,os.path.getsize(_loadname)/1000./1000.))
    