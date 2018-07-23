import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim 
import matplotlib.pyplot as plt

import skvideo.io
from datetime import datetime

from custom_ant import AntEnvCustom # Custom ant
from lgrp_class import lgrp_class # Gaussian random path
from vae_class import vae_class # VAE
from ppo import NNValueFunction,Policy,run_episode,run_policy,add_value,discount,\
    add_disc_sum_rew,add_gae,build_train_set,log_batch_stats,run_episode_vid

from util import PID_class,quaternion_to_euler_angle,multi_dim_interp,Scaler,\
    display_frames_as_gif,print_n_txt,Logger

class antTrainEnv_dlpg_class(object):
    def __init__(self,_name='Ant',_tMax=3,_nAnchor=20,_maxRepeat=3,
            _hypGainPrior=1/3,_hypLenPrior=1/4,
            _hypGainPost=1/3,_hypLenPost=1/4,
            _levBtw=0.8,_pGain=0.01,
            _zDim=16,_hDims=[64,64],_vaeActv=tf.nn.elu,_vaeOutActv=tf.nn.sigmoid,_vaeQactv=None,
            _PLOT_GRP=True,_SAVE_TXT=True,_VERBOSE=True):
        # Some parameters
        self.name = _name
        self.tMin = 0
        self.tMax = _tMax
        self.nAnchor = _nAnchor
        self.maxRepeat = _maxRepeat
        self.SAVE_TXT = _SAVE_TXT
        self.VERBOSE = _VERBOSE

        # Noramlize trajecotry
        self.NORMALIZE_SCALE = True 


        if self.SAVE_TXT:
            txtName = 'results/'+self.name+'.txt'
            self.f = open(txtName,'w') # Open txt file
            print_n_txt(_f=self.f,_chars='Text name: '+txtName,
                _DO_PRINT=True,_DO_SAVE=self.SAVE_TXT)
        
        # Initialize Ant gym 
        self.env = AntEnvCustom()
        # GRP sampler (prior)
        nDataPrior = 2
        nTest = (int)((self.tMax-self.tMin)/self.env.dt)
        tData = np.linspace(start=self.tMin,stop=self.tMax,num=nDataPrior).reshape((-1,1))
        xData = np.random.rand(nDataPrior,self.env.actDim) # Random positions 
        xData[0,:] = (xData[0,:]+xData[-1,:])/2.0
        xData[-1,:] = xData[0,:]
        lData = np.ones(shape=(nDataPrior,1))

        lData = np.ones(shape=(nDataPrior,1))
        tTest = np.linspace(start=self.tMin,stop=self.tMax,num=nTest).reshape((-1,1))
        lTest = np.ones(shape=(nTest,1))

        # hyp = {'gain':1/3,'len':1/4,'noise':1e-8} # <= This worked fine
        hypPrior = {'gain':_hypGainPrior,'len':_hypLenPrior,'noise':1e-8}
        self.GRPprior = lgrp_class(_name='GPR Prior',_tData=tData,
                                   _xData=xData,_lData=lData,_tTest=tTest,
                                   _lTest=lTest,_hyp=hypPrior)
            
        # GRP posterior
        tData = np.linspace(start=self.tMin,stop=self.tMax,num=self.nAnchor).reshape((-1,1))
        xData = np.random.rand(self.nAnchor,self.env.actDim) # Random positions 
        lData = np.ones(shape=(self.nAnchor,1))
        lData[1:self.nAnchor-1] = _levBtw
        hypPost = {'gain':_hypGainPost,'len':_hypLenPost,'noise':1e-8}
        self.GRPposterior = lgrp_class(_name='GPR Posterior',_tData=tData,
                                   _xData=xData,_lData=lData,_tTest=tTest,
                                   _lTest=lTest,_hyp=hypPost)
        if _PLOT_GRP:
            self.GRPprior.plot_all(_nPath=10,_figsize=(12,4))
            self.GRPposterior.plot_all(_nPath=10,_figsize=(12,4))

        # PID controller (Kp=0.01,Ki=0.00001,Kd=0.002,windup=5000)
        self.PID = PID_class(Kp=_pGain,Ki=0.00001,Kd=0.002,windup=5000,
                        sample_time=self.env.dt,dim=self.env.actDim)
        # VAE (this will be our policy function)

        # optm = tf.train.AdamOptimizer
        # optmParam = {'lr':0.001,'beta1':0.9,'beta2':0.9,'epsilon':1e-8}

        optm = tf.train.GradientDescentOptimizer
        optmParam = {'lr':0.001}
        self.VAE = vae_class(_name=self.name,_xDim=self.nAnchor*self.env.actDim,
                             _zDim=_zDim,_hDims=_hDims,_cDim=0,
                             _actv=_vaeActv,_outActv=_vaeOutActv,_qActv=_vaeQactv,
                             _bn=None,
                             _optimizer=optm,
                             _optm_param=optmParam,
                             _VERBOSE=False)
        # Reward Scaler
        self.qScaler = Scaler(1)
        # Check parameters
        self.check_params()


    # Check parameters
    def check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/'%(self.name) in var.name]
        if self.VERBOSE:
            print ("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name  = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print (" [%02d] Name:[%s] Shape:[%s]" % (i,w_name,w_shape))
        
    # Save 
    def save_net(self,_sess,_savename=None):
        """ Save name """
        if _savename==None:
            _savename='nets/net_%s.npz'%(self.name)
        """ Get global variables """
        self.g_wnames,self.g_wvals,self.g_wshapes = [],[],[]
        for i in range(len(self.g_vars)):
            curr_wname = self.g_vars[i].name
            curr_wvar  = [v for v in tf.global_variables() if v.name==curr_wname][0]
            curr_wval  = _sess.run(curr_wvar)
            
            curr_wval_sqz = curr_wval
            # curr_wval_sqz  = curr_wval.squeeze() # ???
            curr_wval_sqz = np.asanyarray(curr_wval_sqz,order=(1,-1))
            
            self.g_wnames.append(curr_wname)
            self.g_wvals.append(curr_wval_sqz)
            self.g_wshapes.append(curr_wval.shape)
        """ Save """
        np.savez(_savename,g_wnames=self.g_wnames,g_wvals=self.g_wvals,g_wshapes=self.g_wshapes)
        if self.VERBOSE:
            print ("[%s] Saved. Size is [%.4f]MB" % 
                   (_savename,os.path.getsize(_savename)/1000./1000.))
        
    # Restore
    def restore_net(self,_sess,_loadname=None):
        if _loadname==None:
            _loadname='nets/net_%s.npz'%(self.name)
        l = np.load(_loadname)
        g_wnames = l['g_wnames']
        g_wvals  = l['g_wvals']
        g_wshapes = l['g_wshapes']
        for widx,wname in enumerate(g_wnames):
            curr_wvar  = [v for v in tf.global_variables() if v.name==wname][0]
            _sess.run(tf.assign(curr_wvar,g_wvals[widx].reshape(g_wshapes[widx])))
        if self.VERBOSE:
            print ("Weight restored from [%s] Size is [%.4f]MB" % 
                   (_loadname,os.path.getsize(_loadname)/1000./1000.))
    
    # Basic functionality test (sample and rollout)
    def basic_test(self):
        # Sample Trajectory for GP prior
        avgRwdPrior,retPrior = self.unit_rollout_from_grp_prior(self.maxRepeat)
        print ("[Prior] avgRwd:[%.3f] xDisp:[%.3f] hDisp:[%.3f]"%
            (avgRwdPrior,retPrior['xDisp'],retPrior['hDisp']))
        # Set anchor points from trajectory
        self.set_anchor_grp_posterior_from_traj(retPrior['sampledTraj'],_levBtw=0.8)
        avgRwdMean,retMean = self.unit_rollout_from_grp_mean(self.maxRepeat)
        print ("[Mean]  avgRwd:[%.3f] xDisp:[%.3f] hDisp:[%.3f]"%
            (avgRwdMean,retMean['xDisp'],retMean['hDisp']))
        avgRwdPost,retPost = self.unit_rollout_from_grp_posterior(self.maxRepeat)
        print ("[Post]  avgRwd:[%.3f] xDisp:[%.3f] hDisp:[%.3f]"%
            (avgRwdPost,retPost['xDisp'],retPost['hDisp']))

        # Plot
        cmap = plt.get_cmap('inferno')
        colors = [cmap(i) for i in np.linspace(0,1,self.env.actDim+1)]
        plt.figure(figsize=(12,5))
        muTest = self.GRPposterior.muTest
        sigmaTest = np.sqrt(self.GRPposterior.varTest)
        for dIdx in range(self.env.actDim):
            hVar = plt.fill_between(self.GRPposterior.tTest.squeeze(),
                            (muTest[:,dIdx:dIdx+1]-2*sigmaTest).squeeze(),
                            (muTest[:,dIdx:dIdx+1]+2*sigmaTest).squeeze(),
                            facecolor=colors[dIdx], interpolate=True, alpha=0.2)
            hPrior,=plt.plot(self.GRPposterior.tTest,retPrior['sampledTraj'][:,dIdx],
                    '-',color=colors[dIdx],lw=1) 
            hMean,=plt.plot(self.GRPposterior.tTest,retMean['sampledTraj'][:,dIdx],
                    ':',color=colors[dIdx],lw=2)
            hPost,=plt.plot(self.GRPposterior.tTest,retPost['sampledTraj'][:,dIdx],
                    '--',color=colors[dIdx],lw=2)
        plt.legend([hPrior,hMean,hPost,hVar],
            ['Sampled from Prior (original)','GRP Mean','Sampled from Posterior','GRP Var.'],
            fontsize=13)
        plt.show()

    # Scale-up trajectory
    def scale_up_traj(self,rawTraj):
        return self.env.minPosDeg+(self.env.maxPosDeg-self.env.minPosDeg)*rawTraj
    
    # Rollout trajectory
    def rollout(self,_pursuitTraj,_maxRepeat=1,_VERBOSE=True,_DO_RENDER=False):
        
        lenTraj = _pursuitTraj.shape[0]
        cntRepeat = 0
        # Reset
        obs = self.env.reset_model()
        self.env.seed(0)
        self.PID.clear()
        for _ in range(10): 
            sec = self.env.sim.data.time
            refPosDeg = _pursuitTraj[0,:]
            cPosDeg = np.asarray(obs[5:13])*180.0/np.pi # Current pos
            degDiff = cPosDeg-refPosDeg
            self.PID.update(degDiff,sec)
            action = self.PID.output
            actionRsh = action[[6,7,0,1,2,3,4,5]] # rearrange
            obs,_,_,_ = self.env.step(actionRsh)
        self.PID.clear() # Reset PID after burn-in
        self.env.sim.data.time = 0
        tick,frames,titleStrs = 0,[],[]
        tickPursuit = 0
        timeList = np.zeros(shape=(lenTraj*_maxRepeat)) # 
        cPosDegList = np.zeros(shape=(lenTraj*_maxRepeat,self.env.actDim)) # Current joint trajectory
        rPosDegList = np.zeros(shape=(lenTraj*_maxRepeat,self.env.actDim)) # Reference joint trajectory
        rSum = 0
        rContactSum = 0
        rCtrlSum = 0
        rFwdSum = 0
        rHeadingSum = 0
        rSrvSum = 0
        xInit = self.env.get_body_com("torso")[0]
        hInit = self.env.get_heading()
        rewards = []
        while cntRepeat < _maxRepeat:
            # Some info
            sec = self.env.sim.data.time # Current time 
            x = self.env.get_body_com("torso")[0]
            q = self.env.data.get_body_xquat('torso')
            rX,rY,rZ = quaternion_to_euler_angle(q[0],q[1],q[2],q[3])

            # Render 
            if _DO_RENDER:
                frame = self.env.render(mode='rgb_array',width=200,height=200)
            else:
                frame = ''
            frames.append(frame) # Append to frames for futher animating 
            titleStrs.append('%.2fsec x:[%.2f] heading:[%.1f]'%(sec,x,rZ))

            # PID controller 
            timeList[tick] = sec
            refPosDeg = _pursuitTraj[min(tickPursuit,lenTraj-1),:]
            rPosDegList[tick,:] = refPosDeg
            cPosDeg = np.asarray(obs[5:13])*180.0/np.pi # Current pos
            cPosDegList[tick,:] = cPosDeg
            degDiff = cPosDeg-refPosDeg
            self.PID.update(degDiff,sec)

            # Do action 
            action = self.PID.output
            actionRsh = action[[6,7,0,1,2,3,4,5]] # rearrange

            obs,reward,done,rwdDetal = self.env.step(actionRsh.astype(np.float16))
            rSum += reward
            
            rContactSum += rwdDetal['reward_contact']
            rCtrlSum += rwdDetal['reward_ctrl']
            rFwdSum += rwdDetal['reward_forward']
            rHeadingSum += rwdDetal['reward_heading']
            rSrvSum += rwdDetal['reward_survive']

            rewards.append(reward)

            # Print out (for debugging)
            DO_PRINT = False
            if DO_PRINT:
                print ('sec: [%.2f] done: %s'%(sec,done))
                print (' cPosDeg:   %s'%(np.array2string(cPosDeg,precision=2,
                    formatter={'float_kind':lambda x: "%.2f" % x},
                    separator=', ',suppress_small=False,sign=' ')))
                print (' refPosDeg: %s'%(np.array2string(refPosDeg,precision=2,
                    formatter={'float_kind':lambda x: "%.2f" % x},
                    separator=', ',suppress_small=False,sign=' ')))           
                print (' degDiff:   %s'%(np.array2string(degDiff,precision=2,
                    formatter={'float_kind':lambda x: "%.2f" % x},
                    separator=', ',suppress_small=False,sign=' ')))
                print (' action:    %s'%(np.array2string(action,precision=2,
                    formatter={'float_kind':lambda x: "%.2f" % x},
                    separator=', ',suppress_small=False,sign=' ')))
                print (' reward:    %.3f (fwd:%.3f+ctrl:%.3f+contact:%.3f+survive:%.3f)'
                    %(reward,rwdDetal['reward_forward'],rwdDetal['reward_ctrl'],
                      rwdDetal['reward_contact'],rwdDetal['reward_survive']))
            # Increase tick 
            tick += 1
            tickPursuit += 1
            # Loop handler
            if tickPursuit >= lenTraj:
                cntRepeat += 1
                tickPursuit = 0
            # Done handler
            if done:
                break
        xFinal = self.env.get_body_com("torso")[0]
        hFinal = self.env.get_heading()
        xDisp = xFinal - xInit
        hDisp = hFinal - hInit

        rAvg = rSum/tick
        rContactAvg = rContactSum/tick
        rCtrlAvg = rCtrlSum/tick
        rFwdAvg = rFwdSum/tick
        rHeadingAvg = rHeadingSum/tick
        rSrvAvg = rSrvSum/tick

        if _VERBOSE:
            print ("Repeat:[%d] done. Avg reward is [%.3f]. xDisp is [%.3f]. hDisp is [%.3f]."
                   %(self.maxRepeat,rSum/tick,xDisp,hDisp))
        ret = {'rewards':rewards,'frames':frames,'titleStrs':titleStrs,
              'xDisp':xDisp,'hDisp':hDisp,
              'rAvg':rAvg,'rContactAvg':rContactAvg,'rCtrlAvg':rCtrlAvg,
              'rFwdAvg':rFwdAvg,'rHeadingAvg':rHeadingAvg,'rSrvAvg':rSrvAvg,
              'rSum':rSum,'rContactSum':rContactSum,'rCtrlSum':rCtrlSum,
              'rFwdSum':rFwdSum,'rHeadingSum':rHeadingSum,'rSrvSum':rSrvSum}
        return ret
    
    # Sample from GRP prior
    def sample_from_grp_prior(self):
        # Reset 
        nDataPrior = 2
        nTest = (int)((self.tMax-self.tMin)/self.env.dt)
        tData = np.linspace(start=self.tMin,stop=self.tMax,num=nDataPrior).reshape((-1,1))
        xData = np.random.rand(nDataPrior,self.env.actDim) # Random positions 
        xData[0,:] = (xData[0,:]+xData[-1,:])/2.0
        xData[-1,:] = xData[0,:]
        lData = np.ones(shape=(nDataPrior,1))
        self.GRPprior.set_data(_tData=tData,_xData=xData,_lData=lData,_EPS_RU=False)

        # Sample Trajectory for GP prior
        sampledTraj = self.GRPprior.sample_paths(_nPath=1)[0]
        return sampledTraj
    
    # Rollout from given trajectory
    def unit_rollout_from_traj(self,_pursuitTraj,_maxRepeat=3,_DO_RENDER=False):
        ret = self.rollout(_pursuitTraj,_maxRepeat,_VERBOSE=False,_DO_RENDER=_DO_RENDER)
        avgRwd = np.asarray(ret['rewards'],dtype=np.float32).mean()
        return avgRwd,ret

    # Get anchor poiints from trajecotry
    def get_anchor_from_traj(self,_sampledTraj):
        tInterp = np.linspace(start=self.tMin,stop=self.tMax,
                      num=self.nAnchor).reshape((-1,1))
        xInterp = multi_dim_interp(tInterp,self.GRPprior.tTest,_sampledTraj)
        return xInterp
    
    # Set anchor points to GRP posterior
    def set_anchor_grp_posterior_from_traj(self,_sampledTraj,_levBtw=0.8):
        tInterp = np.linspace(start=self.tMin,stop=self.tMax,
                      num=self.nAnchor).reshape((-1,1))
        xInterp = multi_dim_interp(tInterp,self.GRPprior.tTest,_sampledTraj)
        lInterp = np.ones_like(tInterp)
        lInterp[1:self.nAnchor-1] = _levBtw
        self.GRPposterior.set_data(_tData=tInterp,_xData=xInterp,_lData=lInterp,
                                   _EPS_RU=True)
        
    # Set anchor points to GRP posterior
    def set_anchor_grp_posterior(self,_anchors,_levBtw=0.8):
        tInterp = np.linspace(start=self.tMin,stop=self.tMax,
                      num=self.nAnchor).reshape((-1,1))
        xInterp = _anchors
        lInterp = np.ones_like(tInterp)
        lInterp[1:self.nAnchor-1] = _levBtw
        self.GRPposterior.set_data(_tData=tInterp,_xData=xInterp,_lData=lInterp,
                                   _EPS_RU=True)
    
    # Rollout from GRP posterior 
    def unit_rollout_from_grp_prior(self,_maxRepeat,_DO_RENDER=False):
        sampledTraj = self.sample_from_grp_prior()
        pursuitTraj = self.scale_up_traj(sampledTraj)
        avgRwd,ret = self.unit_rollout_from_traj(pursuitTraj,_maxRepeat=_maxRepeat,_DO_RENDER=_DO_RENDER)
        ret['sampledTraj'] = sampledTraj
        ret['pursuitTraj'] = pursuitTraj
        return avgRwd,ret
    
    # Rollout from GRP posterior 
    def unit_rollout_from_grp_posterior(self,_maxRepeat,_DO_RENDER=False):
        sampledTraj = self.GRPposterior.sample_paths(_nPath=1)[0]
        pursuitTraj = self.scale_up_traj(sampledTraj)
        avgRwd,ret = self.unit_rollout_from_traj(pursuitTraj,_maxRepeat=_maxRepeat,_DO_RENDER=_DO_RENDER)
        ret['sampledTraj'] = sampledTraj
        ret['pursuitTraj'] = pursuitTraj
        return avgRwd,ret
    
    # Rollout from GRP mean
    def unit_rollout_from_grp_mean(self,_maxRepeat,_DO_RENDER=False):
        pursuitTraj = self.scale_up_traj(self.GRPposterior.muTest)
        avgRwd,ret = self.unit_rollout_from_traj(pursuitTraj,_maxRepeat=_maxRepeat,_DO_RENDER=_DO_RENDER)
        ret['sampledTraj'] = self.GRPposterior.muTest
        ret['pursuitTraj'] = pursuitTraj
        return avgRwd,ret

    # Run
    def train_dlpg(self,_sess,_seed=0,_maxEpoch=500,_batchSize=100,_nIter4update=1e3,
        _nPrevConsider=20,_nPrevBestQ2Add=50,
        _SAVE_VID=True,_MAKE_GIF=False,_PLOT_GRP=False,
        _PLOT_EVERY=5,_DO_RENDER=True,_SAVE_NET_EVERY=10):
        self.sess = _sess

        # Initialize VAE weights
        self.sess.run(tf.global_variables_initializer()) 
        
        # Expirence memory
        xList = np.zeros((_batchSize,self.env.actDim*self.nAnchor))
        qList = np.zeros((_batchSize))
        xLists = ['']*_maxEpoch
        qLists = ['']*_maxEpoch
        
        for _epoch in range(_maxEpoch):
            priorProb = 0.1+0.8*np.exp(-4*(_epoch/500)**2) # Schedule eps-greedish (0.9->0.1)
            levBtw = 0.8+0.15*(1-priorProb) # Schedule leveraged GRP (0.8->0.95)
            xDispList,hDispList = np.zeros((_batchSize)),np.zeros((_batchSize))
            rSumList,rContactSumList,rCtrlSumList,rFwdSumList,rHeadingSumList,rSrvSumList = \
                np.zeros((_batchSize)),np.zeros((_batchSize)),np.zeros((_batchSize)),\
                np.zeros((_batchSize)),np.zeros((_batchSize)),np.zeros((_batchSize))
            for _iter in range(_batchSize):  
                np.random.seed(seed=(_epoch*_batchSize+_iter)) # 

                # -------------------------------------------------------------------------------------------- #
                if (np.random.rand()<priorProb) | (_epoch==0): # Sample from prior
                    _,ret = self.unit_rollout_from_grp_prior(self.maxRepeat)
                else: # Sample from posterior (VAE)
                    sampledX = self.VAE.sample(_sess=self.sess).reshape((self.nAnchor,self.env.actDim))
                    if self.NORMALIZE_SCALE:
                        sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
                    self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=levBtw)
                    _,ret = self.unit_rollout_from_grp_posterior(self.maxRepeat)
                # -------------------------------------------------------------------------------------------- #

                # Get anchor points of previous rollout 
                xInterp = self.get_anchor_from_traj(ret['sampledTraj'])
                xVec = np.reshape(xInterp,newshape=(1,-1))
                # Append rewards 
                xList[_iter,:] = xVec
                qList[_iter] = np.asarray(ret['rewards']).sum() # Sum of rewards!
                xDispList[_iter] = ret['xDisp']
                hDispList[_iter] = ret['hDisp']
                rSumList[_iter] = ret['rSum']
                rContactSumList[_iter] = ret['rContactSum']
                rCtrlSumList[_iter] = ret['rCtrlSum']
                rFwdSumList[_iter] = ret['rFwdSum']
                rHeadingSumList[_iter] = ret['rHeadingSum']
                rSrvSumList[_iter] = ret['rSrvSum'] 
            # Train
            xLists[_epoch] = xList
            qLists[_epoch] = qList
            # Get the best out of previous episodes 
            for _bIdx in range(0,_nPrevConsider):
                if _bIdx == 0: # Add current one for sure 
                    xAccList = xList
                    qAccList = qList
                else:
                    xAccList = np.concatenate((xAccList,xLists[max(0,_epoch-_bIdx)]),axis=0)
                    qAccList = np.concatenate((qAccList,qLists[max(0,_epoch-_bIdx)]))
            # Add high q episodes (_nPrevBestQ2Add)
            nAddPrevBest = _nPrevBestQ2Add
            sortedIdx = np.argsort(-qAccList)
            xTrain = xAccList[sortedIdx[:nAddPrevBest],:]
            qTrain = qAccList[sortedIdx[:nAddPrevBest]]
            # Add current episodes (batchSize)
            xTrain = np.concatenate((xTrain,xList),axis=0)
            qTrain = np.concatenate((qTrain,qList))
            # Add random episodes (nRandomAdd)
            nRandomAdd = _batchSize 
            randIdx = np.random.permutation(xAccList.shape[0])[:nRandomAdd]
            xRand = xAccList[randIdx,:]
            qRand = qAccList[randIdx]
            xTrain = np.concatenate((xTrain,xRand),axis=0)
            qTrain = np.concatenate((qTrain,qRand))
            
            # Train
            self.qScaler.reset() # Reset every update 
            self.qScaler.update(qTrain) # Update Q scaler
            qScale,qOffset = self.qScaler.get() # Scaler 
            scaledQ = qScale*(qTrain-qOffset)
            # print (scaledQ)
            self.VAE.train(_sess=self.sess,_X=xTrain,_Y=None,_C=None,_Q=scaledQ,
                            _maxIter=_nIter4update,_batchSize=128,_PRINT_EVERY=0,_PLOT_EVERY=0,_INIT_VAR=False)
            # Print
            str2print = ("[%d/%d](#total:%d) avgQ:[%.3f] XdispMean:[%.3f] XdispVar:[%.3f] absHdispMean:[%.1f] priorProb:[%.2f]"%
                    (_epoch,_maxEpoch,(_epoch+1)*_batchSize,qList.mean(),
                    xDispList.mean(),xDispList.var(),np.abs(hDispList).mean(),priorProb))
            print_n_txt(_f=self.f,_chars=str2print,_DO_PRINT=True,_DO_SAVE=self.SAVE_TXT)
            
            str2print = (" rSum:[%.3f] = (contact:%.3f+ctrl:%.3f+fwd:%.3f+heading:%.3f+survive:%.3f) [rSumMax:%.3f]"%
                (rSumList.mean(),rContactSumList.mean(),rCtrlSumList.mean(),rFwdSumList.mean(),
                    rHeadingSumList.mean(),rSrvSumList.mean(),rSumList.max()))
            print_n_txt(_f=self.f,_chars=str2print,_DO_PRINT=True,_DO_SAVE=self.SAVE_TXT)


            # SHOW EVERY 
            if ((_epoch%_PLOT_EVERY)==0 ) | (_epoch==(_maxEpoch-1)):
                # Rollout 
                sampledX = self.VAE.sample(_sess=self.sess).reshape((self.nAnchor,self.env.actDim))
                if self.NORMALIZE_SCALE:
                    sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
                self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=levBtw)
                _,ret = self.unit_rollout_from_grp_mean(
                        _maxRepeat=self.maxRepeat,_DO_RENDER=_DO_RENDER)
                str2print = ("    [GRP mean] sumRwd:%.3f=cntct:%.2f+ctrl:%.2f+fwd:%.2f+hd:%.2f+srv:%.2f) xD:[%.3f] hD:[%.1f]"%
                        (ret['rSum'],ret['rContactSum'],ret['rCtrlSum'],ret['rFwdSum'],ret['rHeadingSum'],ret['rSrvSum'],
                         ret['xDisp'],ret['hDisp']))
                print_n_txt(_f=self.f,_chars=str2print,_DO_PRINT=True,_DO_SAVE=self.SAVE_TXT)
                # Make video
                if _SAVE_VID: 
                    outputdata = np.asarray(ret['frames']).astype(np.uint8)
                    vidName = 'vids/ant_dlpg_epoch%03d.mp4'%(_epoch)
                    skvideo.io.vwrite(vidName,outputdata)
                    str2print =  ("     Video [%s] saved."%(vidName))
                    print_n_txt(_f=self.f,_chars=str2print,_DO_PRINT=True,_DO_SAVE=self.SAVE_TXT)
                # Make GIF 
                if _MAKE_GIF:
                    NSKIP = 3 # For memory issues
                    display_frames_as_gif(ret['frames'][::NSKIP],_intv_ms=20,_figsize=(8,8),_fontsize=15,
                                        _titleStrs=ret['titleStrs'][::NSKIP])
                # Plot sampled trajectories 
                if _PLOT_GRP:
                    nrTrajectories2plot = 3
                    for _i in range(nrTrajectories2plot):
                        np.random.seed(seed=_i)
                        sampledX = self.VAE.sample(_sess=self.sess).reshape((self.nAnchor,self.env.actDim))
                        if self.NORMALIZE_SCALE:
                            sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
                        self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=levBtw)
                        self.GRPposterior.plot_all(_nPath=1,_figsize=(8,3))
                        _,ret = self.unit_rollout_from_grp_mean(_maxRepeat=self.maxRepeat,_DO_RENDER=False)
                        str2print =  ("    [GRP-%d] sumRwd:%.3f=cntct:%.2f+ctrl:%.2f+fwd:%.2f+hd:%.2f+srv:%.2f) xD:[%.3f] hD:[%.1f]"%
                            (_i,ret['rSum'],ret['rContactSum'],ret['rCtrlSum'],ret['rFwdSum'],ret['rHeadingSum'],ret['rSrvSum'],
                            ret['xDisp'],ret['hDisp']))
                        print_n_txt(_f=self.f,_chars=str2print,_DO_PRINT=True,_DO_SAVE=self.SAVE_TXT)
            # Save network every 
            if ((_epoch%_SAVE_NET_EVERY)==0 ) | (_epoch==(_maxEpoch-1)):
                saveName = 'nets/net_%s_epoch%04d.npz'%(self.name,_epoch)
                self.save_net(_sess=_sess,_savename=saveName)

    # Make video using current policy
    def make_video(self,_vidName=None,_seed=None,_PRINT_VAESAMPLE=False):
        # Get Anchor points 
        if _seed is not None:
            np.random.seed(seed=_seed)
        sampledX = self.VAE.sample(_sess=self.sess,_seed=_seed).reshape((self.nAnchor,self.env.actDim))
        if self.NORMALIZE_SCALE:
            sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
        # Set GRP
        levBtw = 1.0
        self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=levBtw)
        # Rollout
        _,ret = self.unit_rollout_from_grp_mean(
            _maxRepeat=self.maxRepeat,_DO_RENDER=True)
        # Make video
        outputdata = np.asarray(ret['frames']).astype(np.uint8)
        if _vidName is None:
            now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
            _vidName = 'vids/ant_dlpg_%s.mp4'%(now)
        skvideo.io.vwrite(_vidName,outputdata)
        print(" make_video:[%s] saved."%(_vidName))
        print(" sumRwd:%.3f=cntct:%.2f+ctrl:%.2f+fwd:%.2f+hd:%.2f+srv:%.2f) xD:[%.3f] hD:[%.1f]"%
            (ret['rSum'],ret['rContactSum'],ret['rCtrlSum'],ret['rFwdSum'],ret['rHeadingSum'],
            ret['rSrvSum'],ret['xDisp'],ret['hDisp']))
        
## 
class antTrainEnv_ppo_class(object):
    def __init__(self):
        self.env = AntEnvCustom()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.env.reset() # Reset 
        # render_img = env.render(mode='rgb_array')
        print ("obs_dim:[%d] act_dim:[%d]"%(self.obs_dim,self.act_dim))

        self.obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
        # Logger
        self.env_name = 'Ant'
        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
        self.logger = Logger(logName=self.env_name,now=now,_NOTUSE=True)
        self.aigym_path = os.path.join('/tmp', self.env_name, now)
        # Scaler
        self.scaler = Scaler(self.obs_dim)
        # Value function
        hid1_mult = 10
        self.val_func = NNValueFunction(self.obs_dim, hid1_mult) 
        # Policy Function
        kl_targ = 0.003
        policy_logvar = -1.0
        self.policy = Policy(self.obs_dim,self.act_dim,kl_targ,hid1_mult,policy_logvar) 

    def train(self,_maxEpoch=10000,_batchSize=50,
                _SAVE_VID=True,_MAKE_GIF=False):

        trajectories = run_policy(self.env,self.policy,self.scaler,self.logger,episodes=5)
        add_value(trajectories,self.val_func)  # add estimated values to episodes
        gamma = 0.995 # Discount factor 
        lam = 0.95 # Lambda for GAE
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        print ('observes shape:',trajectories[0]['observes'].shape)
        print ('actions shape:',trajectories[0]['actions'].shape)
        print ('rewards shape:',trajectories[0]['rewards'].shape)
        print ('unscaled_obs shape:',trajectories[0]['unscaled_obs'].shape)
        print ('values shape:',trajectories[0]['values'].shape)
        print ('disc_sum_rew shape:',trajectories[0]['disc_sum_rew'].shape)
        print ('advantages shape:',trajectories[0]['advantages'].shape)

        for _epoch in range(_maxEpoch):
            # 1. Run policy
            trajectories = run_policy(self.env,self.policy,self.scaler,self.logger,episodes=_batchSize)
            # 2. Get (predict) value from the critic network 
            add_value(trajectories,self.val_func)  # add estimated values to episodes
            # 3. Get GAE
            gamma = 0.995 # Discount factor 
            lam = 0.95 # Lambda for GAE
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
            # add various stats to training log:
            # log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
            # Update
            self.policy.update(observes, actions, advantages,self.logger)  # update policy
            self.val_func.fit(observes, disc_sum_rew,self.logger)  # update value function
            # logger.write(display=True)  # write logger results to file and stdout
            
            # Print
            for _tIdx in range(len(trajectories)):
                rs = trajectories[_tIdx]['rewards']
                if _tIdx == 0: rTotal = rs
                else: rTotal = np.concatenate((rTotal,rs))
                # Reward details      
            reward_contacts,reward_ctrls,reward_forwards,reward_headings,reward_survives = [],[],[],[],[]
            tickSum = 0
            for _traj in trajectories:
                tickSum += _traj['rewards'].shape[0]
                cTraj = _traj['rDetails']
                for _iIdx in range(len(cTraj)):
                    reward_contacts.append(cTraj[_iIdx]['reward_contact'])
                    reward_ctrls.append(cTraj[_iIdx]['reward_ctrl'])
                    reward_forwards.append(cTraj[_iIdx]['reward_forward'])
                    reward_headings.append(cTraj[_iIdx]['reward_heading'])
                    reward_survives.append(cTraj[_iIdx]['reward_survive'])
            tickAvg = tickSum / _batchSize
            sumRwd = rTotal.sum() / _batchSize
            sumReward_contact = np.asarray(reward_contacts).sum() / _batchSize
            sumReward_ctrl = np.asarray(reward_ctrls).sum() / _batchSize
            sumReward_forward = np.asarray(reward_forwards).sum() / _batchSize
            sumReward_heading = np.asarray(reward_headings).sum() / _batchSize
            sumReward_survive = np.asarray(reward_survives).sum() / _batchSize
            print ("[%d/%d](#total:%d) sumRwd:[%.3f](cntct:%.3f+ctrl:%.3f+fwd:%.3f+head:%.3f+srv:%.3f) tickAvg:[%d]"%
                (_epoch,_maxEpoch,(_epoch+1)*_batchSize,sumRwd,
                sumReward_contact,sumReward_ctrl,sumReward_forward,sumReward_heading,sumReward_survive,tickAvg))
            
            # SHOW EVERY
            PLOT_EVERY = 20 
            DO_ANIMATE = False
            if ((_epoch%PLOT_EVERY)==0 ) | (_epoch==(_maxEpoch-1)):
                ret = run_episode_vid(self.env,self.policy,self.scaler)
                print ("  [^] sumRwd:[%.3f] Xdisp:[%.3f] hDisp:[%.1f]"%
                    (np.asarray(ret['rewards']).sum(),ret['xDisp'],ret['hDisp']))
                if MAKE_GIF:
                    display_frames_as_gif(ret['frames'])
                if SAVE_VID:
                    outputdata = np.asarray(ret['frames']).astype(np.uint8)
                    vidName = 'vids/ant_ppo_epoch%03d.mp4'%(_epoch)
                    skvideo.io.vwrite(vidName,outputdata)
                    print ("[%s] saved."%(vidName))
        print ("Done.") 