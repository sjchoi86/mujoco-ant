import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim 
import matplotlib.pyplot as plt

import skvideo.io

from custom_ant import AntEnvCustom # Custom ant
from lgrp_class import lgrp_class # Gaussian random path
from vae_class import vae_class # VAE

from util import PID_class,quaternion_to_euler_angle,multi_dim_interp,Scaler,display_frames_as_gif

class antTrainEnv_class(object):
    def __init__(self,_tMax=3,_nAnchor=20,_maxRepeat=3,_hypGain=1/3,_hypLen=1/4,_pGain=0.01,
            _zDim=16,_hDims=[64,64],_vaeActv=tf.nn.elu,
            _PLOT_GRP=True):
        # Some parameters
        self.tMin = 0
        self.tMax = _tMax
        self.nAnchor = _nAnchor
        self.maxRepeat = _maxRepeat
        
        # Initialize Ant gym 
        self.env = AntEnvCustom()
        # GRP sampler (prior)
        nDataPrior = 2
        nTest = (int)((self.tMax-self.tMin)/self.env.dt)
        tData = np.linspace(start=self.tMin,stop=self.tMax,num=nDataPrior).reshape((-1,1))
        xData = np.random.rand(nDataPrior,self.env.actDim) # Random positions 
        # xData[0,:] = xData[0,:]/3.0+1.0/3.0
        xData[-1,:] = (xData[0,:]+xData[-1,:])/2.0 # Initial and final poses must be the same
        xData[0,:] = xData[-1,:]
        lData = np.ones(shape=(nDataPrior,1))
        tTest = np.linspace(start=self.tMin,stop=self.tMax,num=nTest).reshape((-1,1))
        lTest = np.ones(shape=(nTest,1))

        # hyp = {'gain':1/3,'len':1/4,'noise':1e-8} # <= This worked fine
        hyp = {'gain':_hypGain,'len':1/4,'noise':1e-8}
        self.GRPprior = lgrp_class(_name='GPR Prior',_tData=tData,
                                   _xData=xData,_lData=lData,_tTest=tTest,
                                   _lTest=lTest,_hyp=hyp)
            
        # GRP posterior
        tData = np.linspace(start=self.tMin,stop=self.tMax,num=self.nAnchor).reshape((-1,1))
        xData = np.random.rand(self.nAnchor,self.env.actDim) # Random positions 
        lData = np.ones(shape=(self.nAnchor,1))
        lData[1:self.nAnchor-1] = 0.8
        self.GRPposterior = lgrp_class(_name='GPR Posterior',_tData=tData,
                                   _xData=xData,_lData=lData,_tTest=tTest,
                                   _lTest=lTest,_hyp=hyp)
        if _PLOT_GRP:
            self.GRPprior.plot_all(_nPath=1,_figsize=(12,4))
            self.GRPposterior.plot_all(_nPath=1,_figsize=(12,4))

        # PID controller (Kp=0.01,Ki=0.00001,Kd=0.002,windup=5000)
        self.PID = PID_class(Kp=_pGain,Ki=0.00001,Kd=0.002,windup=5000,
                        sample_time=self.env.dt,dim=self.env.actDim)
        # VAE (this will be our policy function)
        self.VAE = vae_class(_name='VAE',_xDim=self.nAnchor*self.env.actDim,
                             _zDim=_zDim,_hDims=_hDims,_cDim=0,
                             _actv=_vaeActv,_outActv=None,_qActv=tf.nn.tanh,
                             _bn=None,_optimizer=tf.train.AdamOptimizer,
                             _optm_param={'lr':0.001,'beta1':0.9,'beta2':0.9,'epsilon':1e-8},
                             _VERBOSE=False)
        # Reward Scaler
        self.qScaler = Scaler(1)

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
        rewardSum = 0
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
            rewardSum += reward
            
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
        xFinal = self.env.get_body_com("torso")[0]
        hFinal = self.env.get_heading()
        xDisp = xFinal - xInit
        hDisp = hFinal - hInit

        rAvg = rewardSum/tick
        rContactAvg = rContactSum/tick
        rCtrlAvg = rCtrlSum/tick
        rFwdAvg = rFwdSum/tick
        rHeadingAvg = rHeadingSum/tick
        rSrvAvg = rSrvSum/tick


        if _VERBOSE:
            print ("Repeat:[%d] done. Avg reward is [%.3f]. xDisp is [%.3f]. hDisp is [%.3f]."
                   %(self.maxRepeat,rewardSum/tick,xDisp,hDisp))
        ret = {'rewards':rewards,'frames':frames,'titleStrs':titleStrs,
              'xDisp':xDisp,'hDisp':hDisp,
              'rAvg':rAvg,'rContactAvg':rContactAvg,'rCtrlAvg':rCtrlAvg,
              'rFwdAvg':rFwdAvg,'rHeadingAvg':rHeadingAvg,'rSrvAvg':rSrvAvg}
        return ret
    
    # Sample from GRP prior
    def sample_from_grp_prior(self):
        # Reset 
        nDataPrior = 2
        nTest = (int)((self.tMax-self.tMin)/self.env.dt)
        tData = np.linspace(start=self.tMin,stop=self.tMax,num=nDataPrior).reshape((-1,1))
        xData = np.random.rand(nDataPrior,self.env.actDim) # Random positions 
        # xData[0,:] = xData[0,:]/3.0+1.0/3.0
        xData[-1,:] = (xData[0,:]+xData[-1,:])/2.0 # Initial and final poses must be the same
        xData[0,:] = xData[-1,:]
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
        _SAVE_VID=True,_MAKE_GIF=False,_PLOT_GRP=False,_PLOT_EVERY=5):
        self.sess = _sess

        # Initialize VAE weights
        self.sess.run(tf.global_variables_initializer()) 
        
        # Expirence memory
        xList = np.zeros((_batchSize,self.env.actDim*self.nAnchor))
        qList = np.zeros((_batchSize))
        xLists = ['']*_maxEpoch
        qLists = ['']*_maxEpoch
        xDispList = np.zeros((_batchSize))
        hDispList = np.zeros((_batchSize))
        rAvgList = np.zeros((_batchSize))
        rContactList = np.zeros((_batchSize))
        rCtrlList = np.zeros((_batchSize))
        rFwdList = np.zeros((_batchSize))
        rHeadingList = np.zeros((_batchSize))
        rSrvList = np.zeros((_batchSize))

        for _epoch in range(_maxEpoch):
            priorProb = 0.1+0.4*np.exp(-4*(_epoch/_maxEpoch)**2) # Schedule eps-greedish..
            for _iter in range(_batchSize):  
                np.random.seed(seed=(_epoch*_batchSize+_iter)) # ??
                if (np.random.rand()<priorProb) | (_epoch==0): # Sample from prior
                    avgRwd,ret = self.unit_rollout_from_grp_prior(self.maxRepeat)
                else: # Sample from posterior (VAE)
                    sampledX = self.VAE.sample(_sess=self.sess).reshape((self.nAnchor,self.env.actDim))
                    sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
                    self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=0.6+0.3*(1-priorProb))
                    avgRwd,ret = self.unit_rollout_from_grp_posterior(self.maxRepeat)
                # Get anchor points
                xInterp = self.get_anchor_from_traj(ret['sampledTraj'])
                xVec = np.reshape(xInterp,newshape=(1,-1))
                # Append
                xList[_iter,:] = xVec
                qList[_iter] = avgRwd
                xDispList[_iter] = ret['xDisp']
                hDispList[_iter] = ret['hDisp']
                rAvgList[_iter] = ret['rAvg']
                rContactList[_iter] = ret['rContactAvg']
                rCtrlList[_iter] = ret['rCtrlAvg']
                rFwdList[_iter] = ret['rFwdAvg']
                rHeadingList[_iter] = ret['rHeadingAvg']
                rSrvList[_iter] = ret['rSrvAvg']
                # Print 
                PRINT_EACH_ROLLOUT = False
                if PRINT_EACH_ROLLOUT:
                    print ("  [%d/%d][%d/%d] rwd:[%.2f] xDisp:[%.3f] hDisp:[%.1f]"%
                        (_epoch,_maxEpoch,_iter,_batchSize,avgRwd,ret['xDisp'],ret['hDisp']))
            # Train
            xLists[_epoch] = xList
            qLists[_epoch] = qList
            # Get the best out of previous episodes 
            for _bIdx in range(0,5):
                if _bIdx == 0: # Add current one for sure 
                    xAccList = xList
                    qAccList = qList
                else:
                    xAccList = np.concatenate((xAccList,xLists[max(0,_epoch-_bIdx)]),axis=0)
                    qAccList = np.concatenate((qAccList,qLists[max(0,_epoch-_bIdx)]))
            # Add high q episodes (batchSize)
            sortedIdx = np.argsort(-qAccList)
            xTrain = xAccList[sortedIdx[:_batchSize],:]
            qTrain = qAccList[sortedIdx[:_batchSize]]
            # Add current episodes (batchSize)
            xTrain = np.concatenate((xTrain,xList),axis=0)
            qTrain = np.concatenate((qTrain,qList))
            # Add random episodes (batchSize)
            randIdx = np.random.permutation(xAccList.shape[0])[:_batchSize]
            xRand = xAccList[randIdx,:]
            qRand = qAccList[randIdx]
            xTrain = np.concatenate((xTrain,xRand),axis=0)
            qTrain = np.concatenate((qTrain,qRand))
            
            # Train
            self.qScaler.update(qTrain) # Update Q scaler
            qScale,qOffset = self.qScaler.get() # Scaler 
            self.VAE.train(_sess=self.sess,_X=xTrain,_Y=None,_C=None,_Q=qScale*(qTrain-qOffset),
                            _maxIter=_nIter4update,_batchSize=64,_PRINT_EVERY=0,_PLOT_EVERY=0,_INIT_VAR=False)
            # Print
            print ("[%d/%d](#total:%d) avgRwd:[%.3f] XdispMean:[%.3f] XdispVar:[%.3f] absHdispMean:[%.1f] priorProb:[%.2f]"%
                (_epoch,_maxEpoch,(_epoch+1)*_batchSize,qList.mean(),
                    xDispList.mean(),xDispList.var(),np.abs(hDispList).mean(),priorProb))
            print (" rAvg:[%.3f] = (contact:%.3f+ctrl:%.3f+fwd:%.3f+heading:%.3f+survive:%.3f)"%
                (rAvgList.mean(),rContactList.mean(),rCtrlList.mean(),rFwdList.mean(),rHeadingList.mean(),
                    rSrvList.mean()))
            # SHOW EVERY 
            if ((_epoch%_PLOT_EVERY)==0 ) | (_epoch==(_maxEpoch-1)):
                # Rollout 
                sampledX = self.VAE.sample(_sess=self.sess).reshape((self.nAnchor,self.env.actDim))
                sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
                self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=0.8)
                avgRwd,ret = self.unit_rollout_from_grp_mean(
                        _maxRepeat=self.maxRepeat,_DO_RENDER=True)
                print ("    [Rollout w/ GRP mean] avgRwd:[%.3f](%.3f=contact:%.2f+ctrl:%.2f+fwd:%.2f+heading:%.2f+survive:%.2f) Xdisp:[%.3f] hDisp:[%.1f]"%
                        (avgRwd,ret['rAvg'],
                        ret['rContactAvg'],ret['rCtrlAvg'],ret['rFwdAvg'],ret['rHeadingAvg'],ret['rSrvAvg'],
                        ret['xDisp'],ret['hDisp']))
                # Make video
                if _SAVE_VID: 
                    outputdata = np.asarray(ret['frames']).astype(np.uint8)
                    vidName = 'vids/ant_dlpg_epoch%03d.mp4'%(_epoch)
                    skvideo.io.vwrite(vidName,outputdata)
                    print ("     Video [%s] saved."%(vidName))
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
                        sampledX = (sampledX-sampledX.min())/(sampledX.max()-sampledX.min())
                        self.set_anchor_grp_posterior(_anchors=sampledX,_levBtw=0.9)
                        self.GRPposterior.plot_all(_nPath=1,_figsize=(8,3))
                        avgRwd,ret = self.unit_rollout_from_grp_mean(_maxRepeat=self.maxRepeat,_DO_RENDER=False)
                        print ("  [GRP-%d] avgRwd:[%.3f](%.3f=contact:%.2f+ctrl:%.2f+fwd:%.2f+heading:%.2f+survive:%.2f) Xdisp:[%.3f] hDisp:[%.1f]"%
                                (_i,avgRwd,ret['rAvg'],
                                ret['rContactAvg'],ret['rCtrlAvg'],ret['rFwdAvg'],ret['rHeadingAvg'],ret['rSrvAvg'],
                                ret['xDisp'],ret['hDisp']))
