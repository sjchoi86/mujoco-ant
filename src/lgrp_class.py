from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np
import matplotlib.pyplot as plt

def kernel_levse(_X1,_X2,_L1,_L2,_hyp={'gain':1,'len':1,'noise':1e-8}):
    hyp_gain = float(_hyp['gain'])**2
    hyp_len  = float(_hyp['len'])
    xDists = cdist(_X1,_X2,'euclidean')
    lDists = cdist(_L1,_L2,'cityblock')
    Kse = hyp_gain*np.exp(-(xDists** 2)/(hyp_len**2))
    Klev = np.cos(np.pi/2.0*lDists)
    K = Klev*Kse
    return K,{'xDists':xDists,'lDists':lDists,'Kse':Kse,'Klev':Klev}

class lgrp_class(object):
    def __init__(self,_name,_tData,_xData,_lData,_hyp,_tTest,_lTest):
        # Parse input args 
        self.name = _name
        self.hyp = _hyp
        self.tTest = _tTest
        self.lTest = _lTest
        self.nTest = self.tTest.shape[0]
        # Set data
        self.set_data(_tData=_tData,_xData=_xData,_lData=_lData,_EPS_RU=True)

    def set_data(self,_tData,_xData,_lData,_EPS_RU=True):
        self.tData = _tData
        self.tMin,self.tMax = self.tData.min(),self.tData.max()
        self.xData = _xData
        self.lData = _lData
        self.nData = self.xData.shape[0]
        self.dim = self.xData.shape[1]
        # Do epsilon run-up
        if _EPS_RU: 
            self._eps_runup()
        # Compute GRP
        self._define_grp()
        
    def _eps_runup(self):
        # Epsilon run-up parameters
        tEps = 2e-2
        xEpsRate = tEps
        tData,xData,lData,nData = self.tData,self.xData,self.lData,self.nData
        # t-eps
        tData = np.insert(tData,1,tData[0,0]+tEps,0)
        tData = np.insert(tData,-1,tData[-1,0]-tEps, 0)
        # L-eps
        lData = np.insert(lData,1,lData[0,0],0)
        lData = np.insert(lData,-1,lData[-1,0],0) 
        # X-start-eps
        diff = xData[1,:]-xData[0,:]
        if np.linalg.norm(diff) <= 1e-6: uvec = 0.0*diff
        else: uvec = diff / np.linalg.norm(diff)
        peru = xData[0,:] + uvec*xEpsRate
        xData = np.insert(xData,1,peru,0)
        # X-final-eps
        diff = xData[-1,:]-xData[-2,:]
        if np.linalg.norm(diff) <= 1e-6: uvec = 0.0*diff
        else: uvec = diff / np.linalg.norm(diff)
        peru = xData[-1,:] - uvec*xEpsRate
        xData = np.insert(xData,-1,peru,0)
        # Increase n
        nData += 2
        self.tData,self.xData,self.lData,self.nData = tData,xData,lData,nData
        
    def _define_grp(self):
        tData,xData,lData,nData,hyp,tTest,lTest \
            = self.tData,self.xData,self.lData,self.nData,self.hyp,self.tTest,self.lTest

        Ktd_mu,_ = kernel_levse(tTest,tData,lTest,np.ones_like(lData),hyp)
        Ktd_var,_ = kernel_levse(tTest,tData,lTest,lData,hyp)
        Kdd_mu,_ = kernel_levse(tData,tData,np.ones_like(lData),np.ones_like(lData),hyp)
        Kdd_var,_ = kernel_levse(tData,tData,lData,lData,hyp)
        Ktt,_ = kernel_levse(tTest,tTest,lTest,lTest,hyp)
        xDataMean = xData.mean(axis=0)
        muTest = np.matmul(Ktd_mu,np.linalg.solve(
            Kdd_mu+1e-9*np.eye(nData),xData-xDataMean))+xDataMean
        _varTest = Ktt - np.matmul(Ktd_var,np.linalg.solve(
            Kdd_var+1e-9*np.eye(nData),Ktd_var.T))
        Rtt = np.linalg.cholesky(_varTest+1e-9*np.eye(self.nTest))
        varTest = np.diag(_varTest).reshape((-1,1))
        self.muTest,self._varTest,self.Rtt,self.varTest \
            = muTest,_varTest,Rtt,varTest
            
    def sample_paths(self,_nPath=1):
        sampledPaths = []
        for pIdx in range(_nPath):
            R = np.random.randn(self.nTest,self.dim)
            sampledPath = self.muTest+np.matmul(self.Rtt,R)
            sampledPaths.append(sampledPath)
        return sampledPaths
    
    def plot_all(self,_nPath=1,_figsize=(12,6)):
        _titleStr=self.name
        # Plot mu, var, and sampled paths
        dim,varTest,tTest,muTest,tData,xData \
            = self.dim,self.varTest,self.tTest,self.muTest,self.tData,self.xData
        sampledPaths = self.sample_paths(_nPath=_nPath)
        cmap = plt.get_cmap('inferno')
        colors = [cmap(i) for i in np.linspace(0,1,dim+1)]
        plt.figure(figsize=_figsize)
        for dIdx in range(dim):
            # Plot 2-sigma
            sigmaTest = np.sqrt(varTest)
            hVar=plt.fill_between(tTest.squeeze(),
                             (muTest[:,dIdx:dIdx+1]-2*sigmaTest).squeeze(),
                             (muTest[:,dIdx:dIdx+1]+2*sigmaTest).squeeze(),
                             facecolor=colors[dIdx], interpolate=True, alpha=0.2)
            # Plot sampled paths
            for sIdx in range(_nPath):
                hSample,=plt.plot(tTest,sampledPaths[sIdx][:,dIdx:dIdx+1],
                                  linestyle='--',color=colors[dIdx],lw=1)
            # Plot mu
            hMu,=plt.plot(tTest,muTest[:,dIdx],linestyle='-',color=colors[dIdx],lw=3)
            hData,=plt.plot(tData,xData[:,dIdx],linestyle='None',marker='o',
                            ms=5,mew=2,mfc='w',color=colors[dIdx])
            plt.xlim(self.tMin-0.1,self.tMax+0.1) 
            plt.ylim(-0.2,1.2)
        plt.xlabel('Time [s]', fontsize=20)
        plt.title(_titleStr,fontsize=20)
        plt.legend([hData,hMu,hVar,hSample],
               ['Anchor Points','Mean Function','Variance Function','Sampled Paths'],
                   fontsize=13)
        plt.show()