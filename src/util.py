import os,math,glob,shutil,csv
import numpy as np
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf


def display_frames_as_gif(frames,_intv_ms=100,_figsize=(6,6),_fontsize=15,_titleStrs=None):
    plt.figure(figsize=_figsize)
    patch = plt.imshow(frames[0])
    if _titleStrs is None:
        title = plt.title('[%d/%d]'%(0,len(frames)),fontsize=_fontsize)
    else:
        title = plt.title('[%d/%d] %s'%(0,len(frames),_titleStrs[0]),fontsize=_fontsize)
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
        if _titleStrs is None:
            title.set_text('[%d/%d]'%(i,len(frames)))
        else:
            title.set_text('[%d/%d] %s'%(i,len(frames),_titleStrs[i]))
    anim = animation.FuncAnimation(plt.gcf(),animate,frames=len(frames),interval=_intv_ms)
    display(display_animation(anim,default_mode='loop'))
    os.system("rm None0000000.png")
    # !rm None0000000.png

# PID class
class PID_class:
    def __init__(self,Kp=0.01,Ki=0.00001,Kd=0.001,windup=10000,sample_time=0.01,dim=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.windup = windup
        self.sample_time = sample_time
        self.dim = dim
        self.current_time = 0.00
        self.last_time = self.current_time
        self.clear()
    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = np.zeros(shape=self.dim)
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = np.zeros(shape=self.dim)
        # Windup Guard
        self.windup_guard = self.windup*np.ones(shape=self.dim)
        self.output = np.zeros(shape=self.dim)
        # Init others
        self.current_time = 0.00
        self.last_time = self.current_time
    def update(self,feedback_value,current_time):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
            :align:   center
            Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time
        delta_time = self.current_time - self.last_time + 1e-6 
        # <= 1e-6: increase numerical stability
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            self.ITerm = np.clip(self.ITerm,-self.windup_guard,+self.windup_guard)
            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

# Convert quaternion to euler angle (for MuJoCo)
def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z

# Multi-dimensional interpolation
def multi_dim_interp(_x,_xp,_fp):
    nOut = _x.shape[0]
    dimOut = _fp.shape[1]
    _y = np.zeros(shape=(nOut,dimOut))
    for dIdx in range(dimOut): # For each dim
        _y[:,dIdx] = np.interp(_x[:,0],_xp[:,0],_fp[:,dIdx]) 
    return _y

# Track a given trajectory to pursuit 
def track_traj_with_pid(env,PID,pursuitTraj,maxRepeat,_VERBOSE=True):
    lenTraj = pursuitTraj.shape[0]
    cntRepeat = 0
    # Reset
    obs_dim,act_dim,dT = env.observation_space.shape[0],env.action_space.shape[0],env.dt
    obs = env.reset_model()
    PID.clear()
    for i in range(10): 
        sec = env.sim.data.time
        refPosDeg = pursuitTraj[0,:]
        cPosDeg = np.asarray(obs[5:13])*180.0/np.pi # Current pos
        degDiff = cPosDeg-refPosDeg
        PID.update(degDiff,sec)
        action = PID.output
        actionRsh = action[[6,7,0,1,2,3,4,5]] # rearrange
        obs,_,_,_ = env.step(actionRsh)
    PID.clear() # Reset PID after burn-in
    env.sim.data.time = 0
    tick,frames,titleStrs = 0,[],[]
    tickPursuit = 0
    timeList = np.zeros(shape=(lenTraj*maxRepeat)) # 
    cPosDegList = np.zeros(shape=(lenTraj*maxRepeat,act_dim)) # Current joint trajectory
    rPosDegList = np.zeros(shape=(lenTraj*maxRepeat,act_dim)) # Reference joint trajectory
    rewardSum = 0
    xInit = env.get_body_com("torso")[0]
    hInit = env.get_heading()
    while cntRepeat < maxRepeat:
        # Some info
        sec = env.sim.data.time # Current time 
        x = env.get_body_com("torso")[0]
        q = env.data.get_body_xquat('torso')
        rX,rY,rZ = quaternion_to_euler_angle(q[0],q[1],q[2],q[3])

        # Render 
        frame = env.render(mode='rgb_array',width=200,height=200)
        frames.append(frame) # Append to frames for futher animating 
        titleStrs.append('%.2fsec x:[%.2f] heading:[%.1f]'%(sec,x,rZ))

        # PID controller 
        timeList[tick] = sec
        refPosDeg = pursuitTraj[min(tickPursuit,lenTraj-1),:]
        rPosDegList[tick,:] = refPosDeg
        cPosDeg = np.asarray(obs[5:13])*180.0/np.pi # Current pos
        cPosDegList[tick,:] = cPosDeg
        degDiff = cPosDeg-refPosDeg
        PID.update(degDiff,sec)

        # Do action 
        action = PID.output
        actionRsh = action[[6,7,0,1,2,3,4,5]] # rearrange
        obs,reward,done,rwdDetal = env.step(actionRsh.astype(np.float16))
        rewardSum += reward

        # Print out
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
    xFinal = env.get_body_com("torso")[0]
    hFinal = env.get_heading()
    xDisp = xFinal - xInit
    hDisp = hFinal - hInit
    if _VERBOSE:
        print ("Repeat:[%d] done. Avg reward is [%.3f]. xDisp is [%.3f]. hDisp is [%.3f]."
               %(maxRepeat,rewardSum/tick,xDisp,hDisp))
    return timeList,cPosDegList,rPosDegList,frames,titleStrs

def gpu_sess():
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess

def cpu_sess():
    sess = tf.Session()
    return sess

def plot_imgs(_imgs,_imgSz=(28,28),_nR=1,_nC=10,_figsize=(15,2),_title=None,_titles=None):
    nr,nc = _nR,_nC
    fig = plt.figure(figsize=_figsize)
    if _title is not None:
        fig.suptitle(_title, size=15)
    gs  = gridspec.GridSpec(nr,nc)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(_imgs):
        ax = plt.subplot(gs[i]); plt.axis('off')
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_aspect('equal')
        if len(img.shape) == 1:
            img = np.reshape(img,newshape=_imgSz) 
        plt.imshow(img,cmap='Greys_r',interpolation='none')
        plt.clim(0.0, 1.0)
        if _titles is not None:
            plt.title(_titles[i],size=12)
    plt.show()

# Scaler 
class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0
        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
        
        Usage:
            # Scaler
            scaler = Scaler(obs_dim)
            scale, offset = scaler.get()
            obs = (obs - offset) * scale  # center and scale 
            scaler.update(unscaled) # Add to scaler
    """
    def __init__(self,obs_dim):
        self.obs_dim = obs_dim
        self.vars = np.zeros(self.obs_dim)
        self.means = np.zeros(self.obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True
    def reset(self):
        self.first_pass = True
    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        x = np.reshape(x,newshape=(-1,self.obs_dim))
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0] # Number of data 
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n # Total number of data
    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1/(np.sqrt(self.vars) + 0.1)/1.0, self.means 


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self,logName,now,_VERBOSE=True,_NOTUSE=False):
        """
        Args:
            logName: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        Usage:
            import numpy as np
            from datetime import datetime
            from util import Logger
            now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
            logger = Logger(logName='LoggerUsage',now=now,_VERBOSE=True)
            # Add log
            logger.log({'x': np.random.rand(),'y':1+np.random.rand()})
            # Write to csv
            logger.write(display=True)
            # Close csv
            logger.close()
        """
        self.VERBOSE = _VERBOSE
        self.NOTUSE = _NOTUSE
        if self.NOTUSE: return 
        path = os.path.join('log-files',logName,now)
        if not os.path.exists(path):
            os.makedirs(path)
            if self.VERBOSE: print ("mkdir: [%s]."%(path))

        path = os.path.join(path, 'log.csv')
        self.write_header = True
        self.log_entry = {}
        self.f = open(path, 'w')
        self.path = path
        if os.path.exists(self.path): print ("fopen: [%s]."%(self.path))
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout
        Args:
            display: boolean, print to stdout
        """
        if self.NOTUSE: return 
        if display:
            self.disp(self.log_entry)
        if self.write_header:
            fieldnames = [x for x in self.log_entry.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.write_header = False
        self.writer.writerow(self.log_entry)
        self.log_entry = {} # Empty window 

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        # print('\n')
    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.
        Args:
            items: dictionary of items to update
        """
        if self.NOTUSE: return 
        self.log_entry.update(items)
    def close(self):
        """ Close log file - log cannot be written after this """
        if self.NOTUSE: return 
        self.f.close()
        if self.VERBOSE:
            print ("[%s] saved."%(self.path))
