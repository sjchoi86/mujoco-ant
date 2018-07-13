import os,math
import numpy as np
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt
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
def track_traj_with_pid(env,PID,pursuitTraj,maxRepeat):
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
    print ("Repeat:[%d] done. Avg reward is [%.3f]. xDisp is [%.3f]. hDisp is [%.3f]."
           %(maxRepeat,rewardSum/tick,xDisp,hDisp))
    return timeList,cPosDegList,rPosDegList,frames,titleStrs

def gpu_sess():
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess  