import gym,mujoco_py,warnings,time
gym.logger.set_level(40)
warnings.filterwarnings("ignore") 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs import mujoco
np.set_printoptions(precision=2,linewidth=150)
print ("Packages Loaded")

# PID class
class PID_class:
    def __init__(self,Kp=0.001,Ki=0.0,Kd=0.0001
         ,windup=100,sample_time=0.01,dim=1):
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
        delta_time = self.current_time - self.last_time
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



# Init env
env = mujoco.AntEnv()
obs_dim = env.observation_space.shape[0] # 111
act_dim = env.action_space.shape[0] # 8
env.reset() # Reset 
init_qpos = env.init_qpos
init_qpos[7:] = np.asarray([0,90,0,-90,0,-90,0,90])*np.pi/180.0
env.set_state(init_qpos,env.init_qvel)
env.render()
print ("Environment initialized. obs_dim:[%d] act_dim:[%d]"
    %(obs_dim,act_dim))
obs, reward, done, _ = env.step(np.zeros(act_dim))

# Set reference position 
refPosDeg1 = np.array([
    +0, 90
    ,0, -90
    ,0, -90
    ,0, 90
    ],dtype=float)

refPosDeg2 = np.array([
    +40, 20  # +0, 90
    ,40, -20 # ,0, -90
    ,40, -20 # ,0, -90
    ,40, 20 # ,0, 90
    ],dtype=float)

# Set PID
PID = PID_class(Kp=0.004,Ki=0.01,Kd=0.001,windup=100,sample_time=0.05)

# Run
for i in range(10000):
    # Plot
    env.render() 
    
    # Time
    t = env.sim.data.time

    # Set refPos
    if t < 10:
        refPosDeg = refPosDeg1
    elif t < 20:
        refPosDeg = refPosDeg2
    elif t < 30:
        refPosDeg = refPosDeg1
    elif t < 40:
        refPosDeg = refPosDeg2
    elif t < 50:
        refPosDeg = refPosDeg1
    elif t < 60:
        refPosDeg = refPosDeg2
    elif t < 70:
        refPosDeg = refPosDeg1
    elif t < 80:
        refPosDeg = refPosDeg2

    # Current position (in Deg)
    cPosDeg = np.asarray(obs[5:13])*180.0/np.pi

    # PID controller 
    degDiff = cPosDeg-refPosDeg
    PID.update(degDiff,t)
    action = PID.output
    
    # Step 
    actionRsh = action[[6,7,0,1,2,3,4,5]]
    obs, reward, done, _ = env.step(actionRsh.astype(np.float16))
    
    # Print out
    print ('t: %.2f\n degDiff: %s'%(t,degDiff))
    print (' action: %s'%(action))
    