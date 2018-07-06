import gym,mujoco_py,warnings,time
gym.logger.set_level(40)
warnings.filterwarnings("ignore") 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs import mujoco
from util import PID_class
np.set_printoptions(precision=2,linewidth=150)
print ("Packages Loaded")


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
initPosDeg = np.array([
    +0, 90
    ,0, -90
    ,0, -90
    ,0, 90
    ],dtype=float)

# Set PID
PID = PID_class(Kp=0.01,Ki=0.001,Kd=0.001,windup=100,sample_time=0.01)

# 
secTrigger = 10
refPosDeg = initPosDeg

# Run
for i in range(10000):
    # Plot
    env.render() 
    
    # Time
    sec = env.sim.data.time

    # Set refPos every setTrigger 
    if sec > secTrigger:
        secTrigger = secTrigger + 5
        degRange = 30.
        refPosDeg = initPosDeg-degRange+2*degRange*np.random.rand(8)
    
    # Current position (in Deg)
    cPosDeg = np.asarray(obs[5:13])*180.0/np.pi

    # PID controller 
    degDiff = cPosDeg-refPosDeg
    PID.update(degDiff,sec)
    action = PID.output
    
    # Step 
    actionRsh = action[[6,7,0,1,2,3,4,5]] # rearrange
    obs, reward, done, _ = env.step(actionRsh.astype(np.float16))
    
    # Print out
    print ('sec: %.2f\n degDiff: %s'%(sec,degDiff))
    print (' action: %s'%(action))
    