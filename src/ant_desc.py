import gym,mujoco_py,warnings,time
gym.logger.set_level(40)
warnings.filterwarnings("ignore") 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym.envs import mujoco
from util import PID_class
from custom_ant import AntEnvCustom
np.set_printoptions(precision=2,linewidth=150)


# Init env
env = AntEnvCustom()
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
PID = PID_class(Kp=0.01,Ki=0.00001,Kd=0.001,windup=10000,sample_time=0.01,dim=8)

# 
secTrigger = 0 
refPosDeg = initPosDeg

# Run
maxTick = 10000
for tick in range(maxTick):
    # Plot
    env.render() 
    
    # Get time 
    sec = env.sim.data.time

    # Set refPos every setTrigger 
    if sec > secTrigger:
        secTrigger = secTrigger + 2
        minPosDeg = np.array([-30,30,-30,-70,-30,-70,-30,30])
        maxPosDeg = np.array([+30,70,+30,-30,+30,-30,+30,70])
        refPosDeg = minPosDeg + (maxPosDeg-minPosDeg)*np.random.rand(act_dim)
    
    # Current position (in Deg)
    cPosDeg = np.asarray(obs[5:13])*180.0/np.pi

    # PID controller 
    degDiff = cPosDeg-refPosDeg
    PID.update(degDiff,sec)
    action = PID.output
    
    # Step 
    actionRsh = action[[6,7,0,1,2,3,4,5]] # rearrange
    obs, reward, done, rwdDetal = env.step(actionRsh.astype(np.float16))
    
    # Print out
    DO_PRINT = True
    if DO_PRINT:
        print ('tick: [%d] sec: [%.2f] done: %s'%(tick,sec,done))
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
        print (' reward:    %.3f (fwd:%.3f ctrl:%.3f)'
            %(reward,rwdDetal['reward_forward'],rwdDetal['reward_ctrl']))
    
    