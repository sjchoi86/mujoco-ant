import os,math
import numpy as np
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
import matplotlib.pyplot as plt

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