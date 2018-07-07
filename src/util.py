import numpy as np

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

