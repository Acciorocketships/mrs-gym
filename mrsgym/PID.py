from collections import deque
from scipy import signal
from mrsgym.BulletSim import *

class PID:

	def __init__(self, **kwargs):
		# Constants
		self.Kp = 1.0
		self.Ki = 1.0
		self.Kd = 1.0
		self.Imax = float('inf')
		self.Omax = float('inf')
		self.Istart = 0.0
		self.wc = 1.0
		self.hist_len = 10
		self.set_constants(kwargs)
		# Variables
		self.hist = deque([0 for _ in range(self.hist_len)])
		self.int = torch.tensor(self.Istart)
		self.filter = signal.butter(2, self.wc) if (self.wc < 1.0) else None


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val


	def update(self, err, dt=None, err_dot=None):
		# P, I, D calculation
		if dt is None:
			dt = BulletSim.DT
		self.hist.append(err)
		if len(self.hist) > self.hist_len:
			self.hist.popleft()
		if err_dot is None:
			if self.wc < 1.0:
				yi = torch.tensor(signal.lfilter_zi(self.filter[0], self.filter[1]))
				y, _ = signal.lfilter(self.filter[0], self.filter[1], torch.tensor(self.hist), zi=yi*self.hist[0])
				err_dot = y[-1]
			else:
				err_dot = (self.hist[-1] - self.hist[-2]) / dt
		self.int = torch.clamp(self.int + err * dt, -self.Imax, self.Imax)
		err_int = self.int
		# Control calculation
		control = self.Kp * err + self.Ki * err_int + self.Kd * err_dot
		control = torch.clamp(control, -self.Omax, self.Omax)
		return control


	def limit_control(self, val, limit):
		C = 0.1
		if not isinstance(val, torch.Tensor):
			val = torch.tensor(val)
		ctrl = min(torch.exp(C * abs(val) / limit), torch.tensor(2*self.Omax))
		return ctrl * -torch.sign(val)