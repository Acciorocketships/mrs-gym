import torch

# Dyanmics and Parameter Defaults:
# https://support.dce.felk.cvut.cz/mediawiki/images/5/5e/Dp_2017_gopalakrishnan_eswarmurthi.pdf

class QuadControl:

	def __init__(self, **kwargs):
		self.THRUST_COEFF = 1.5
		self.DRAG_COEFF = 1.3
		self.ARM_LENGTH = 0.175
		self.MASS = 0.5
		self.set_constants(kwargs)

	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val

	# Input: angular velocity of each motor (4,)
	# Output: force at each motor and the torque in the Z direction (5,)
	# speed units: rad/s, force units: N
	def speed_to_force(self, speed=[0,0,0,0]):
		if not isinstance(speed, torch.Tensor):
			speed = torch.tensor(speed)
		w2 = speed ** 2
		forces = torch.zeros(5)
		forces[:4] = self.THRUST_COEFF * w2
		forces[4] = self.DRAG_COEFF * (w2[0]+w2[2]-w2[1]-w2[3])
		return forces

	# Input: [Fthrust, Tyaw, Tpitch, Troll]
	# Fthrust units: N, Typr units: Nm, output units: rad/s
	def control_to_speed(self, control=[0,0,0,0]):
		if not isinstance(control, torch.Tensor):
			control = torch.tensor(control)
		wrdiff = control[3] / self.THRUST_COEFF / self.ARM_LENGTH # = w1^2 - w3^2
		wpdiff = control[2] / self.THRUST_COEFF / self.ARM_LENGTH # = w0^2 - w2^2
		wydiff = control[1] / self.DRAG_COEFF # = w0^2 - w1^2 + w2^2 - w3^2
		wtsum = control[0] / self.THRUST_COEFF # = w0^2 + w1^2 + w2^2 + w3^2
		w0s = (((wydiff + wtsum) / 2) + wpdiff) / 2
		w1s = (((wtsum - wydiff) / 2) + wrdiff) / 2
		w2s = w0s - wpdiff
		w3s = w1s - wrdiff
		speed = torch.sqrt(torch.tensor([w0s,w1s,w2s,w3s]))
		return speed

	def control_to_force(self, control=[0,0,0,0]):
		return self.speed_to_force(self.control_to_speed(control))