import torch
import numpy as np
from collections import deque
from helper.Trainer import Trainer
from helper.Util import *


def generate_mrs(env, model=None, trainer=None, datapoints=50000, episode_length=float('inf'), action_fn=lambda a, s: a):

		if trainer is None:
			trainer = Trainer(K=env.K_HOPS)

		env.RETURN_A = True
		state = None
		context = {}
		done = True
		t = 0

		while True:
			# Reset environment every episode_length steps or on collision
			if done:
				state = env.reset()
				context = {"A": env.calc_Ak()}
				done = False
				t = 0
			# Episode Length
			t += 1
			if t >= episode_length:
				done = True
			# Compile State
			Xk = state
			Ak = context["A"]
			# Get action
			action = model.forward(Ak.unsqueeze(0), Xk.unsqueeze(0))[0,:,:]
			expert_action = action.clone()
			action = action_fn(action.cpu(), Xk)
			# Log state
			trainer.set_state(A=Ak[:,:,0].cpu(), X=Xk[:,:,0].cpu(), done=done, expert=expert_action.cpu(), context=context)
			# Step environment forward
			state, reward, envdone, context = env.step(action)
			done = done | envdone
			# Stopping Conditions
			num_datapoints = len(trainer.history["X"])
			if num_datapoints % 100 == 0:
				print("Samples: %d" % num_datapoints)
			if num_datapoints >= datapoints:
				break

		return trainer