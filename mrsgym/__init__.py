from gym.envs.registration import register
# Import anything here that should be available when the library is imported
# Ex: from airsim_gym.update_settings import *
from mrsgym.Util import *
from mrsgym.MRS import *
from mrsgym.BulletSim import *
from mrsgym.Object import *
from mrsgym.Quadcopter import *
from mrsgym.Environment import *
from mrsgym.EnvCreator import *
from mrsgym.MRSWrapper import *

register(
    id='mrs-v0',
    entry_point='mrsgym:MRS',
)

register(
	id='mrs-rllib-multiagent-v0',
	entry_point='mrsgym:MRS_RLlib_MultiAgent',
)

register(
	id='mrs-rllib-v0',
	entry_point='mrsgym:MRS_RLlib'
)