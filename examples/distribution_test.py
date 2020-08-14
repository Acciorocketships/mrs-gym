from mrsgym.Util import *
from torch.distributions import *
import torch

N = 50
z = Uniform(low=torch.zeros(N,1), high=torch.ones(N,1))
xy_normal = Normal(torch.zeros(N,2), 1.0)
sphere_transform = SphereTransform(radius=0.25, within=True)
xy_circle = TransformedDistribution(xy_normal, [sphere_transform])
dist = CombinedDistribution([xy_circle, z], mixer='cat', dim=1)

print(dist.sample())

import pdb; pdb.set_trace()