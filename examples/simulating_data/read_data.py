from helper.Trainer import Trainer


data_path = 'data/flocking_N=12.pt'

K = 0 # groups data into sequences of K+1 consecutive timesteps
data = Trainer(K=K)
data.load_trainer(path=data_path)

batch = data.get_batch(batch_size=16)
print("batch (random samples of {} consecutive timesteps): ".format(K+1), batch.keys())
print("A (batch x N x N x K+1) - ", batch['A'].shape)
print("X (batch x N x D x K+1) - ", batch['X'].shape)

idx = 0
episode = data.get_episode(idx=idx)
print("\nepisode (gets the entire {}th episode): ".format(idx), episode.keys())
print("X (episode_length x N x D) - ", episode['X'].shape)

episodes = data.get_episodes()
print("\nepisodes (gets all episodes): ", episodes.keys())
print("X (num_episodes x episode_length x N x D) - ", episodes['X'].shape)

idx = -1
state = data.get_state(idx=-1)
print("\nstate (gets the state at timestep {:1} and the preceding {:2} timesteps): ".format(idx, K), state.keys())
print("X (N x D x K+1) - ", state['X'].shape)