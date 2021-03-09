from lunar_landing_solved import Agent
from lunar_lander import LunarLanderEnvironment
from rl_glue import RLGlue
from tqdm import tqdm
import numpy as np


agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9,
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 5000, #50000
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001,
    'weights': np.load("ActionValueNetworkWeightsTrained.npy", allow_pickle=True)
}

experiment_parameters = {
    'timeout': 1000,
    'num_episodes': 10,

}

environment_parameters = {'render': True}
rl_glue = RLGlue(LunarLanderEnvironment, Agent)
# print(np.load("ActionValueNetworkWeights.npy", allow_pickle=True))


for run in range(1, 2):
    agent_parameters["seed"] = run
    agent_parameters["network_config"]["seed"] = run
    agent_parameters["seed"] = run
    rl_glue.rl_init(agent_parameters, environment_parameters)

    for episode in tqdm(range(1, experiment_parameters["num_episodes"] + 1)):

        # run episode
        rl_glue.rl_episode(experiment_parameters["timeout"])
        episode_reward = rl_glue.rl_agent_message("get_sum_reward")
        print(f"\nEpisode reward: {episode_reward}")


