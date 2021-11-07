from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
from dqn_agent import Agent
import json

#env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[256, 256], drop_p=0.0, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=100):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        hidden_layers: list of integers, the sizes of the hidden layers
    """
    agent = Agent(state_size=37, action_size=4, seed=seed, hidden_layers=hidden_layers, GAMMA=GAMMA, LR=LR, UPDATE_EVERY=UPDATE_EVERY)
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=10)   # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        done = False
        while (done == False):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]  
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\r\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            if np.mean(scores_window)<13.0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                agent.save_checkpoint('checkpoint_trained.pth')

            else:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                agent.save_checkpoint('checkpoint_solved.pth')
    return scores


scores = dict()

with open("scores.json", 'r+') as dataFile:
    scores = json.loads(dataFile.readline())
 
scores['hidden_layers = [256,256], drop_p=0.0], GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50'] = dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[128, 128], drop_p=0.0, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50)
scores['hidden_layers = [256,256], drop_p=0.2], GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50'] = dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[128, 128], drop_p=0.2, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50)
scores['hidden_layers = [128,128], drop_p=0.0], GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50'] = dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[128, 128], drop_p=0.0, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50)
scores['hidden_layers = [128,128], drop_p=0.2], GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50'] = dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[128, 128], drop_p=0.2, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50)
scores['hidden_layers = [64,64],   drop_p=0.0], GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50'] = dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[64, 64],   drop_p=0.0, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50)
scores['hidden_layers = [64,64],   drop_p=0.2], GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50'] = dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0, hidden_layers=[64, 64],   drop_p=0.2, GAMMA=0.99, LR=5e-4, UPDATE_EVERY=50)

json = json.dumps(scores)
f = open("scores.json","w")
f.write(json)
f.close()


import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [20, 20]
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

nr_of_plots = len(scores)
fig, axs = plt.subplots(nr_of_plots)
fig.subplots_adjust(hspace=.5)
idx_plot = 0
for key, value in scores.items():
    score = scores[key]
    score_mean = running_mean(score, 100)
    axs[idx_plot].set_title(key)
    axs[idx_plot].plot(score)
    axs[idx_plot].plot(score_mean)
    axs[idx_plot].grid()
    idx_plot = idx_plot + 1
plt.show()