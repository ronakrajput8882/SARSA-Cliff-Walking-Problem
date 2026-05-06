import random
import numpy as np
import gymnasium as gym

Q = np.zeros((48,4))
gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

env = gym.make("CliffWalking-v1")

def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
    
for episode in range(episodes):
    
    done = False
    state, _ = env.reset()
    action = epsilon_greedy(state)
    
    total_reward = 0
    episode_len = 0
    
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        next_action = epsilon_greedy(next_state)
        
        Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
        state = next_state
        action = next_action
    
        total_reward += reward
        episode_len += 1
    
    print(f"Episode {episode + 1 }/{episodes} total reward = {total_reward} & Ep length = {episode_len}")
    
    
env = gym.make("CliffWalking-v1", render_mode="human")
state, _ =env.reset()
done = False
total_reward = 0
episode_len = 0
while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    episode_len += 1
env.close()