import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

num_states = env.observation_space.n
num_actions = env.action_space.n
print("numero de estados possiveis", num_states)
print("numero de acoes possiveis", num_actions)

recompensas_vencidas = 0

#parametros
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1.0
exploration_decay = 0.99
max_steps_per_episode = 200
target_reward = 16

Q = np.zeros((num_states, num_actions))
rewards_per_episode = []

def choose_action(state, exploration_prob):
    if np.random.uniform(0, 1) < exploration_prob:
        return env.action_space.sample()  # Ação aleatória
    else:
        return np.argmax(Q[state, :])  # Ação com maior valor Q

n_episodes = 1000
reward_total = 0

for episode in range(n_episodes):
    observation, info = env.reset()
    state = observation
    episode_reward = 0
    step = 0

    while step < max_steps_per_episode:
        action = choose_action(state, exploration_prob)
        observation, reward, terminated, truncated, info = env.step(action)

        next_max_q = np.max(Q[observation, :])  # Pegando o valor Q máximo do próximo estado
        target = reward + discount_factor * next_max_q

        Q[state, action] += learning_rate * (target - Q[state, action])

        state = observation
        episode_reward += reward
        step += 1

        if terminated or truncated:
            observation, info = env.reset()
        
        if episode_reward > target_reward:
            print("Jogo Vencido!!! Episódio: ", episode)
            print("Recompensa: ", episode_reward)
            recompensas_vencidas = recompensas_vencidas + episode_reward
            break
    
    exploration_prob *= exploration_decay
    print(f"Episode {episode + 1}: Recompensa Total = {episode_reward}") 
    rewards_per_episode.append(episode_reward)

env.close()
print("RECOMPENSA TOTAL: ", reward_total)
print("RECOMPENSA ACUMULADA: ", recompensas_vencidas)
plt.plot(rewards_per_episode)
plt.xlabel('Episódio')
plt.ylabel('Recompensa')
plt.title('Recompensa por Episódio')
plt.show()