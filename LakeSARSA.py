import gymnasium as gym
import numpy as np
import wandb
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1",is_slippery = True)#, render_mode="human")

num_states = env.observation_space.n #numero estados possiveis
num_actions = env.action_space.n #numero acoes possiveis

print("numero de estados possiveis", num_states)
print("numero de acoes possiveis", num_actions)

#parametros
learning_rate = 0.1 ##taxa aprendizado
discount_factor = 0.99  ##fator de desconto
exploration_prob = 1.0 ##prob exploracao inicial
exploration_decay = 0.99 ##taxa decaimento da prob de exploração
max_steps_per_episode = 100 ##teste

target_reward = 1 ## 1 signfica que foi atingido o objetivo (chegar ao bau)


Q = np.zeros((num_states, num_actions))

rewards_per_episode = []
reward_total = 0

#funcao de escolha da acao com base na tabela Q
def choose_action(state, exploration_prob):
    if np.random.uniform(0, 1) < exploration_prob:
        a = env.action_space.sample()  # Ação aleatória
        return a
    else:
        return np.argmax(Q[state, :])  # Ação com maior valor Q

n_episodes = 100

for episode in range (n_episodes):

    observation, info = env.reset()
    state = observation
    episode_reward = 0
    step = 0

    while step < max_steps_per_episode:

        action = choose_action(state, exploration_prob)
        observation, reward, terminated, truncated, info = env.step(action) ##passando informacoes do proximo passo
        next_action = choose_action(observation, exploration_prob) ##nova acao ##parametro observation = informacoes

        current_q = Q[state, action]
        next_q = Q[observation, next_action]
        target = reward + discount_factor * next_q
        Q[state, action] = current_q + learning_rate * (target - current_q)

        state = observation
        episode_reward += reward
        step += 1

        if terminated or truncated:
            observation, info = env.reset()
        
        if episode_reward >= target_reward:
            print("Jogo Vencido!!! Episódio: ", episode)
            print("Recompensa: ", episode_reward)
            reward_total += episode_reward
            break
        
    
    exploration_prob *= exploration_decay

    print(f"Episode {episode + 1}: Recompensa Total = {episode_reward}") 
    rewards_per_episode.append(episode_reward)
    reward_total += episode_reward

env.close()
print("RECOMPENSA TOTAL: ", reward_total)

plt.plot(rewards_per_episode)
plt.xlabel('Episódio')
plt.ylabel('Recompensa')
plt.title('Recompensa por Episódio')
plt.show()

