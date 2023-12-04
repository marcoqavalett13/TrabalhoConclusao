import gymnasium as gym
import numpy as np
import wandb
import matplotlib.pyplot as plt


env = gym.make("Taxi-v3")

recompensas_vencidas = 0
reward_total = 0

num_states = env.observation_space.n #numero estados possiveis
num_actions = env.action_space.n #numero acoes possiveis
print("numero de estados possiveis", num_states)
print("numero de acoes possiveis", num_actions)

#parametros
learning_rate = 0.1##taxa aprendizado
discount_factor = 0.99  ##fator de desconto da taxa de aprendizado
exploration_prob = 1.0 ##prob exploracao inicial
exploration_decay = 0.99 ##taxa decaimento da prob de exploração
max_steps_per_episode = 200 ##teste

target_reward = 16.0 ##considera-se 8 de pontuação já como vencido o jogo pois demonstra que o algoritmo
#conseguiu buscar o passageiro e largar ele
#utilizei 16 para aumentar a dificuldade


Q = np.zeros((num_states, num_actions))
rewards_per_episode = []

#funcao de escolha da acao com base na tabela Q
def choose_action(state, exploration_prob):
    if np.random.uniform(0, 1) < exploration_prob:
        a = env.action_space.sample()  # Ação aleatória
        return a
    else:
        return np.argmax(Q[state, :])  # Ação com maior valor Q

n_episodes = 1000 #numero de episodios

for episode in range (n_episodes):

    observation, info = env.reset() #passando informacoes
    state = observation #estado recebe o valor de observation
    print("Estado Atual: ", state)
    episode_reward = 0
    step = 0
    win_episode = 0

    while step < max_steps_per_episode:

        action = choose_action(state, exploration_prob) #escolhe uma ação recebendo a taxa de exploracao e o estado atual como parametro
        observation, reward, terminated, truncated, info = env.step(action) ##passando informacoes do proximo passo
        next_action = choose_action(observation, exploration_prob) ##nova acao ##parametro observation = informacoes

        ##logica SARSA##
        current_q = Q[state, action] #valor atual do Q
        #print(f"Valor atual da tabela Q: ", Q)
        next_q = Q[observation, next_action]
        #print("Valor subsequente da tabela Q: ", next_q)
        target = reward + discount_factor * next_q##alvo
        Q[state, action] = current_q + learning_rate * (target - current_q) ##tabela Q recebe esses valores
        ##logica SARSA##

        state = observation
        episode_reward += reward #acumulação de recompensas
        step += 1 #avança 1 passo

        if terminated or truncated:
            observation, info = env.reset()
        
        if episode_reward > target_reward:
            print("Jogo Vencido!!! Episódio: ", episode)
            print("Recompensa: ", episode_reward)
            recompensas_vencidas += episode_reward
            win_episode = episode
            break
        
        #run.log({"Recompensa": episode_reward, "Episodios vencidos": win_episode,  })
        
    
    exploration_prob *= exploration_decay ##decaimento a probabilidade de exploração
    #no inicio é bom ter uma alta probabilidade de exploração para o algoritmo tentar novos caminhos e possibilidades
    #porem ao longo do algoritmo é altamente recomendavel reduzir essa taxa para que ele consiga efetivamente focar em apenas 
    #uma estratégia e atingir o objetivo
    #run.log({"Probabilidade de exploração: ", exploration_prob})

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



