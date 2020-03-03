from agent import Agent
from monitor import interact
import gym
import numpy as np
# results:
# nA=6, alpha=0.1, epsilon=0.1, gamma=0.9   -> Episode 20000/20000 || Best average reward 5.646
# nA=6, alpha=0.1, epsilon=0.1, gamma=0.75  -> Episode 20000/20000 || Best average reward 5.737
# nA=6, alpha=0.1, epsilon=0.1, gamma=0.6   -> Episode 20000/20000 || Best average reward 5.675

# nA=6, alpha=0.1, epsilon=0.05, gamma=0.75 -> Episode 20000/20000 || Best average reward 7.832
# nA=6, alpha=0.1, epsilon=0.05, gamma=0.9  -> Episode 20000/20000 || Best average reward 7.564

# nA=6, alpha=0.1, epsilon=0.01, gamma=0.9  -> Episode 20000/20000 || Best average reward 8.94
# nA=6, alpha=0.1, epsilon=0.01, gamma=0.75 -> Episode 20000/20000 || Best average reward 9.392
# nA=6, alpha=0.1, epsilon=0.01, gamma=0.6  -> Episode 20000/20000 || Best average reward 8.95
### more realistic for 2 lines above:
# nA=6, alpha=0.1, epsilon=0.01, gamma=0.75 -> 9.075999999999999 over 5 tries

# nA=6, alpha=0.25, epsilon=0.01, gamma=0.9 -> 9.065999999999999 over 5 tries
# nA=6, alpha=0.25, epsilon=0.01, gamma=0.75-> 8.970000000000002 over 5 tries
# nA=6, alpha=0.5, epsilon=0.01, gamma=0.75 -> 9.056000000000001 over 5 tries
# nA=6, alpha=0.75, epsilon=0.01, gamma=0.75-> 8.895999999999999 over 5 tries
# epsilon:
#epsilon = 1 - max(self.epsilon, 1 / self.episode_cntr)
#########################################################################################
# epsilon decay by 25% every 1000 episodes after min epsilon is reached
#nA=6, alpha=0.1, epsilon=0.01, gamma=0.75 -> 9.044 over 5 tries
env = gym.make('Taxi-v2')
agent = Agent(nA=6, alpha=0.1, epsilon=0.01, gamma=0.9)


bestAvgRewards = []
for i in range(5):  
    avg_rewards, best_avg_reward = interact(env, agent)
    bestAvgRewards.append(best_avg_reward)

print('-'*100)
print(f"{ np.mean(bestAvgRewards) } over 5 tries")