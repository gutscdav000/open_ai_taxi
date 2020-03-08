import numpy as np, random
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.1, epsilon=0.1, gamma=0.9, learning_algo="qlearning"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.epsilon = epsilon
        self.original_epsilon = epsilon
        self.gamma = gamma 
        self.episode_cntr = 1
        # test hyper param
        self.epsilon_decay = None
        # learning type 
        self.learning_algo = learning_algo
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment


        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
#         policy_s = np.ones(env.nA) * self.epsilon / env.nA
#         policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / env.nA)
#         return np.random.choice(np.arange(self.nA), p=policy_s)
        
        # calculate epsilon
        epsilon = 1 - max(self.epsilon, 1 / self.episode_cntr)
        
        if epsilon == self.epsilon:
            if self.epsilon_decay is None:
                self.epsilon_decay = 1
            else:
                if self.epsilon_decay % 1000 == 0:
                    self.epsilon *= 0.75
                    
                self.epsilon_decay += 1
    
        if random.uniform(0,1) > epsilon:
            # explore
            return np.random.choice(self.nA)
        else:
            # exploit
            return np.argmax(self.Q[state])
            
        


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.learning_algo == 'qlearning':
            self.q_learn_step(state,action,reward, next_state, done)
        elif self.learning_algo == 'sarsa':
            self.sarsa_step(state, action, reward, next_state, done)
        elif self.learning_algo == 'expected_sarsa':
            self.expected_sarsa_step(state, action, reward, next_state, done)
        else:
            raise Exception("Invalid learning method please choose: qlearning, sarsa, or expected_sarsa")
        
        
    def q_learn_step(self, state,action,reward, next_state, done):
        
        if not done:
            Q_sa_next = np.max(self.Q[next_state])
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * Q_sa_next) - self.Q[state][action]))
        else:
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward - self.Q[state][action]))
            self.episode_cntr += 1
            
    def expected_sarsa_step(self, state,action,reward, next_state, done):
        
        if not done:
            probs = np.ones(self.nA) * self.epsilon /self.nA
            probs[np.argmax(self.Q[state])] +=  1 - self.epsilon # 1 - max(self.epsilon, 1 / self.episode_cntr) #
            next_action = np.random.choice(np.arange(self.nA), p=probs)
            Q_sa_next = np.sum(self.Q[next_state] * probs)
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * Q_sa_next) - self.Q[state][action]))
        else:
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward - self.Q[state][action]))
            self.episode_cntr += 1
            
    def sarsa_step(self, state,action,reward, next_state, done):
        
        if not done:
            policy_s = np.ones(self.nA) * self.epsilon / self.nA
            policy_s[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA) # may be wrong
            next_action = np.random.choice(np.arange(self.nA), p=policy_s)
            Q_sa_next = self.Q[next_state][next_action]
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.gamma * Q_sa_next) - self.Q[state][action]))
        else:
            self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward - self.Q[state][action]))
            self.episode_cntr += 1
    
#     def __str__(self):
#         return f"Agent: nA={self.nA}, alpha={self.alpha}, epsilon={self.original_epsilon}, gamma={self.gamma} "
        