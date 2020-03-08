# open_ai_taxi
open ai's Taxi-v3 toy problem based on the paper, "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition". Using hierarchical reinforcement learning to find optimal strategies to pick up a passenger, get them to a location, and drop them off.


## Best score results
the score results below have been normalized by running 5 trials of 20,000 episodes a piece:

| Env        | Q Learning     | Expecated Sarsa  |  Sarsa
| ---------- |:-------------: | -----:| -----:|
| Taxi-V3    | 8.55           | 7.62  | 7.44  |


## Expected Sarsa
| Learning Rate             |  Reward by Trial |
:-------------------------:|:-------------------------:
![](images/learning_rate_expected_sarsa.png)  |  ![](images/trial_reward_expected_sarsa.png)

### Algorithm Pseudocode

![](images/expected_sarsa.png) 
