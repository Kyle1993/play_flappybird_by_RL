# Play Flappy Bird by RL  

this is a implement of DQN and Actor-Critic in game Flappy Bird  
we use pytorch in python3.5

## Requirement
```
pytorch
pygame
PIL
```

    
    
## Usage
the main hyper-parameters are set in the config.py, gpu=-1 means use cpu
#### train
```python
train dqn model:
python3 main_dqn.py

train actor-critic model:
python3 main_ac.py

note:the model will be saved each 100 episodes
```
#### test
```python
train dqn model:
python3 test_dqn.py

train actor-critic model:
python3 test_ac.py

note: in test_*.py, you must set which episode_num of saved model should be load to be tested
```