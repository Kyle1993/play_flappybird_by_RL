import game.wrapped_flappy_bird as game
import random
import numpy as np
from itertools import count
import PIL.Image as Image


class flappy_bird_env():
    '''
    NOTE:

        state:
         state is a stack of last frame during frame_skip,
         shape:[frame_skip ,80 ,80],here is [4,80,80]

        action:
         have 2 types:
         [1,0] means do nothing
         [0,1] means flap

        reward:
         reward is the sum of frames during frame_skip
    '''
    def __init__(self):
        self.game = game.GameState()
        self.frame_skip = 4
        self.donothing_action = np.asarray([1,0], dtype=np.int)
        self.flap_action = np.asarray([0,1], dtype=np.int)

    def state_shape(self):
        return (self.frame_skip,80,80)

    def action_shape(self):
        return (2,)

    def init(self):
        state,reward,done = self.step(self.donothing_action)
        return state,reward,done

    # convert to gray, reshape to (80,80)
    def img_process(self,img):
        img = Image.fromarray(img).convert('L').resize((80,80))
        greyimg = np.asarray(img)
        return greyimg

    def step(self,action):
        state = []
        reward = 0
        done = False
        single_state, reward_, done_ = self.game.frame_step(action)
        single_state = self.img_process(single_state)
        reward += reward_
        done = done or done_
        state.append(single_state)

        for frame_id in range(self.frame_skip-1):
            single_state, reward_, done_ = self.game.frame_step(self.donothing_action)
            single_state = self.img_process(single_state)
            reward += reward_
            done = done or done_
            state.append(single_state)

        state = np.asarray(state,dtype=np.float)
        return state,reward,done

if __name__ == "__main__":
    env = flappy_bird_env()
    s,r,d = env.init()

    print(s.shape)
    for step in count(1):
        flag = random.randint(0,10)%2
        if flag==0:
            action = env.donothing_action
        else:
            action = env.flap_action
        s_,r_,d_ = env.step(action)
        print(s_.shape)



