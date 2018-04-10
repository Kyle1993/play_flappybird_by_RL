
class DQN_Config():
    def __init__(self):
        self.memory_size = 5000
        self.batch_size = 32
        self.gamma = 0.99
        self.gpu = -1         # GPU id, -1 means use cpu
        self.lr = 1e-4
        self.epsilon_decay = 1e-4
        self.note = 'None'

    def todict(self):
        return vars(self)


class AC_Config():
    def __init__(self):
        self.memory_size = 5000
        self.batch_size = 32
        self.gamma = 0.99
        self.gpu = -1         # GPU id, -1 means use cpu
        self.actor_lr = 1e-5
        self.critic_lr = 1e-5
        self.note = 'None'

    def todict(self):
        return vars(self)
