import numpy as np

class Agent(object):
    def __init__(self, k, policy, init_type='constant', init_value=0.0, init_std=1.0, gamma=None, bandit=None):
        self.policy = policy
        self.k = k
        self.gamma = gamma
        self.init_type = init_type
        self.init_value = init_value
        self.init_std = init_std
        self.reset(init_type, init_value, init_std)
        self.set_bandit(k, bandit)
        self.set_name()

    def set_name(self):
        if self.init_value != 0.0:
            self.name = str(self.policy)+'_init_'+str(self.init_value)
        else:
            self.name = str(self.policy)
    
    def set_bandit(self, k, bandit):
        if bandit is None:
            self.bandit = GaussianBandit(k)
        else:
            assert bandit.k == k
            self.bandit = bandit
    
    def reset(self, init_type='constant', value=0.0, std=1.0):
        if init_type == 'constant':
            self.value_estimation = value * np.ones(self.k)
        elif init_type == 'random_uniform':
            self.value_estimation = np.random.uniform(0.0, value, self.k)
        elif init_type == 'random_normal':
            self.value_estimation = np.absolute(np.random.normal(value, std, self.k))
        else:
            raise            
        self.step = 1
        self.action_counts = np.array([0]*self.k, dtype=np.float32)
        self.step_size = np.array([0]*self.k, dtype=np.float32)
        self.optimal_counts = []
        self.reward = []
        self.reward_cumulated = [0.0]
        self.optimal_cumulated = [0.0]
    
    def pull(self):
        action = self.choose_action()
        reward, is_optimal = self.get_reward(action)
        self.update(action, reward, is_optimal)
    
    def choose_action(self):
        action = self.policy.get_action(self)
        self.action_counts[action] +=1
        self.update_step_size(action)
        return action
    
    def update_step_size(self, action):
        if self.gamma is not None:
            self.step_size = np.array([self.gamma]*self.k)
        else:
            self.step_size[action] = 1/self.action_counts[action]
            
    def get_reward(self, action):
        return self.bandit.get_reward(action)
    
    def update(self, action, reward, is_optimal):
        self.value_estimation[action] += self.step_size[action]*(reward-self.value_estimation[action])
        self.step +=1
        self.optimal_counts.append(is_optimal)
        self.optimal_cumulated.append(self.optimal_cumulated[-1]+int(is_optimal))
        self.reward.append(reward)
        self.reward_cumulated.append(self.reward_cumulated[-1]+reward)