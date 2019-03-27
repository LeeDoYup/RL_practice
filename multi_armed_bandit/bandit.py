import numpy as np

class Bandit(object):
    def __init__(self, k):
        self.k = k
        self.reward_expected = None
        self.optimal_bandit = None
    
    def reset(self):
        self.reward_expected = None
        self.optimal_bandit = None
    
    def get_reward(self, bandit_number):
        return 0, True #reward, optimal or not

class GaussianBandit(Bandit):
    def __init__(self, k, mean=0.0, std=1.0):
        super(GaussianBandit, self).__init__(k)
        self.reset(mean=mean, std=std)
    
    def reset(self, mean=0.0, std=1.0):
        self.mean, self.std = mean, std
        self.reward_expected = np.random.normal(self.mean, self.std, self.k)
        self.reward_std = np.array([1.0]*self.k)
        self.optimal_bandit = np.argmax(self.reward_expected)
    
    def get_bandit_stat(self, bandit_number):
        return (self.reward_expected[bandit_number], self.reward_std[bandit_number])
    
    def get_reward(self, bandit_number, bandit_std=1.0):
        bandit_reward, bandit_std = self.get_bandit_stat(bandit_number)
        return (np.random.normal(bandit_reward, bandit_std), bandit_number==self.optimal_bandit)