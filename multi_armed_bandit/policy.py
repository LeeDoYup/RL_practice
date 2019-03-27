import numpy as np

class Policy(object):
    def get_action(self, agent):
        pass

class e_Greedy_Policy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __str__(self):
        return "{}_greedy".format(self.epsilon)
        
    def get_action(self, agent):
        if np.random.uniform() > self.epsilon:
            action = np.argmax(agent.value_estimation)
            tie_action_check = np.where(agent.value_estimation==agent.value_estimation[action])[0]
            if len(tie_action_check) == 1:
                return action
            else:
                return np.random.choice(tie_action_check)
        else:
            return np.random.randint(agent.k)

class Greedy_Policy(e_Greedy_Policy):
    def __init__(self):
        super(Greedy_Policy, self).__init__(0)

class Random_Policy(e_Greedy_Policy):
    def __init__(self):
        super(Random_Policy, self).__init__(1)
        
class UCB_Policy(Policy):
    def __init__(self, c):
        self.c = c
    
    def __str__(self):
        return "{}_UCB".format(self.c)
        
    def get_action(self, agent):
        zero_count_actions = np.where(agent.action_counts==0)[0]
        if len(zero_count_actions) > 0:
            return np.random.choice(zero_count_actions)
        confidence = np.sqrt(np.log(agent.step)/agent.action_counts)
        value_ucb = agent.value_estimation + self.c * confidence
        action = np.argmax(value_ucb)
        tie_action_check = np.where(value_ucb==value_ucb[action])[0]
        if len(tie_action_check) < 2:
            return action
        else:
            return np.random.choice(tie_action_check)