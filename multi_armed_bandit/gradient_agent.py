from multi_armed_bandit.agent import *

class GradientBanditAgent(Agent):
    def __init__(self, k, policy, init_type='constant', init_value=0.0, init_std=1.0, gamma=None,  baseline_type='avaerage', bandit=None):
        assert gamma is not None
        super(GradientBanditAgent, self).__init__(k, policy, init_type, init_value, init_std, gamma, bandit)
        self.baseline_type = baseline_type
        if baseline_type is None:
            self.name +='_no_baseline'

    def set_name(self):
        self.name = 'Gradient_gamma_'+str(self.gamma)

    def reset(self, init_type='constant', value=0.0, std=1.0):
        super(GradientBanditAgent, self).reset(init_type, value, std)
        self.baseline = 0.0

    def update(self, action, reward, is_optimal):
        if self.baseline_type == 'average':
            self.baseline += self.baseline + (reward-self.baseline)/float(self.step)

        softmax_update = (-1.0)*np.exp(self.value_estimation)/np.sum(np.exp(self.value_estimation))
        softmax_update[action] += 1.0
        self.value_estimation += self.step_size * (reward-self.baseline) * softmax_update

        self.step +=1
        self.optimal_counts.append(is_optimal)
        self.optimal_cumulated.append(self.optimal_cumulated[-1]+int(is_optimal))
        self.reward.append(reward)
        self.reward_cumulated.append(self.reward_cumulated[-1]+reward)