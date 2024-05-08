
import numpy as np

def utility(rew,alpha_,beta_,lambda_):
    util = np.zeros((len(rew),1))
    for i in range(len(rew)):
        if rew[i] >= 0:
            util[i] = np.power(rew[i],alpha_)
        else:
            util[i] = -lambda_*np.power(abs(rew[i]),beta_)
    return util

def util_plus(rew,alpha_):
    return np.power(abs(rew),alpha_)

def util_minus_abs(rew,beta_,lambda_):
    return lambda_*np.power(abs(rew),beta_)


def weight(prob,gamma):
    if prob < 0:
        prob = 0
    if prob > 1:
        prob = 1
    return (prob**gamma)/(((prob**gamma)+(1-prob)**gamma)**(1/gamma))

def ro(i, probs, utils, gamma):
    ro_val = 0
    p = 0
    if utils[i] >= 0:
        for j in range(len(utils)):
            if utils[j] >= utils[i]:
                p += probs[j]
    if utils[i] < 0:
        for j in range(len(utils)):
            if utils[j] <= utils[i]:
                p += probs[j]
    ro_val = weight(p, gamma) - weight(p-probs[i], gamma)
    return ro_val

def cpt(probs,rewards,alpha_,beta_,lambda_,gamma_pos,gamma_neg):
    cpt_val = 0
    utils = utility(rewards,alpha_,beta_,lambda_)
    for i in range(len(utils)):
        if utils[i] >= 0:
            cpt_val += np.real(ro(i, probs, utils, gamma_pos))*utils[i]
        else:
            cpt_val += np.real(ro(i, probs, utils, gamma_neg))*utils[i]
    return cpt_val




def cpt_estimator(samples, ro, alpha_,beta_,lambda_,gamma_pos,gamma_neg):
    n = len(samples)
    probs = np.zeros((samples.shape[0],1))
    samples = np.sort(samples)
    