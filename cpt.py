import numpy as np


def utility(rew,alpha_,beta_,lambda_):
    util = np.zeros((len(rew),1))
    for i in range(len(rew)):
        if rew[i] >= 0:
            util[i] = np.power(rew[i],alpha_)
        else:
            util[i] = -lambda_*np.power(abs(rew[i]),beta_)
    return util

def weight(prob,gamma):
    if prob < 0:
        prob = 0
    if prob > 1:
        prob = 1
    return (prob**gamma)/(((prob**gamma)+(1-prob)**gamma)**(1/gamma))

def weight_derivative(p,gamma):
    if p < 0:
        p = 0
    if p > 1:
        p = 1
    return gamma*(((1-p)**(gamma-1))*((p*((p**gamma+(1-p)**gamma)**(-1/gamma))**gamma)))/(p*(p**gamma+(1-p)**gamma))

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


def util_plus(rew,alpha_):
    return np.power(abs(rew),alpha_)

def util_minus_abs(rew,beta_,lambda_):
    return lambda_*np.power(abs(rew),beta_)
    
def util_plus_derivative(rew,alpha_):
    eps = 1e-12
    if rew == 0:
        return alpha_*np.power(abs(eps),alpha_-1)
    else:
        return alpha_*np.power(abs(rew),alpha_-1)

def util_minus_derivative(rew,beta_,lambda_):
    eps = 1e-12
    if rew == 0:
        return lambda_*beta_*np.power(abs(eps),beta_-1)
    else:
        return lambda_*beta_*np.power(abs(rew),beta_-1)


def get_next_state(s, action, state_trans):
    probs = state_trans[s,action,:]
    return np.random.choice(len(state_trans[s,action,:]),p=probs)


def cpt_estimate_from_samples(samples, alpha_, beta_, lambda_, gamma_pos, gamma_neg):
    samples = np.array(samples)
    samples = samples.reshape(-1,1)
    idx = np.argsort(samples, axis=0)
    samples = np.take_along_axis(samples, idx, axis=0)
    n_max = len(samples)
    c_plus = 0
    c_minus = 0
    for i in range(1, n_max+1):
        if samples[i-1] >= 0:
            c_plus += util_plus(samples[i-1], alpha_)*(weight((n_max+1-i)/n_max,gamma_pos)-weight((n_max-i)/n_max,gamma_pos))
        else:
            c_minus += util_minus_abs(samples[i-1], beta_, lambda_)*(weight(i/n_max,gamma_neg)-weight((i-1)/n_max,gamma_neg))
    cpt_est = c_plus-c_minus
    return cpt_est

def cpt_estimate_single_agent(n_max, s, a, v, alpha_, beta_, lambda_, gamma_pos, gamma_neg, discount, R, P):
    samples = np.zeros((n_max,1))
    c_plus = 0
    c_minus = 0
    for i in range(n_max):
        s_next = get_next_state(s,a,P)
        r = R[s,a,s_next]
        samples[i] = r+discount*v[s_next]
    idx = np.argsort(samples, axis=0)
    samples = np.take_along_axis(samples, idx, axis=0)
    for i in range(1, n_max+1):
        if samples[i-1] >= 0:
            c_plus += util_plus(samples[i-1], alpha_)*(weight((n_max+1-i)/n_max,gamma_pos)-weight((n_max-i)/n_max,gamma_pos))
        else:
            c_minus += util_minus_abs(samples[i-1], beta_, lambda_)*(weight(i/n_max,gamma_neg)-weight((i-1)/n_max,gamma_neg))
    cpt_est = c_plus-c_minus
    return cpt_est