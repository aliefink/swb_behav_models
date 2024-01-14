import numpy as np, itertools
from joblib import Parallel, delayed
from scipy.optimize import minimize

# assume RW learning here
def fit_RW(params, choices, rewards, output='nll'):
    ''' 
    Fit the RW model to a single subject's data.
        choices is a np.array with "A" or "B" for each trial
        rewards is a np.array with 1 (reward) or 0 (no reward) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    beta, lr_pos, lr_neg = params 
    nblocks, ntrials = rewards.shape

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:]    = [0, 0]

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
            else:
                c = 1
                choices_A[b, t] = 0

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = rewards[b, t] - ev[b, t, c]

            # determine which learning rate to use
            if pe[b, t] > 0:
                # update EV
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_pos * pe[b, t])
            else:
                # update EV
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_neg * pe[b, t])
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
            
    if output == 'nll':
        return choice_nll
    elif output == 'all':
        subj_dict = {'params'     : np.array([beta, lr_pos, lr_neg]),
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_A'  : choices_A, 
                     'rewards'    : rewards, 
                     'pe'         : pe, 
                     'choice_nll' : choice_nll}
        return subj_dict
    
def softmax(EVs, beta):
    if type(EVs) is list:
        EVs = np.array(EVs)
    return np.exp(beta*EVs) / np.sum(np.exp(beta*EVs))

def minimize_negLL(func_obj, param_values, choices, rewards, param_bounds):
    result = minimize(func_obj, 
                      param_values, 
                      (choices, rewards),
                      bounds=(param_bounds))
    return result

# Example choices and rewards
choices = np.array([['A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'A', 'A', 'A'],
                    ['B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A'],
                    ['B', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A']])
rewards = np.array([[0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1.],
                    [0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1.],
                    [0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.]])

# specify param bounds (beta, alpha_posPE, alpha_negPE)
param_bounds = ((0.01, 20), (0, 1), (0, 1))

res_nll = np.inf # set initial neg LL to be inf

# Define the parameter ranges
beta_range       = np.linspace(.75,7,3)
lr_pos_range     = np.linspace(.25,1,4)
lr_neg_range     = np.linspace(.25,1,4)

# Generate all possible combinations of parameter values
param_combo_guesses = itertools.product(beta_range, lr_pos_range, lr_neg_range)

# Minimize neg LL for all parameter combinations in parallel using joblib
results = Parallel(n_jobs=-1, verbose=5)(delayed(minimize_negLL)(fit_RW, param_values, choices, rewards, param_bounds) for param_values in param_combo_guesses)

# Find the parameter combination with the smallest neg LL
best_result = min(results, key=lambda x: x.fun)
param_fits = best_result.x

# get best model fit info
fit_dict = fit_RW(param_fits, choices, rewards, output='all')
fit_dict['BIC'] = len(param_fits) * np.log(len(choices.shape[0]*choices.shape[1])) + 2*res_nll