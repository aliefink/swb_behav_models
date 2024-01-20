from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from joblib import Parallel, delayed
import random
from sklearn.metrics import r2_score
import random

#contains:
    # fit_swb
    # min_rss_swb
    # fit_base_pt
    # minimize_negll
    # parallel_run_base_pt
    # norm2riskaversion
    # norm2lossaversion
    # norm2invtmp
    # negll_base_pt_pyEM
    # simulate_base_pt
    # run_dual_risk_pt
    # negll_dual_risk_pt
    # simulate_dual_risk_pt
    # fit_dual_risk_pt
    # param_init
    # simulation_norm_gamble_choices
    # simulation_util_norm_gamble_choices
    # get_pt_task_data_mle_emmap
    # get_pt_utils
    # get_glm_data_single_subj
    # get_glm_data_all_subj



############### SWB GLMS ################


#fit swb glm function

def fit_swb(params,df,n_regs,reg_list,lam_method='exp',output='rss'):

    #params is list of lambda estimate + beta estimates 
    betas = params[1:] # list of beta estimates - first index = intercept 
    lam = params[0] # lamda estimate
    K = len(params) # num free params in optimization - used to calculate BIC

    if lam_method == 'exp':
        ls = [1,lam,lam**2] #exponential lambda 
    elif lam_method == 'linear':
        ls = [1,lam,lam*2] #linear lambda 
    else: 
        betas = params # lam not being estimated - not in input params list 
        lam = [1] #lamda
        ls = [1,1,1] #none
        K = K-1 #param count is -1 because lam is not being optimized
        
    #initialize mood estimate equation     
    param_eq = 0

    for n in range(n_regs):
        b = betas[n+1] # first beta value = intercept, so need +1 for weights
        l1 = ls[0] #t-1 decay
        l2 = ls[1] #t-2 decay
        l3 = ls[2] #t-3 decay
        #regressor index for t-1,t-2,t-3
        i1 = (n*3)
        i2 = (n*3)+1
        i3 = (n*3)+2
        #regressor vars to extract from df 
        reg1 = reg_list[i1]
        reg2 = reg_list[i2]
        reg3 = reg_list[i3]
        #regresssor vectors 
        reg1_vec = np.array(df[reg1])
        reg2_vec = np.array(df[reg2])
        reg3_vec = np.array(df[reg3])

        param_eq += (b*l1*reg1_vec) + (b*l2*reg2_vec) + (b*l3*reg3_vec)

    # get the estimated mood rating from the parameter equation (plus intercept!)
    mood_est = [betas[0]]*len(df) + param_eq
    # actual mood obs
    mood_obs = np.array(df['z_rate'])
    #compute the vector of residuals
    mood_residuals = mood_obs - mood_est
    rss = np.sum(mood_residuals**2)

    if output == 'rss':
        return rss
    
    # output for fitting 
    elif output == 'all': 
        subj_dict = {'params'     : params,
                     'reg_list'   : reg_list,
                     'lam_method' : lam_method,
                     'mood_est'   : mood_est,
                     'mood_obs'   : mood_obs,
                     'mood_resid' : mood_residuals,
                     'rss'        : rss,
                     'bic'        : K * np.log(len(mood_residuals)) - 2*np.log((rss/len(mood_residuals))),
                     'aic'        : 2*K + n*np.log(rss/len(mood_residuals))
                     } #https://stats.stackexchange.com/questions/338501/calculating-the-aicc-and-bic-with-rss-instead-of-likelihood

        return subj_dict


#### option to use minimize function to minimize rss instead of least_sq optimization 


def min_rss_swb(subj_df, n_regs, reg_list, param_inits,lam_method='exp'):
    
    # INPUTS:
    # model_df:     model data for subj 
    # subj_id:      subj data for fitting
    # n_regs:       number of task variables in model (ex: ev,cr,rpe = 3 n_regs)
    # reg_list:     list of column names in df as str (should be len n_regs*3, 3 trials for each variable)
    # param_inits:  list of initial parameter value combinations to iterate through - fn will run through optimization for each item in param_guesses [(nparams)(nparams)]
    # lam_method:   calculation of lam param ['exp','linear','none']

    #initialize best result & initial rss value to minimize 
    best_result = []
    rss_optim   = np.inf

    # calculate bounds for each param init 
    n_beta_bounds = n_regs+1
    # remove lam from bounds if none 
    if lam_method == 'none': #remove lam from estimation + bounds input
        bounds = tuple([(-100,100)]*n_beta_bounds)
    else: 
        bounds = tuple([(0.001,1)]+[(-100,100)]*n_beta_bounds)

    for params in param_inits:

        if lam_method == 'none': #remove lam from estimation + bounds input
            params = params[1:]

        #run minimization for each param combo in param_inits
        result = minimize(fit_swb, # objective function
                    params,
                    args=(subj_df,n_regs,reg_list,lam_method), #reg_list should be in long form (3 str per n reg)
                    bounds=bounds) # arguments #method='L-BFGS-B'
        
        #extract rss from result output 
        rss = result.fun #residuals output from best model
        if rss < rss_optim: #goal to minimize cost function, find params that give lowest possible rss
            rss_optim = rss                
            best_result = result 
    
    if rss_optim == np.inf:
        print('No solution for this subject')
        return None
    
    else:
        best_params = best_result.x
        #fit model with optim params
        fit_dict = {}
        fit_dict['best_result'] = best_result
        fit_dict['subj_dict']   = fit_swb(best_params,subj_df,n_regs,reg_list,lam_method,output='all') #run fit function to get all outputs (better than 2 separate fns)

    return fit_dict


############ prospect theory models ##############

##### MLE parameter estimation for base prospect theory

def fit_base_pt(params, subj_df, prior=None, output='mle'):

    risk_aversion, loss_aversion, inverse_temp = params

    if output == 'npl':
        risk_aversion = norm2riskaversion(risk_aversion) #transform parameter from gaussian space back into native model space using parameter-specific sigmoid function
        risk_aversion_bounds = [0.00001, 2] #set upper and lower bounds
        if risk_aversion< min(risk_aversion_bounds) or risk_aversion> max(risk_aversion_bounds): #prevent estimation from parameter values outside of bounds 
            return 10000000
        
        loss_aversion = norm2lossaversion(loss_aversion) #transform parameter from gaussian space back into native model space using parameter-specific sigmoid function
        loss_aversion_bounds = [0.00001, 6] #set upper and lower bounds
        if loss_aversion< min(loss_aversion_bounds) or loss_aversion> max(loss_aversion_bounds):  #prevent estimation from parameter values outside of bounds 
            return 10000000
        
        inverse_temp = norm2invtmp(inverse_temp) #transform parameter from gaussian space back into native model space using parameter-specific sigmoid function
        this_beta_bounds = [0.00001, 8]  #set upper and lower bounds
        if inverse_temp < min(this_beta_bounds) or inverse_temp > max(this_beta_bounds):  #prevent estimation from parameter values outside of bounds 
            return 10000000

    #Initialize choice probability vector to calculate negative log likelihood
        # actual subj choice info   
    choice_list = []
    choice_prob_list = []
        # predicted subj choice info 
    choice_pred_list = []
    choice_pred_prob = []

    #Initialize empty data vectors to return all relevant data if output = 'all'
    tr          = []
    trial_list  = []
    util_g      = []
    util_s      = []
    choice_util = []
    p_g         = []
    p_s         = []
    safe        = []
    high        = []
    low         = []
    w_safe      = []
    w_high      = []
    w_low       = []
    

    for trial in range(len(subj_df)):

        trial_info = subj_df.iloc[trial]
        trial_type = trial_info['TrialType']
        choice = trial_info['GambleChoice']
        high_bet = trial_info['HighBet']
        low_bet = trial_info['LowBet']
        safe_bet = trial_info['SafeBet']


        #store trial info 
        choice_list.append(choice)
        tr.append(trial)
        trial_list.append(trial_type)
        high.append(high_bet)
        low.append(low_bet)
        safe.append(safe_bet)

        ##### Utility calculations #####

        # transform to high bet value to utility (gamble)
        if high_bet > 0: #mix or gain trials
            weighted_high_bet = 0.5 * ((high_bet)**risk_aversion)
        else: #loss trials
            weighted_high_bet = 0 
        
        w_high.append(weighted_high_bet)

        # transform to low bet value to utility (gamble)
        if low_bet < 0: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * ((-low_bet)**risk_aversion)
            
        else: #gain trials
            weighted_low_bet = 0 
        
        w_low.append(weighted_low_bet)
        
        util_gamble = weighted_high_bet + weighted_low_bet
        util_g.append(util_gamble)
    

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion
        else: #loss trials
            util_safe = -loss_aversion * ((-safe_bet)**risk_aversion)

        w_safe.append(util_safe)
        util_s.append(util_safe)


        ##### Choice probability calculation #####

        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        
        p_g.append(p_gamble)
        p_s.append(p_safe)

        # append probability of chosen options
        if choice == 'gamble':
            choice_prob_list.append(p_gamble)
            choice_util.append(util_gamble)

        elif choice == 'safe':
            choice_prob_list.append(p_safe)
            choice_util.append(p_safe)
        
        
        #getting stochastic predictions of model 
        choice_pred = random.choices(['gamble','safe'],weights=[p_gamble,p_safe])[0]
        choice_pred_list.append(choice_pred)

        if choice_pred == 'gamble':
            choice_pred_prob.append(p_gamble)
        else:
            choice_pred_prob.append(p_safe)

    # calculate negative log likelihood of choice probabilities 
            
    negll = -np.sum(np.log(choice_prob_list))
    
    if np.isnan(negll):
        negll = np.inf
    
    # output for MLE optimization
    if output == 'mle': 
        return negll
    
    # output for fitting 
    elif output == 'all': 
        subj_dict = {'params'         : [risk_aversion, loss_aversion, inverse_temp],
                     'tr'             : tr,
                     'TrialType'      : trial_list,
                     'GambleChoice'   : choice_list,
                     'ChoiceProb'     : choice_prob_list,
                     'ChoiceUtil'     : choice_util,
                     'ChoicePred'     : choice_pred_list,
                     'ChoicePredProb' : choice_pred_prob,
                     'util_gamble'    : util_g,
                     'util_safe'      : util_s, 
                     'p_gamble'       : p_g,
                     'p_safe'         : p_s,
                     'HighBet'        : high,
                     'LowBet'         : low,
                     'SafeBet'        : safe,
                     'WeightedHigh'   : w_high,
                     'WeightedLow'    : w_low,
                     'WeightedSafe'   : w_safe,
                     'negll'          : negll,
                     'BIC'            : len(params) * np.log(150) + 2*negll,
                     'AIC'            : 2*len(params) + 2*negll}
        return subj_dict
    
    # output for EM MAP optimization
    elif output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval): 
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
    
    
##### MLE parameter estimation for base prospect theory

def minimize_negll(func_obj, param_values, df, param_bounds):
    # minimize negll via MLE via gradient descent

    result = minimize(func_obj, 
                      param_values, 
                      df,
                      bounds=param_bounds)
    return result
    


def parallel_run_base_pt(min_fn, fit_fn,param_combo_guesses,param_bounds,subj_df,n_jobs=-2):
    '''
    Maximum likelihood estimation with parallel processing 

    Inputs:
        - min_fn: minimization function 
        - fit_fn: model fitting function (should return negll only)
        - param_combo_guesses: grid of initial param values for parallel min_fn runs 
        - param_bounds: min/max bounds for params in this format: (0,5),(0,5),(0,10)
        - subj_df: pandas df of subj task data
    
    Returns:
        - fit_dict: output of fit_fn

    '''

    
    ##### Minimize negll via parallel mle

    # Parallel fn :
        # Basic syntax Parallel(n_jobs,verbose) ( delayed(optim_fn)(optim_fun inputs) loop for parallel fn inputs )
        # requires Parallel & delayed from joblib
        # n_jobs=-2 - num cpus used, -1 for all, -2 for all but one, +num for specific num
        # verbose default is none, higher than 10 will give all
        # delayed() = hold memory for function to run in parallel
        # optim_fn = minimization fn
        # ()() = inputs for optim_fn in delay - negll fn, params, data, bounds
        # (()()____): iterations of initial param values 
    
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(min_fn)(fit_fn, param_values, (subj_df), param_bounds) for param_values in param_combo_guesses)

    # determine optimal parameter combination from negll
    fit_dict = {}
    best_result = min(results, key=lambda x: x.fun) # use lambda function to get negll from each run in results (lambda args: expression) 
    param_fits = best_result.x
    fit_dict['best_result'] = best_result
    # run fit_fn with param_fits get best model fit info ### implement this with GLMs!
    fit_dict['subj_dict'] = fit_fn(param_fits, subj_df, output='all')
    
    
    return fit_dict


##### EM MAP parameter estimation for base prospect theory

#variable transformation functions
def norm2riskaversion(aversion_param):
    return 2 / (1 + np.exp(-aversion_param))
def norm2lossaversion(aversion_param):
    return 6 / (1 + np.exp(-aversion_param))
def norm2invtmp(invtemp):
    return 10 / (1 + np.exp(-invtemp))

#negll calculation and fit fn (update MLE fns to have two output options!)
def negll_base_pt_pyEM(params, subj_df, prior=None, output='npl'):

    risk_aversion, loss_aversion, inverse_temp = params
    
    risk_aversion = norm2riskaversion(risk_aversion) #transform parameter from gaussian space back into native model space using parameter-specific sigmoid function
    risk_aversion_bounds = [0.00001, 2] #set upper and lower bounds
    if risk_aversion< min(risk_aversion_bounds) or risk_aversion> max(risk_aversion_bounds): #prevent estimation from parameter values outside of bounds 
        return 10000000
    
    loss_aversion = norm2lossaversion(loss_aversion) #transform parameter from gaussian space back into native model space using parameter-specific sigmoid function
    loss_aversion_bounds = [0.00001, 6] #set upper and lower bounds
    if loss_aversion< min(loss_aversion_bounds) or loss_aversion> max(loss_aversion_bounds):  #prevent estimation from parameter values outside of bounds 
        return 10000000
    
    inverse_temp = norm2invtmp(inverse_temp) #transform parameter from gaussian space back into native model space using parameter-specific sigmoid function
    this_beta_bounds = [0.00001, 10]  #set upper and lower bounds
    if inverse_temp < min(this_beta_bounds) or inverse_temp > max(this_beta_bounds):  #prevent estimation from parameter values outside of bounds 
        return 10000000

    #Initialize choice probability vector to calculate negative log likelihood
    choice_prob_list = []
    choice_list = []

    #Initialize empty data vectors to return all relevant data if output = 'all'
    tr          = []
    trial_list  = []
    util_g      = []
    util_s      = []
    choice_util = []
    p_g         = []
    p_s         = []
    safe        = []
    high        = []
    low         = []
    w_safe      = []
    w_high      = []
    w_low       = []

    for trial in range(len(subj_df)):

        trial_info = subj_df.iloc[trial]
        trial_type = trial_info['TrialType']
        choice = trial_info['GambleChoice']
        high_bet = trial_info['HighBet']
        low_bet = trial_info['LowBet']
        safe_bet = trial_info['SafeBet']


        #store trial info 
        choice_list.append(choice)
        tr.append(trial)
        trial_list.append(trial_type)
        high.append(high_bet)
        low.append(low_bet)
        safe.append(safe_bet)

        ##### Utility calculations #####

        # transform to high bet value to utility (gamble)
        if high_bet > 0: #mix or gain trials
            weighted_high_bet = 0.5 * ((high_bet)**risk_aversion)
        else: #loss trials
            weighted_high_bet = 0 
        
        w_high.append(weighted_high_bet)

        # transform to low bet value to utility (gamble)
        if low_bet < 0: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * ((-low_bet)**risk_aversion)
            
        else: #gain trials
            weighted_low_bet = 0 
        
        w_low.append(weighted_low_bet)
        
        util_gamble = weighted_high_bet + weighted_low_bet
        util_g.append(util_gamble)
      

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion
        else: #loss trials
            util_safe = -loss_aversion * ((-safe_bet)**risk_aversion)

        w_safe.append(util_safe)
        util_s.append(util_safe)


        ##### Choice probability calculation #####

        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        # p_safe = 1-p_gamble
        p_g.append(p_gamble)
        p_s.append(p_safe)

        # append probability of chosen options
        if choice == 'gamble':
            choice_prob_list.append(p_gamble)
            choice_util.append(util_gamble)

        elif choice == 'safe':
            choice_prob_list.append(p_safe)
            choice_util.append(p_safe)

    # calculate negative log likelihood of choice probabilities 
            
    negll = -np.sum(np.log(choice_prob_list))
    
    if np.isnan(negll):
        negll = np.inf
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval): 
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all': #WHATEVER YOU WANT TO OUTPUT IF NOT OPTIMIZING
        subj_dict = {'params'      : [risk_aversion, loss_aversion, inverse_temp],
                     'tr'          : tr,
                     'TrialType'   : trial_list,
                     'GambleChoice':choice_list,
                     'ChoiceProb'  : choice_prob_list,
                     'ChoiceUtil'  : choice_util,
                     'util_gamble' : util_g,
                     'util_safe'   : util_s, 
                     'p_gamble'    : p_g,
                     'p_safe'      : p_s,
                     'HighBet'     : high,
                     'LowBet'      : low,
                     'SafeBet'     : safe,
                     'WeightedHigh': w_high,
                     'WeightedLow' : w_low,
                     'WeightedSafe': w_safe,
                     'negll'       : negll,
                     'bic'         : len(params) * np.log(150) + 2*negll}
        
        return subj_dict

### base prospect theory model as a simulator for parameter recovery

def simulate_base_pt(params,trials):
    #inputs: 
    #params - risk, loss, temp
    #trials - number of trials for simulation (for EMU SWB always 150)
    risk_aversion, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    tr = []
    trial_list = []
    choice_prob = []
    choice_pred = []
    util_g = []
    util_s = []
    choice_util = []
    p_g = []
    p_s = []
    safe = []
    high = []
    low = []

    #load task code master df 
    swb_trial_master = pd.read_csv('/sc/arion/projects/guLab/Alie/SWB/swb_behav_models/data/swb_trial_master.csv')

    task = swb_trial_master.sample(frac = 1) #randomize task order 

    #loop through trials
    for trial in range(len(task)):

        trial_type = task.TrialType.iloc[trial]
        safe_bet = task.SafeBet.iloc[trial]
        high_bet = task.HighBet.iloc[trial]
        low_bet = task.LowBet.iloc[trial]
        trial_list.append(trial_type)

        safe.append(safe_bet)
        high.append(high_bet)
        low.append(low_bet)


        # transform to high bet value to utility (gamble)
        if high_bet > 0: #mix or gain trials
            weighted_high_bet = 0.5 * ((high_bet)**risk_aversion)
        else: #loss trials
            weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
        
        # transform to low bet value to utility (gamble)
        if low_bet >= 0: #gain trials
            weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
        else: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * ((-low_bet)**risk_aversion)
        
        util_gamble = weighted_high_bet + weighted_low_bet
    

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion
        else: #loss trials
            util_safe = -loss_aversion * ((-safe_bet)**risk_aversion)
        
        # utility options for calculating EV - utils separate, ug - us to combine or Uchosen - Unchosen (will differ by participant) 
        #inverse temp < 1 more exporatory, > 1 more exploitative
        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        

        util_g.append(util_gamble)
        util_s.append(util_safe)
        p_g.append(p_gamble)
        p_s.append(p_safe)

        choice = random.choices(['gamble','safe'],weights=[p_gamble,p_safe])[0]
        choice_pred.append(choice)

        if choice == 'gamble':
            choice_prob.append(p_gamble)
            choice_util.append(util_gamble)
        else:
            choice_prob.append(p_safe)
            choice_util.append(util_safe)

        tr.append(trial)



    data = {'tr':tr,'TrialType':trial_list,'GambleChoice':choice_pred,'ChoiceProb':choice_prob, 'ChoiceUtil':choice_util,
                       'util_gamble':util_g,'util_safe':util_s,'p_gamble':p_g,'p_safe':p_s,'SafeBet':safe,'HighBet':high,'LowBet':low}
    DF = pd.DataFrame(data)
    
    return DF


###### dual risk prospect theory model

def run_dual_risk_pt(subj_df,risk_gain_inits,risk_loss_inits,loss_inits,temp_inits,bounds):
    # gradient descent to minimize neg LL

    subj_df = (subj_df)
    res_nll = np.inf


    # guess several different starting points for rho
    for risk_gain_guess in risk_gain_inits:
        for risk_loss_guess in risk_loss_inits:
            for loss_guess in loss_inits:
                for temp_guess in temp_inits:
            
                    # guesses for alpha, theta will change on each loop
                    init_guess = (risk_gain_guess, risk_loss_guess, loss_guess, temp_guess)
                    
                    # minimize neg LL
                    result = minimize(negll_dual_risk_pt, 
                                    x0 = init_guess, 
                                    args = subj_df,
                                    method='L-BFGS-B',
                                    bounds=bounds) #should probably not be hard coded..
                    
                    # if current negLL is smaller than the last negLL,
                    # then store current data
                    if result.fun < res_nll:
                        res_nll = result.fun
                        param_fits = result.x
                        risk_gain_aversion, risk_loss_aversion,loss_aversion, inverse_temp = param_fits
                        optim_vars = init_guess
                    

    if res_nll == np.inf:
        print('No solution for this patient')
        risk_gain_aversion=0
        risk_loss_aversion=0
        loss_aversion=0
        inverse_temp=0
        BIC=0
        optim_vars=0
    else:
        BIC = len(init_guess) * np.log(len(subj_df)) + 2*res_nll
    
    return risk_gain_aversion, risk_loss_aversion, loss_aversion, inverse_temp, BIC, optim_vars


def negll_dual_risk_pt(params, subj_df):
    risk_aversion_gain, risk_aversion_loss, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    choiceprob_list = []

    #loop through trials
    for trial in range(len(subj_df)):

        # get relevant trial info
        trial_info = subj_df.iloc[trial]
        high_bet = trial_info['high_bet']
        low_bet = trial_info['low_bet']
        safe_bet = trial_info['safe_bet']
        trial_type = trial_info['type']
        choice = trial_info['choice_pred']
        #outcome = trial_info['Profit']

        # transform to high bet value to utility (gamble)
        if high_bet > 0: #mix or gain trials
            weighted_high_bet = 0.5 * ((high_bet)**risk_aversion_gain) #different risk aversion parameter for gain values
        else: #loss trials
            weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
        
        # transform to low bet value to utility (gamble)
        if low_bet >= 0: #gain trials
            weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
        else: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * ((-low_bet)**risk_aversion_loss) #different risk aversion parameter for loss values
        
        util_gamble = weighted_high_bet + weighted_low_bet
    

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion_gain #gain risk aversion parameter
        else: #loss trials
            util_safe = -loss_aversion * ((-safe_bet)**risk_aversion_loss) #loss risk aversion parameter
        
        # utility options for calculating EV - utils separate, ug - us to combine or Uchosen - Unchosen (will differ by participant) 
        #inverse temp < 1 more exporatory, > 1 more exploitative
        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        

        # if np.isnan(p_gamble): #if utility is too large, probabiities will come out nan and 0
        #     p_gamble = 0.99
        #     p_safe = 0.01
        # if np.isnan(p_safe):
        #     p_safe = 0.99
        #     p_gamble = 0.01

        # append probability of chosen options
        if choice == 'gamble':
            choiceprob_list.append(p_gamble) 
        elif choice == 'safe':
            choiceprob_list.append(p_safe)

    # compute the neg LL of choice probabilities across the entire task
    negLL = -np.sum(np.log(choiceprob_list))
    
    if np.isnan(negLL):
        return np.inf
    else:
        return negLL
    
def simulate_dual_risk_pt(params,rep,trials):
    risk_aversion_gain, risk_aversion_loss, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    rep_list = []
    tr = []
    trial_list = []
    choice_prob = []
    choice_pred = []
    util_g = []
    util_s = []
    p_g = []
    p_s = []
    safe = []
    high = []
    low = []
    

    for rep in range(rep):
        types = ['mix','gain','loss']
        trial_types = random.choices(types,k=trials)

        #loop through trials
        for trial in range(trials):
            type = trial_types[trial]
            trial_list.append(type)
            if type == 'mix':
                safe_bet = 0
                low_bet = round(random.uniform(-1.5,-0.3),2) 
                high_bet = round(random.uniform(3.0,0.06),2) 
            elif type == 'gain':
                high_bet = round(random.uniform(0.34,3.0),2) 
                safe_bet = round(random.uniform(0.2,0.6),2) #need to constrain to always be less than high bet!!!
                while safe_bet >= high_bet:
                    safe_bet = round(random.uniform(0.2,0.6),2) #need to constrain to always be less than high bet!!!
                low_bet = 0
            elif type == 'loss':
                low_bet = round(random.uniform(-3.0,-0.34),2)
                safe_bet = round(random.uniform(-0.2,-0.6),2)
                while safe_bet <= low_bet:
                    safe_bet = round(random.uniform(-0.2,-0.6),2) #need to constrain to always be greater than low bet!!!
                high_bet = 0
            
            safe.append(safe_bet)
            high.append(high_bet)
            low.append(low_bet)


            # transform to high bet value to utility (gamble)
            if high_bet > 0: #mix or gain trials
                weighted_high_bet = 0.5 * ((high_bet)**risk_aversion_gain) #different risk aversion parameter for gain values
            else: #loss trials
                weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
            
            # transform to low bet value to utility (gamble)
            if low_bet >= 0: #gain trials
                weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
            else: #loss and mix trials
                weighted_low_bet = -0.5 * loss_aversion * ((-low_bet)**risk_aversion_loss) #different risk aversion parameter for loss values
            
            util_gamble = weighted_high_bet + weighted_low_bet
        

            # transform safe bet value to utility (safe)
            if safe_bet >= 0: #gain or mix trials
                util_safe = (safe_bet)**risk_aversion_gain #gain risk aversion parameter
            else: #loss trials
                util_safe = -loss_aversion * ((-safe_bet)**risk_aversion_loss) #loss risk aversion parameter
            
            # utility options for calculating EV - utils separate, ug - us to combine or Uchosen - Unchosen (will differ by participant) 
            #inverse temp < 1 more exporatory, > 1 more exploitative
            # convert EV to choice probabilities via softmax
            p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
            p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
            

            # if np.isnan(p_gamble):
            #     p_gamble = 0.99
            #     p_safe = 0.01
            # if np.isnan(p_safe):
            #     p_safe = 0.99
            #     p_gamble = 0.01

            util_g.append(util_gamble)
            util_s.append(util_safe)
            p_g.append(p_gamble)
            p_s.append(p_safe)



            choice = random.choices(['gamble','safe'],weights=[p_gamble,p_safe])[0]

            choice_pred.append(choice)


            if choice == 'gamble':
                choice_prob.append(p_gamble)
            else:
                choice_prob.append(p_safe)

            tr.append(trial)

            rep_list.append(rep)
        

    data = {'rep':rep_list,'tr':tr,'type':trial_list,'choice_pred':choice_pred,'choice_prob':choice_prob,
                       'util_gamble':util_g,'util_safe':util_s,'p_gamble':p_g,'p_safe':p_s,'safe_bet':safe,'high_bet':high,'low_bet':low}
    DF = pd.DataFrame(data)
    

    return DF


def param_init(n_values, n_iter, upper_bound, lower_bound, method, beta_shape=0):
    #inputs:
        #n_values: how many parameter values needed
        #n_iter: how many rounds of initialization; if method = 'rand' will return dict with array n_iter x n_values; if method = 'mc_grid' will return dict with grid n_iter x n_iter for each param (n_values)
        #upper_bound: max possible parameter value
        #lower_bound: min possible parameter value
        #beta_shape: [a,b] alpha and beta values for beta distribution
        #method: rand, mc_grid

    #outputs:
        #param dict - dictionary of param_id: init values (number of values = n_values)

    if method == 'rand':
        param_array = np.zeros(shape=(n_iter,n_values))
        for iter in range(n_iter):
            for val in range(n_values):
                if iter%2==0:
                    param_array[iter,val] = random.uniform(lower_bound,lower_bound+1) #hacky way to bias random initialization to have more numbers between 0-1
                if iter%2!=0:
                    param_array[iter,val] = random.uniform(lower_bound,upper_bound)
    
    elif method == 'beta':
        a = beta_shape[0]
        b = beta_shape[1]
        N = n_iter

        param_array = (upper_bound - lower_bound) * np.random.beta(a, b, N) + lower_bound

    
    #elif method == 'mc_grid':
        #to do - make large parameter grid for monte carlo method for paramter initialization
        #arianna matlab code:             # %% Free param starting points
            # free0 = cell(nM, 1); % nM is number of models #collection of all start points that you're about to generate

            # for m = 1:nM #can do for bunch of models at once but doesn't need to be loop
            #     numStartingPoints = 10000; % Number of starting points to sample using Monte Carlo method #40k here
            #     free0{m} = zeros(numStartingPoints, nX); #empty matrix for params
            #     for s = 1:numStartingPoints 
            #         % Generate random starting point for free parameters
            #         free0{m}(s, xIndex{m}) = rand(1, length(xIndex{m})); #this is where you generate random starting point - create array for starting points and then outside of this loop through it with model - arianna constrains later! 
            #     end
            # end
    
    
    return param_array


def simulation_norm_gamble_choices(df): #to-do input column names to make this robust to standard + util 
    
    #df is task data for a single subject
    loss_df = df[df.TrialType == 'loss']
    mix_df = df[df.TrialType == 'mix']
    gain_df = df[df.TrialType == 'gain']

    #loss
    loss_dict = {}
    loss_norm = -((loss_df['LowBet'] + loss_df['HighBet'])/2)/loss_df['SafeBet']
    loss_quant = np.quantile(loss_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    loss_x_axis = [np.mean(loss_quant[i:i+2],dtype=np.float64) for i in range(5)]
    loss_dec = loss_df['ChoicePred'].replace(['gamble','safe'],[1,0])
    loss_zip = list(zip(loss_norm,loss_dec))
    loss_dict['loss_norm_evs'] = np.array(loss_norm)
    loss_dict['loss_choices'] = np.array(loss_dec)
    loss_dict['loss_x_axis'] = loss_x_axis
    loss_norm_range = []
    loss_choice_props = []
    for r in range(5):
        loss_ev_range = np.array([loss_quant[r],loss_quant[r+1]])
        loss_gamble_count = [z[1] for z in loss_zip if z[0] >= loss_ev_range[0] and z[0] <= loss_ev_range[1]]
        loss_ev_num = sum(loss_gamble_count)
        loss_ev_prop = loss_ev_num/len(loss_gamble_count)
        loss_norm_range.append(loss_ev_range)
        loss_choice_props.append(loss_ev_prop)
    loss_dict['loss_norm_range'] = loss_norm_range
    loss_dict['loss_choice_props'] = loss_choice_props
    
    #mix
    mix_dict = {}
    mix_norm = ((mix_df['LowBet'] + mix_df['HighBet'])/2) #can't divide by zero
    mix_quant = np.quantile(mix_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    mix_x_axis = [np.mean(mix_quant[i:i+2],dtype=np.float64) for i in range(5)]
    mix_dec = mix_df['ChoicePred'].replace(['gamble','safe'],[1,0])
    mix_zip = list(zip(mix_norm,mix_dec))
    mix_dict['mix_norm_evs'] = np.array(mix_norm)
    mix_dict['mix_choices'] = np.array(mix_dec)
    mix_dict['mix_x_axis'] = mix_x_axis
    mix_norm_range = []
    mix_choice_props = []
    for r in range(5):
        mix_ev_range = np.array([mix_quant[r],mix_quant[r+1]])
        mix_gamble_count = [z[1] for z in mix_zip if z[0] >= mix_ev_range[0] and z[0] <= mix_ev_range[1]]
        mix_ev_num = sum(mix_gamble_count)
        mix_ev_prop = mix_ev_num/len(mix_gamble_count)
        mix_norm_range.append(mix_ev_range)
        mix_choice_props.append(mix_ev_prop)
    mix_dict['mix_norm_range'] = mix_norm_range
    mix_dict['mix_choice_props'] = mix_choice_props

    
    #gain
    gain_dict = {}
    gain_norm = ((gain_df['LowBet'] + gain_df['HighBet'])/2)/gain_df['SafeBet']
    gain_quant = np.quantile(gain_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    gain_x_axis = [np.mean(gain_quant[i:i+2],dtype=np.float64) for i in range(5)]
    gain_dec = gain_df['ChoicePred'].replace(['gamble','safe'],[1,0])
    gain_zip = list(zip(gain_norm,gain_dec))
    gain_dict['gain_norm_evs'] = np.array(gain_norm)
    gain_dict['gain_choices'] = np.array(gain_dec)
    gain_dict['gain_x_axis'] = gain_x_axis
    gain_norm_range = []
    gain_choice_props = []
    for r in range(5):
        gain_ev_range = np.array([gain_quant[r],gain_quant[r+1]])
        gain_gamble_count = [z[1] for z in gain_zip if z[0] >= gain_ev_range[0] and z[0] <= gain_ev_range[1]]
        gain_ev_num = sum(gain_gamble_count)
        gain_ev_prop = gain_ev_num/len(gain_gamble_count)
        gain_norm_range.append(gain_ev_range)
        gain_choice_props.append(gain_ev_prop)
    gain_dict['gain_norm_range'] = gain_norm_range
    gain_dict['gain_choice_props'] = gain_choice_props
    
    return loss_dict, mix_dict, gain_dict


def simulation_util_norm_gamble_choices(df):
    
    #df is task data for a single subject
    loss_df = df[df.TrialType == 'loss']
    mix_df = df[df.TrialType == 'mix']
    gain_df = df[df.TrialType == 'gain']

    #loss
    loss_dict = {}
    #loss_norm = -((loss_df['low_bet'] + loss_df['high_bet'])/2)/loss_df['safe_bet']
    #replacing EV with utility
    loss_norm = -loss_df['util_gamble']/loss_df['util_safe'] #util_g/util_s
    loss_quant = np.quantile(loss_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    loss_x_axis = [np.mean(loss_quant[i:i+2],dtype=np.float64) for i in range(5)]
    loss_dec = loss_df['ChoicePred'].replace(['gamble','safe'],[1,0])
    loss_zip = list(zip(loss_norm,loss_dec))
    loss_dict['loss_norm_evs'] = np.array(loss_norm)
    loss_dict['loss_choices'] = np.array(loss_dec)
    loss_dict['loss_x_axis'] = loss_x_axis
    loss_norm_range = []
    loss_choice_props = []
    for r in range(5):
        loss_ev_range = np.array([loss_quant[r],loss_quant[r+1]])
        loss_gamble_count = [z[1] for z in loss_zip if z[0] >= loss_ev_range[0] and z[0] <= loss_ev_range[1]]
        loss_ev_num = sum(loss_gamble_count)
        loss_ev_prop = loss_ev_num/len(loss_gamble_count)
        loss_norm_range.append(loss_ev_range)
        loss_choice_props.append(loss_ev_prop)
    loss_dict['loss_norm_range'] = loss_norm_range
    loss_dict['loss_choice_props'] = loss_choice_props
    
    #mix
    mix_dict = {}
    #mix_norm = ((mix_df['low_bet'] + mix_df['high_bet'])/2) #can't divide by zero
    #replacing EV with utility
    mix_norm = mix_df['util_gamble'] #can't divide by zero, util gamble has weighted high and low bet already
    mix_quant = np.quantile(mix_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    mix_x_axis = [np.mean(mix_quant[i:i+2],dtype=np.float64) for i in range(5)]
    mix_dec = mix_df['ChoicePred'].replace(['gamble','safe'],[1,0])
    mix_zip = list(zip(mix_norm,mix_dec))
    mix_dict['mix_norm_evs'] = np.array(mix_norm)
    mix_dict['mix_choices'] = np.array(mix_dec)
    mix_dict['mix_x_axis'] = mix_x_axis
    mix_norm_range = []
    mix_choice_props = []
    for r in range(5):
        mix_ev_range = np.array([mix_quant[r],mix_quant[r+1]])
        mix_gamble_count = [z[1] for z in mix_zip if z[0] >= mix_ev_range[0] and z[0] <= mix_ev_range[1]]
        mix_ev_num = sum(mix_gamble_count)
        mix_ev_prop = mix_ev_num/len(mix_gamble_count)
        mix_norm_range.append(mix_ev_range)
        mix_choice_props.append(mix_ev_prop)
    mix_dict['mix_norm_range'] = mix_norm_range
    mix_dict['mix_choice_props'] = mix_choice_props

    
    #gain
    gain_dict = {}
    #gain_norm = ((gain_df['low_bet'] + gain_df['high_bet'])/2)/gain_df['safe_bet']
    #replacing EV with utility
    gain_norm = gain_df['util_gamble']/gain_df['util_safe'] #util_g/util_s
    gain_quant = np.quantile(gain_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    gain_x_axis = [np.mean(gain_quant[i:i+2],dtype=np.float64) for i in range(5)]
    gain_dec = gain_df['ChoicePred'].replace(['gamble','safe'],[1,0])
    gain_zip = list(zip(gain_norm,gain_dec))
    gain_dict['gain_norm_evs'] = np.array(gain_norm)
    gain_dict['gain_choices'] = np.array(gain_dec)
    gain_dict['gain_x_axis'] = gain_x_axis
    gain_norm_range = []
    gain_choice_props = []
    for r in range(5):
        gain_ev_range = np.array([gain_quant[r],gain_quant[r+1]])
        gain_gamble_count = [z[1] for z in gain_zip if z[0] >= gain_ev_range[0] and z[0] <= gain_ev_range[1]]
        gain_ev_num = sum(gain_gamble_count)
        gain_ev_prop = gain_ev_num/len(gain_gamble_count)
        gain_norm_range.append(gain_ev_range)
        gain_choice_props.append(gain_ev_prop)
    gain_dict['gain_norm_range'] = gain_norm_range
    gain_dict['gain_choice_props'] = gain_choice_props
    
    return loss_dict, mix_dict, gain_dict


def get_pt_task_data_mle_emmap(subj_ids,behav_dir,pt_mle_fits,pt_emmap_fits):

    # load task data for each subj, calculate pt params, add all params to subj task_data & save as pt_task_data
    # just add - utilsafe, utilgamble, utilchoice, wsafe, whigh, wlow, psafe, pgamble x 2 for mle & emmap
    # need to calculate then add - utiltCPE, utildCPE, utiltRegret, utildRegret, utiltRelief, utildRelief, utilRPe, utilpRPE,utilnRPE x 2 for mle & emmap
    
    pt_task_dfs = []

    for subj_id in subj_ids:
        #load preprocessed task data with model-free params
        subj_df = pd.read_csv(f'{behav_dir}{subj_id}_task_data')
        #load pt fits data dicts 
            #load mle subj data
        mle_data = pt_mle_fits[subj_id]['subj_dict']
            #emmap param calculations
        emmap_data = pt_emmap_fits[subj_id]

        #add already calculated mle params to subj df
        subj_df['util_safe_mle']   = mle_data['util_safe']
        subj_df['util_gamble_mle'] = mle_data['util_gamble']
        # subj_df['util_choice_mle'] = mle_data['ChoiceUtil'] #only length 149?
        subj_df['wSafe_mle']       = mle_data['WeightedSafe']
        subj_df['wHigh_mle']       = mle_data['WeightedHigh']
        subj_df['wLow_mle']        = mle_data['WeightedLow']
        subj_df['p_safe_mle']      = mle_data['p_safe']
        subj_df['p_gamble_mle']    = mle_data['p_gamble']

        
        #already calculated emmap params to subj_df 
        subj_df['util_safe_emmap']   = emmap_data['util_safe']
        subj_df['util_gamble_emmap'] = emmap_data['util_gamble']
        # subj_df['util_choice_emmap'] = emmap_data['ChoiceUtil'] #only length 149?
        subj_df['wSafe_emmap']       = emmap_data['WeightedSafe']
        subj_df['wHigh_emmap']       = emmap_data['WeightedHigh']
        subj_df['wLow_emmap']        = emmap_data['WeightedLow']
        subj_df['p_safe_emmap']      = emmap_data['p_safe']
        subj_df['p_gamble_emmap']    = emmap_data['p_gamble']

        #calculate cpe/rpe params + add to subj df 
            #must loop through subj_df 
        #mle var calculations
        tcpe_mle = []
        dcpe_mle = []
        tcf_mle  = []
        dcf_mle  = []
        treg_mle = []
        dreg_mle = []
        trel_mle = []
        drel_mle = []
        rpe_mle  = []
        prpe_mle = []
        nrpe_mle = []
        #emmap var calculations
        tcpe_emmap = []
        dcpe_emmap = []
        tcf_emmap  = []
        dcf_emmap  = []
        treg_emmap = []
        dreg_emmap = []
        trel_emmap = []
        drel_emmap = []
        rpe_emmap  = []
        prpe_emmap = []
        nrpe_emmap = []

        for t in range(len(subj_df)):
            trial_info = subj_df.iloc[t]
            if trial_info['GambleChoice']=='gamble':
                # subj choice = gamble
                if trial_info['Outcome']=='good':
                # gamble outcome = good > won high bet 
                    
                    # cpe calculations
                    tcpe_mle.append(mle_data['WeightedHigh'][t]-mle_data['WeightedLow'][t])
                    tcpe_emmap.append(emmap_data['WeightedHigh'][t]-emmap_data['WeightedLow'][t])
                    dcpe_mle.append(mle_data['WeightedHigh'][t]-mle_data['WeightedSafe'][t])
                    dcpe_emmap.append(emmap_data['WeightedHigh'][t]-emmap_data['WeightedSafe'][t])
                    tcf_mle.append(mle_data['WeightedLow'][t]) #### consider whether these should be positive or negative later
                    tcf_emmap.append(emmap_data['WeightedLow'][t]) 
                    dcf_mle.append(mle_data['WeightedSafe'][t])
                    dcf_emmap.append(emmap_data['WeightedSafe'][t])

                    # regret calculations
                    treg_mle.append(0)
                    treg_emmap.append(0)
                    dreg_mle.append(0)
                    dreg_emmap.append(0)

                    # relief calculations
                    trel_mle.append(mle_data['WeightedHigh'][t]-mle_data['WeightedLow'][t])
                    trel_emmap.append(emmap_data['WeightedHigh'][t]-emmap_data['WeightedLow'][t])
                    drel_mle.append(mle_data['WeightedHigh'][t]-mle_data['WeightedSafe'][t])
                    drel_emmap.append(emmap_data['WeightedHigh'][t]-emmap_data['WeightedSafe'][t])

                    # rpe calculations
                    rpe_mle.append(mle_data['WeightedHigh'][t]-mle_data['util_gamble'][t])
                    rpe_emmap.append(emmap_data['WeightedHigh'][t]-emmap_data['util_gamble'][t])
                    prpe_mle.append(mle_data['WeightedHigh'][t]-mle_data['util_gamble'][t])
                    prpe_emmap.append(emmap_data['WeightedHigh'][t]-emmap_data['util_gamble'][t])
                    nrpe_mle.append(0)
                    nrpe_emmap.append(0)


                elif trial_info['Outcome']=='bad':
                # gamble outcome = bad > won low bet 
                    
                    # cpe calculations
                    tcpe_mle.append(mle_data['WeightedLow'][t]-mle_data['WeightedHigh'][t])
                    tcpe_emmap.append(emmap_data['WeightedLow'][t]-emmap_data['WeightedHigh'][t])
                    dcpe_mle.append(mle_data['WeightedLow'][t]-mle_data['WeightedSafe'][t])
                    dcpe_emmap.append(emmap_data['WeightedLow'][t]-emmap_data['WeightedSafe'][t])
                    tcf_mle.append(mle_data['WeightedHigh'][t])
                    tcf_emmap.append(emmap_data['WeightedHigh'][t])
                    dcf_mle.append(mle_data['WeightedSafe'][t])
                    dcf_emmap.append(emmap_data['WeightedSafe'][t])

                    # regret calculations
                    treg_mle.append(mle_data['WeightedLow'][t]-mle_data['WeightedHigh'][t])                
                    treg_emmap.append(emmap_data['WeightedLow'][t]-emmap_data['WeightedHigh'][t])
                    dreg_mle.append(mle_data['WeightedLow'][t]-mle_data['WeightedSafe'][t])
                    dreg_emmap.append(emmap_data['WeightedLow'][t]-emmap_data['WeightedSafe'][t])

                    # relief calculations
                    trel_mle.append(0)
                    trel_emmap.append(0)
                    drel_mle.append(0)
                    drel_emmap.append(0)

                    # rpe calculations
                    rpe_mle.append(mle_data['WeightedLow'][t]-mle_data['util_gamble'][t])
                    rpe_emmap.append(emmap_data['WeightedLow'][t]-emmap_data['util_gamble'][t])
                    prpe_mle.append(0)
                    prpe_emmap.append(0)
                    nrpe_mle.append(mle_data['WeightedLow'][t]-mle_data['util_gamble'][t])
                    nrpe_emmap.append(emmap_data['WeightedLow'][t]-emmap_data['util_gamble'][t])

                else: 
                #fail trials
                    tcpe_mle.append(0)
                    tcpe_emmap.append(0)
                    dcpe_mle.append(0)
                    dcpe_emmap.append(0)
                    tcf_mle.append(0)
                    tcf_emmap.append(0)
                    dcf_mle.append(0)
                    dcf_emmap.append(0)
                    treg_mle.append(0)
                    treg_emmap.append(0)
                    dreg_mle.append(0)
                    dreg_emmap.append(0)
                    trel_mle.append(0)
                    trel_emmap.append(0)
                    drel_mle.append(0)
                    drel_emmap.append(0)
                    rpe_mle.append(0)
                    rpe_emmap.append(0)
                    prpe_mle.append(0)
                    prpe_emmap.append(0)
                    nrpe_mle.append(0)
                    nrpe_emmap.append(0)


            elif trial_info['GambleChoice']=='safe':
                # subj choice = safe
                if trial_info['Outcome']=='good':
                # safe outcome = good > would have won low bet 
                    
                    # cpe calculations
                    tcpe_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedLow'][t])
                    tcpe_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedLow'][t])
                    dcpe_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedLow'][t])
                    dcpe_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedLow'][t])
                    tcf_mle.append(mle_data['WeightedLow'][t])
                    tcf_emmap.append(emmap_data['WeightedLow'][t])
                    dcf_mle.append(mle_data['WeightedLow'][t])
                    dcf_emmap.append(emmap_data['WeightedLow'][t])
                    
                    # regret calculations
                    treg_mle.append(0)
                    treg_emmap.append(0)
                    dreg_mle.append(0)
                    dreg_emmap.append(0)

                    # relief calculations
                    trel_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedLow'][t])
                    trel_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedLow'][t])
                    drel_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedLow'][t])
                    drel_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedLow'][t])
                    
                    # rpe calculations
                    rpe_mle.append(0)
                    rpe_emmap.append(0)
                    prpe_mle.append(0)
                    prpe_emmap.append(0)
                    nrpe_mle.append(0)
                    nrpe_emmap.append(0)

                elif trial_info['Outcome']=='bad':
                # safe outcome = bad > would have won high bet
                    
                    # cpe calculations
                    tcpe_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedHigh'][t])
                    tcpe_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedHigh'][t])
                    dcpe_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedHigh'][t])
                    dcpe_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedHigh'][t])
                    tcf_mle.append(mle_data['WeightedHigh'][t])
                    tcf_emmap.append(emmap_data['WeightedHigh'][t])
                    dcf_mle.append(mle_data['WeightedHigh'][t])
                    dcf_emmap.append(emmap_data['WeightedHigh'][t])
                    
                    # regret calculations
                    treg_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedHigh'][t])                
                    treg_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedHigh'][t])
                    dreg_mle.append(mle_data['WeightedSafe'][t]-mle_data['WeightedHigh'][t])
                    dreg_emmap.append(emmap_data['WeightedSafe'][t]-emmap_data['WeightedHigh'][t])

                    # relief calculations
                    trel_mle.append(0)
                    trel_emmap.append(0)
                    drel_mle.append(0)
                    drel_emmap.append(0)

                    # rpe calculations
                    rpe_mle.append(0)
                    rpe_emmap.append(0)
                    prpe_mle.append(0)
                    prpe_emmap.append(0)
                    nrpe_mle.append(0)
                    nrpe_emmap.append(0)

                else: 
                #fail trials    
                    tcpe_mle.append(0)
                    tcpe_emmap.append(0)
                    dcpe_mle.append(0)
                    dcpe_emmap.append(0)
                    tcf_mle.append(0)
                    tcf_emmap.append(0)
                    dcf_mle.append(0)
                    dcf_emmap.append(0)
                    treg_mle.append(0)
                    treg_emmap.append(0)
                    dreg_mle.append(0)
                    dreg_emmap.append(0)
                    trel_mle.append(0)
                    trel_emmap.append(0)
                    drel_mle.append(0)
                    drel_emmap.append(0)
                    rpe_mle.append(0)
                    rpe_emmap.append(0)
                    prpe_mle.append(0)
                    prpe_emmap.append(0)
                    nrpe_mle.append(0)
                    nrpe_emmap.append(0)

            else: 
            #fail trials
                tcpe_mle.append(0)
                tcpe_emmap.append(0)
                dcpe_mle.append(0)
                dcpe_emmap.append(0)
                tcf_mle.append(0)
                tcf_emmap.append(0)
                dcf_mle.append(0)
                dcf_emmap.append(0)
                treg_mle.append(0)
                treg_emmap.append(0)
                dreg_mle.append(0)
                dreg_emmap.append(0)
                trel_mle.append(0)
                trel_emmap.append(0)
                drel_mle.append(0)
                drel_emmap.append(0)
                rpe_mle.append(0)
                rpe_emmap.append(0)
                prpe_mle.append(0)
                prpe_emmap.append(0)
                nrpe_mle.append(0)
                nrpe_emmap.append(0)

        
        #add calculated cpe/rpe mle params to subj df 
        subj_df['util_tCPE_mle']     = tcpe_mle
        subj_df['util_dCPE_mle']     = dcpe_mle
        subj_df['util_tCF_mle']      = tcf_mle
        subj_df['util_dCF_mle']      = dcf_mle
        subj_df['util_tRegret_mle']  = treg_mle
        subj_df['util_dRegret_mle']  = dreg_mle
        subj_df['util_tRelief_mle']  = trel_mle
        subj_df['util_dRelief_mle']  = drel_mle
        subj_df['util_RPE_mle']      = rpe_mle
        subj_df['util_pRPE_mle']     = prpe_mle
        subj_df['util_nRPE_mle']     = nrpe_mle

        #add calculated cpe/rpe emmap params to subj df 
        subj_df['util_tCPE_emmap']     = tcpe_emmap
        subj_df['util_dCPE_emmap']     = dcpe_emmap
        subj_df['util_tCF_emmap']      = tcf_emmap
        subj_df['util_dCF_emmap']      = dcf_emmap
        subj_df['util_tRegret_emmap']  = treg_emmap
        subj_df['util_dRegret_emmap']  = dreg_emmap
        subj_df['util_tRelief_emmap']  = trel_emmap
        subj_df['util_dRelief_emmap']  = drel_emmap
        subj_df['util_RPE_emmap']      = rpe_emmap
        subj_df['util_pRPE_emmap']     = prpe_emmap
        subj_df['util_nRPE_emmap']     = nrpe_emmap

        #append to list of all subj dfs
        pt_task_dfs.append(subj_df) 

        #save new task df as pt_task_data
        subj_df.to_csv(f'{behav_dir}{subj_id}_pt_task_data')

        return pt_task_dfs


def get_pt_utils(task): #updated to be correct calculations


    util_rpe = []
    util_tcpe = []
    util_dcpe = []
    util_tregret = []
    util_dregret = []
    util_trelief = []
    util_drelief = []

    for t in range(len(task)):
        
        if task['GambleChoice'][t]=='gamble':


            if task['Outcome'][t]=='good':

                util_rpe.append(task['weighted_high'][t]-task['util_g'][t]) #gamble good means won high bet 
                util_tcpe.append(task['weighted_high'][t] - task['weighted_low'][t]) #updated to be correct calculation
                util_dcpe.append(task['weighted_high'][t] - task['util_s'][t])
                util_tregret.append(0)
                util_dregret.append(0)
                util_trelief.append(task['weighted_high'][t] - task['weighted_low'][t]) #outcome - worst possible
                util_drelief.append(task['weighted_high'][t] - task['util_s'][t]) #outcome - worse safe decision
            elif task['Outcome'][t]=='bad':
                util_rpe.append(task['weighted_low'][t]-task['util_g'][t])
                util_tcpe.append(task['weighted_low'][t] - task['weighted_high'][t])
                util_dcpe.append(task['weighted_low'][t] - task['util_s'][t])
                util_tregret.append(task['weighted_low'][t] - task['weighted_high'][t]) #outcome - best possible
                util_dregret.append(task['weighted_low'][t] - task['util_s'][t]) #outcome - better safe decision
                util_trelief.append(0)
                util_drelief.append(0)
            else: #fail trials
                util_rpe.append(0)
                util_tcpe.append(0)
                util_dcpe.append(0)
                util_tregret.append(0)
                util_dregret.append(0)
                util_trelief.append(0)
                util_drelief.append(0)
        
        elif task['GambleChoice'][t]=='safe':
            util_rpe.append(0)
            if task['Outcome'][t]=='good':
                util_tcpe.append(task['util_s'][t] - task['weighted_low'][t])
                util_dcpe.append(task['util_s'][t] - task['weighted_low'][t])
                util_tregret.append(0)
                util_dregret.append(0)
                util_trelief.append(task['util_s'][t] - task['weighted_low'][t]) #outcome - worst possible
                util_drelief.append(task['util_s'][t] - task['weighted_low'][t]) #no difference for safe trials
            elif task['Outcome'][t]=='bad':
                util_tcpe.append(task['util_s'][t] - task['weighted_high'][t])
                util_dcpe.append(task['util_s'][t] - task['weighted_high'][t])
                util_tregret.append(task['util_s'][t] - task['weighted_high'][t]) #outcome - best possible
                util_dregret.append(task['util_s'][t] - task['weighted_high'][t]) #no difference for safe trials 
                util_trelief.append(0)
                util_drelief.append(0)
            else: #fail trials
                util_tcpe.append(0)
                util_dcpe.append(0)
                util_tregret.append(0)
                util_dregret.append(0)
                util_trelief.append(0)
                util_drelief.append(0)
        
        else: #fail trials
            util_rpe.append(0)
            util_tcpe.append(0)
            util_dcpe.append(0)
            util_tregret.append(0)
            util_dregret.append(0)
            util_trelief.append(0)
            util_drelief.append(0)


    task['util_rpe'] = util_rpe
    task['util_tcpe'] = util_tcpe
    task['util_dcpe'] = util_dcpe 
    task['util_tregret'] = util_tregret
    task['util_dregret'] = util_dregret
    task['util_trelief'] = util_trelief
    task['util_drelief'] = util_drelief


    return task

def get_glm_data_all_subj(model_data_vars,subj_ids,behav_dir):
    
    #dictionary to hold all subj data
    all_subj_model_dict = {}   

    # make a list of regressor names from regressor list - each regressor needs 3 vars for t-1,t-2,t-3 trials 
    model_data_dict_keys = []
    for var in model_data_vars:
        t1_col = var + '_t-1' 
        t2_col = var + '_t-2'
        t3_col = var + '_t-3'
        model_data_dict_keys.append(t1_col)
        model_data_dict_keys.append(t2_col)
        model_data_dict_keys.append(t3_col)

    #create model data pandas df with model_data_dict_keys as column names 
    model_df_col_names = ['subj_id','round','rate','zscore_rate','bdi','bai'] + model_data_dict_keys
    all_subj_model_df = pd.DataFrame(columns = model_df_col_names)
      

    for subj_id in subj_ids:

        task_df = pd.read_csv(f'{behav_dir}{subj_id}_pt_task_data')
        rate_df = pd.read_csv(f'{behav_dir}{subj_id}_rate_data')

        # make a list of regressor names from regressor list - each regressor needs 3 vars for t-1,t-2,t-3 trials 
        model_data_dict_keys = []
        for var in model_data_vars:
            t1_col = var + '_t-1' 
            t2_col = var + '_t-2'
            t3_col = var + '_t-3'
            model_data_dict_keys.append(t1_col)
            model_data_dict_keys.append(t2_col)
            model_data_dict_keys.append(t3_col)

        #create model data dictionary with model_data_dict_keys as keys and empty lists as their values (can use append function in loop now)
        model_data_dict = {}
        for i in model_data_dict_keys:
            model_data_dict[i] = []

        # #get rating info
        round       = rate_df['Round'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:] #need index of last round 1 because some pts have multiple round 1 scores, start after last round 1 index
        rate        = rate_df['Rating'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
        zscore_rate = rate_df['zscore_mood'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
        bdi         = rate_df['bdi'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
        bai         = rate_df['bai'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]


        #add subj info and non task vars to model data dict
        model_data_dict['subj_id'] = [subj_id]*50
        model_data_dict['round'] = round
        model_data_dict['rate'] = rate
        model_data_dict['zscore_rate'] = zscore_rate
        model_data_dict['bdi'] = bdi
        model_data_dict['bai'] = bai

        for r in round: #iterate through mood rating rounds (4,7,10...151)
            #calculate row index for task df
            t3 = r-4 #t-3 trial 
            t2 = r-3 #t-2 trial
            t1 = r-2 #t-1 trial
            
            for reg in model_data_vars:
                # make reg name strings for model data keys
                reg_t3_col = reg + '_t-3'
                reg_t2_col = reg + '_t-2'
                reg_t1_col = reg + '_t-1'

                model_data_dict[reg_t3_col].append(task_df[reg][t3])
                model_data_dict[reg_t2_col].append(task_df[reg][t2])
                model_data_dict[reg_t1_col].append(task_df[reg][t1])
        
        #add to master dictionary 
        all_subj_model_dict[subj_id] = model_data_dict #in case dictionary of dictionaries is easier to work with later
        #add to all_subj_model_df
        all_subj_model_df = pd.concat([all_subj_model_df,pd.DataFrame(model_data_dict)])
    
    return model_data_dict

def get_glm_data_single_subj(subj_id,behav_dir,model_data_vars):

    #load subject task data - pt data  
    task_df = pd.read_csv(f'{behav_dir}{subj_id}_pt_task_data')
    rate_df = pd.read_csv(f'{behav_dir}{subj_id}_rate_data')

    # make a list of regressor names from regressor list - each regressor needs 3 vars for t-1,t-2,t-3 trials 
    model_data_dict_keys = []
    for var in model_data_vars:
        t1_col = var + '_t-1' 
        t2_col = var + '_t-2'
        t3_col = var + '_t-3'
        model_data_dict_keys.append(t1_col)
        model_data_dict_keys.append(t2_col)
        model_data_dict_keys.append(t3_col)
    
    #create model data dictionary with model_data_dict_keys as keys and empty lists as their values (can use append function in loop now)
    model_data_dict = {}
    for i in model_data_dict_keys:
        model_data_dict[i] = []

    # #get rating info
    round  = rate_df['Round'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:] #need index of last round 1 because some pts have multiple round 1 scores, start after last round 1 index
    rate   = rate_df['Rating'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
    z_rate = rate_df['zscore_mood'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
    bdi    = rate_df['bdi'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
    bai    = rate_df['bai'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]

    #check if task data is shorter than last round idx
    task_len = len(task_df)
    if task_len < list(round)[-1]:
        round  = list(round)[:-1]
        rate   = list(rate)[:-1]
        z_rate = list(z_rate)[:-1]
        bdi    = list(bdi)[:-1]
        bai    = list(bai)[:-1]



    #add subj info and non task vars to model data dict
    model_data_dict['subj_id']  = [subj_id]*len(round)
    model_data_dict['round']    = round
    model_data_dict['rate']     = rate
    model_data_dict['z_rate']   = z_rate
    model_data_dict['bdi']      = bdi
    model_data_dict['bai']      = bai

    for r in round: #iterate through mood rating rounds (4,7,10...151)
        #calculate row index for task df
        t3 = r-4 #t-3 trial 
        t2 = r-3 #t-2 trial
        t1 = r-2 #t-1 trial
        
        for reg in model_data_vars:
            # make reg name strings for model data keys
            reg_t3_col = reg + '_t-3'
            reg_t2_col = reg + '_t-2'
            reg_t1_col = reg + '_t-1'

            model_data_dict[reg_t3_col].append(task_df[reg][t3])
            model_data_dict[reg_t2_col].append(task_df[reg][t2])
            model_data_dict[reg_t1_col].append(task_df[reg][t1])
    
    return model_data_dict