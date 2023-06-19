from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
import random
from sklearn.metrics import r2_score
import random



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



#function to estimate residuals for parameter optimization
def swb_leastsq(params,df,n_regs,reg_list):
    #params is list of lambda estimate + beta estimates 
    lam = params[0]
    ls = [1,lam,lam**2]
    betas = params[1:]
    param_eq = 0

    for n in range(n_regs):
        #beta value (intercept is first index, so need +1)
        b = betas[n+1] 
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

        param_eq += (b*l1*reg1_vec) + (b*l2*reg2_vec) + (b*l2*reg2_vec)


    mood_est = betas[0] + param_eq
    mood_obs = np.array(df['zscore_mood'])
    #compute the vector of residuals
    mood_residuals = mood_obs - mood_est
    return mood_residuals



def run_swb(df, subj_ids, n_regs, reg_list):
    #df = model data for all subjects
    #subj_ids = desired subj in df
    #n_regs = number of task variables in model (ex: ev,cr,rpe = 3 n_regs)
    #reg_list = list of column names in df as str (should be len n_regs*3, 3 trials for each variable)

    mood_est_df = pd.DataFrame(columns = subj_ids)
    optim_inits_df = pd.DataFrame(columns = subj_ids)
    param_fits_df = pd.DataFrame(columns = subj_ids)
    bic_dict = {}
    rsq_dict = {}

    for ix, subj_id in enumerate(subj_ids):

        subj_df = df[df["subjID"]==subj_id]

        #set sse to np.inf to compare to optim result
        res_cost = np.inf
        param_fits = []
        optim_vars = []
        residuals = []
        rss = []

        #start multiple initializations for betas and lambda - optimize each time and find best, arbitrarily picked n_regs for consistency
        for n in range(n_regs):
            lambda_init = random.uniform(0, 1)
            betas_init = np.random.random(size = (n_regs+1))
            param_inits = np.hstack((lambda_init,betas_init))

            # minimize MSE using scipy.optimize.minimize
            res = least_squares(swb_leastsq, # objective function
                        (param_inits),
                        args=(subj_df,n_regs,reg_list),
                        method='lm') # arguments
            
            residuals = res.fun #residuals output from best model
            cost = res.cost
            if cost < res_cost:
                res_cost = cost
                optim_vars = param_inits
                param_fits = res.x
                residuals = residuals 
                rss = sum(residuals**2)

        
        if res_cost == np.inf:
            print('No solution for ',subj_id)
            mood_est = np.empty(shape=50)
            optim_vars = np.empty(shape=len(param_inits))
            param_fits = np.empty(shape=len(param_inits))
            BIC = 0
            rsq = 0
        else:
            mood_est = np.array(subj_df.zscore_mood) - residuals
            BIC = (len(residuals) * np.log(rss/len(residuals))) + (len(param_inits)*np.log(len(residuals)))
            rsq = r2_score(np.array(subj_df.zscore_mood),mood_est)

        mood_est_df[subj_id] = mood_est
        optim_inits_df[subj_id] = optim_vars
        param_fits_df[subj_id] = param_fits
        bic_dict[subj_id] = BIC
        rsq_dict[subj_id] = rsq
    
    
    return mood_est_df, optim_inits_df, param_fits_df, bic_dict, rsq_dict



#function to extract mood estimates from already optimized parameters 
def swb_fit(betas,lam,df,n_regs,reg_list):
    #params is list of lambda estimate + beta estimates 
    ls = [1,lam,lam**2]
    param_eq = 0

    for n in range(n_regs):
        #beta value (intercept is first index, so need +1)
        b = betas[n+1] 
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

        param_eq += (b*l1*reg1_vec) + (b*l2*reg2_vec) + (b*l2*reg2_vec)


    mood_est = betas[0] + param_eq
    return mood_est

        


def run_pt(subj_df,rho_inits,lambda_inits,beta_inits):
    # gradient descent to minimize neg LL

    # rho    = risk_aversion
    # lambda = loss_aversion
    # beta   = inverse_temp
    subj_df = (subj_df)
    res_nll = np.inf


    # guess several different starting points for rho
    for rho_guess in tqdm(rho_inits):
            for lambda_guess in lambda_inits:
                for beta_guess in beta_inits:
            
                    # guesses for alpha, theta will change on each loop
                    init_guess = (rho_guess, lambda_guess, beta_guess)
                    
                    # minimize neg LL
                    result = minimize(negll_prospect, 
                                    init_guess, 
                                    subj_df, 
                                    bounds=((0,6),(0,50),(.001,50)))
                    
                    # if current negLL is smaller than the last negLL,
                    # then store current data
                    if result.fun < res_nll:
                        res_nll = result.fun
                        param_fits = result.x
                        risk_aversion, loss_aversion, inverse_temp = param_fits
                        optim_vars = init_guess
                    

    if res_nll == np.inf:
        print('No solution for this patient')
        risk_aversion=0
        loss_aversion=0
        inverse_temp=0
        BIC=0
        optim_vars=0
        #return risk_aversion, loss_aversion, inverse_temp, BIC, optim_vars
    else:
        BIC = len(init_guess) * np.log(len(subj_df)) + 2*res_nll
    
    return risk_aversion, loss_aversion, inverse_temp, BIC, optim_vars


def negll_base_pt(params, subj_df):
    risk_aversion, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    choiceprob_list = []
    #utils = pd.DataFrame(columns = ['trial_idx','choice','util_gamble','util_safe','p_gamble','p_safe'])

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
            weighted_high_bet = 0.5 * (high_bet)**risk_aversion
        else: #loss trials
            weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
        
        # transform to low bet value to utility (gamble)
        if low_bet >= 0: #gain trials
            weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
        else: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * (-low_bet)**risk_aversion
        
        util_gamble = weighted_high_bet + weighted_low_bet
    

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion
        else: #loss trials
            util_safe = -loss_aversion * (-safe_bet)**risk_aversion



        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        

        #appending to utils df for later param analysis
        #utils.append({'trial_idx':trial_info['Round'],'choice':choice,'util_gamble':util_gamble,'util_safe':util_safe,'p_gamble':p_gamble,'p_safe':p_safe},ignore_index = True)


        # append probability of chosen options
        if choice == 'gamble':
            choiceprob_list.append(p_gamble)
        elif choice == 'safe':
            choiceprob_list.append(p_safe)
        else:
            choiceprob_list.append(0)

    # compute the neg LL of choice probabilities across the entire task
    negLL = -np.sum(np.log(choiceprob_list))
    
    if np.isnan(negLL):
        return np.inf
    else:
        return negLL
    

##### after finding optimal parameters - get ptEV
def get_ptEV(params, subj_df):
    #put in estimated optimal params from negll_prospect
    risk_aversion, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    tr = []
    choice_prob = []
    choice_pred = []
    util_g = []
    util_s = []
    p_g = []
    p_s = []


    #loop through trials
    for trial in range(len(subj_df)):
        tr.append(trial)

        # get relevant trial info
        trial_info = subj_df.iloc[trial]
        high_bet = trial_info['High.Bet']
        low_bet = trial_info['Low.Bet']
        safe_bet = trial_info['Safe.Bet']
        trial_type = trial_info['TrialType']
        choice = trial_info['Gamble.Choice']
        outcome = trial_info['Profit']

        # transform to high bet value to utility (gamble)
        if high_bet > 0: #mix or gain trials
            weighted_high_bet = 0.5 * (high_bet)**risk_aversion
        else: #loss trials
            weighted_high_bet = 0 
        
        # transform to low bet value to utility (gamble)
        if low_bet >= 0: #gain trials
            weighted_low_bet = 0 
        else: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * (-low_bet)**risk_aversion
        
        util_gamble = weighted_high_bet + weighted_low_bet
    

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion
        else: #loss trials
            util_safe = -loss_aversion * (-safe_bet)**risk_aversion



        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        

        #appending to utils df for later param analysis

        util_g.append(util_gamble)
        util_s.append(util_safe)
        p_g.append(p_gamble)
        p_s.append(p_safe)



        #getting predictions of model 
        if p_gamble > p_safe:
            choice_pred.append('gamble')
        elif p_safe > p_gamble:
            choice_pred.append('safe')
        else:
            choice_pred.append(0)


        #getting model probabilities of actual choices
        if choice == 'gamble':
            choice_prob.append(p_gamble)
        elif choice == 'safe':
            choice_prob.append(p_safe)
        else:
            choice_prob.append(0)

    
    
    DF = pd.DataFrame(data = zip(tr, choice_pred, choice_prob, util_g, util_s, p_g, p_s),
                          columns =['tr','choice_pred','choice_prob','util_gamble','util_safe','p_gamble','p_safe'])
        
    return DF


###### prospect theory model as a simulator not estimator for parameter recovery

def pt_base_simulation(params,rep,trials):
    #inputs: 
    #params - risk, loss, temp
    #rep - number of times to run simulation
    #trials - number of trials for simulation (for EMU SWB always 150)
    risk_aversion, loss_aversion, inverse_temp = params

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
                weighted_high_bet = 0.5 * (high_bet)**risk_aversion
            else: #loss trials
                weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
            
            # transform to low bet value to utility (gamble)
            if low_bet >= 0: #gain trials
                weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
            else: #loss and mix trials
                weighted_low_bet = -0.5 * loss_aversion * (-low_bet)**risk_aversion
            
            util_gamble = weighted_high_bet + weighted_low_bet
        

            # transform safe bet value to utility (safe)
            if safe_bet >= 0: #gain or mix trials
                util_safe = (safe_bet)**risk_aversion
            else: #loss trials
                util_safe = -loss_aversion * (-safe_bet)**risk_aversion
            
            # utility options for calculating EV - utils separate, ug - us to combine or Uchosen - Unchosen (will differ by participant) 
            #inverse temp < 1 more exporatory, > 1 more exploitative
            # convert EV to choice probabilities via softmax
            p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
            p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )

            if np.isnan(p_gamble):
                p_gamble = 0.99
            if np.isnan(p_safe):
                p_safe = 0.99
            

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

def simulation_norm_gamble_choices(df):
    
    #df is task data for a single subject
    loss_df = df[df.type == 'loss']
    mix_df = df[df.type == 'mix']
    gain_df = df[df.type == 'gain']

    #loss
    loss_dict = {}
    loss_norm = -((loss_df['low_bet'] + loss_df['high_bet'])/2)/loss_df['safe_bet']
    loss_quant = np.quantile(loss_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    loss_x_axis = [np.mean(loss_quant[i:i+2],dtype=np.float64) for i in range(5)]
    loss_dec = loss_df['choice_pred'].replace(['gamble','safe'],[1,0])
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
    mix_norm = ((mix_df['low_bet'] + mix_df['high_bet'])/2) #can't divide by zero
    mix_quant = np.quantile(mix_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    mix_x_axis = [np.mean(mix_quant[i:i+2],dtype=np.float64) for i in range(5)]
    mix_dec = mix_df['choice_pred'].replace(['gamble','safe'],[1,0])
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
    gain_norm = ((gain_df['low_bet'] + gain_df['high_bet'])/2)/gain_df['safe_bet']
    gain_quant = np.quantile(gain_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    gain_x_axis = [np.mean(gain_quant[i:i+2],dtype=np.float64) for i in range(5)]
    gain_dec = gain_df['choice_pred'].replace(['gamble','safe'],[1,0])
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
    loss_df = df[df.type == 'loss']
    mix_df = df[df.type == 'mix']
    gain_df = df[df.type == 'gain']

    #loss
    loss_dict = {}
    #loss_norm = -((loss_df['low_bet'] + loss_df['high_bet'])/2)/loss_df['safe_bet']
    #replacing EV with utility
    loss_norm = -loss_df['util_gamble']/loss_df['util_safe'] #util_g/util_s
    loss_quant = np.quantile(loss_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    loss_x_axis = [np.mean(loss_quant[i:i+2],dtype=np.float64) for i in range(5)]
    loss_dec = loss_df['choice_pred'].replace(['gamble','safe'],[1,0])
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
    mix_dec = mix_df['choice_pred'].replace(['gamble','safe'],[1,0])
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
    gain_dec = gain_df['choice_pred'].replace(['gamble','safe'],[1,0])
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

##### base pt model parameter recovery function 

def recover_base_pt(subj_df,risk_inits,loss_inits,temp_inits):
    # gradient descent to minimize neg LL

    subj_df = (subj_df)
    res_nll = np.inf


    # guess several different starting points for rho
    for risk_guess in risk_inits:
            for loss_guess in loss_inits:
                for temp_guess in temp_inits:
                    #print(risk_guess,loss_guess,temp_guess)
            
                    # guesses for alpha, theta will change on each loop
                    init_guess = (risk_guess, loss_guess, temp_guess)
                    
                    # minimize neg LL
                    result = minimize(negll_base_pt, 
                                    x0=init_guess, 
                                    args=subj_df, 
                                    method='L-BFGS-B',
                                    bounds=((0,5),(0,5),(0,10))) #should match bounds given to param_init , should probalby not be hard coded...
                    
                    # if current negLL is smaller than the last negLL,
                    # then store current data
                    if result.fun < res_nll:
                        res_nll = result.fun
                        param_fits = result.x
                        risk_aversion, loss_aversion, inverse_temp = param_fits
                        optim_vars = init_guess
                    

    if res_nll == np.inf:
        print('No solution for this patient')
        risk_aversion=0
        loss_aversion=0
        inverse_temp=0
        BIC=0
        optim_vars=0
        #return risk_aversion, loss_aversion, inverse_temp, BIC, optim_vars
    else:
        BIC = len(init_guess) * np.log(len(subj_df)) + 2*res_nll
    
    return risk_aversion, loss_aversion, inverse_temp, BIC, optim_vars


###### dual risk prospect theory model as a simulator not estimator for parameter recovery
def pt_dual_risk_simulation(params,rep,trials):
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
                weighted_high_bet = 0.5 * (high_bet)**risk_aversion_gain #different risk aversion parameter for gain values
            else: #loss trials
                weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
            
            # transform to low bet value to utility (gamble)
            if low_bet >= 0: #gain trials
                weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
            else: #loss and mix trials
                weighted_low_bet = -0.5 * loss_aversion * (-low_bet)**risk_aversion_loss #different risk aversion parameter for loss values
            
            util_gamble = weighted_high_bet + weighted_low_bet
        

            # transform safe bet value to utility (safe)
            if safe_bet >= 0: #gain or mix trials
                util_safe = (safe_bet)**risk_aversion_gain #gain risk aversion parameter
            else: #loss trials
                util_safe = -loss_aversion * (-safe_bet)**risk_aversion_loss #loss risk aversion parameter
            
            # utility options for calculating EV - utils separate, ug - us to combine or Uchosen - Unchosen (will differ by participant) 
            #inverse temp < 1 more exporatory, > 1 more exploitative
            # convert EV to choice probabilities via softmax
            p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
            p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
            

            if np.isnan(p_gamble):
                p_gamble = 0.99
            if np.isnan(p_safe):
                p_safe = 0.99

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


def negll_dual_risk_pt(params, subj_df):
    risk_aversion_gain, risk_aversion_loss, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    choiceprob_list = []
    #utils = pd.DataFrame(columns = ['trial_idx','choice','util_gamble','util_safe','p_gamble','p_safe'])

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
            weighted_high_bet = 0.5 * (high_bet)**risk_aversion_gain #different risk aversion parameter for gain values
        else: #loss trials
            weighted_high_bet = 0 # -0.5 * loss_aversion * (-high_bet)**risk_aversion - this is never the case so changed to zero 
        
        # transform to low bet value to utility (gamble)
        if low_bet >= 0: #gain trials
            weighted_low_bet = 0 #0.5 * (low_bet)**risk_aversion - this is never the case so changed to zero 
        else: #loss and mix trials
            weighted_low_bet = -0.5 * loss_aversion * (-low_bet)**risk_aversion_loss #different risk aversion parameter for loss values
        
        util_gamble = weighted_high_bet + weighted_low_bet
    

        # transform safe bet value to utility (safe)
        if safe_bet >= 0: #gain or mix trials
            util_safe = (safe_bet)**risk_aversion_gain #gain risk aversion parameter
        else: #loss trials
            util_safe = -loss_aversion * (-safe_bet)**risk_aversion_loss #loss risk aversion parameter
        
        # utility options for calculating EV - utils separate, ug - us to combine or Uchosen - Unchosen (will differ by participant) 
        #inverse temp < 1 more exporatory, > 1 more exploitative
        # convert EV to choice probabilities via softmax
        p_gamble = np.exp(inverse_temp*util_gamble) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        p_safe = np.exp(inverse_temp*util_safe) / ( np.exp(inverse_temp*util_gamble) + np.exp(inverse_temp*util_safe) )
        

        #appending to utils df for later param analysis
        #utils.append({'trial_idx':trial_info['Round'],'choice':choice,'util_gamble':util_gamble,'util_safe':util_safe,'p_gamble':p_gamble,'p_safe':p_safe},ignore_index = True)


        # append probability of chosen options
        if choice == 'gamble':
            choiceprob_list.append(p_gamble)
        elif choice == 'safe':
            choiceprob_list.append(p_safe)
        else:
            choiceprob_list.append(0)

    # compute the neg LL of choice probabilities across the entire task
    negLL = -np.sum(np.log(choiceprob_list))
    
    if np.isnan(negLL):
        return np.inf
    else:
        return negLL
    

def recover_dual_risk_pt(subj_df,risk_gain_inits,risk_loss_inits,loss_inits,temp_inits):
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
                                    bounds=((0,5),(0,5),(0,5),(0,10))) #should probably not be hard coded..
                    
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