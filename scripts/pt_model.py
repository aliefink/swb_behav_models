import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

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



def negll_prospect(params, subj_df):
    risk_aversion, loss_aversion, inverse_temp = params

    # init list of choice prob predictions
    choiceprob_list = []
    #utils = pd.DataFrame(columns = ['trial_idx','choice','util_gamble','util_safe','p_gamble','p_safe'])

    #loop through trials
    for trial in range(len(subj_df)):

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


            

            util_g.append(util_gamble)
            util_s.append(util_safe)
            p_g.append(p_gamble)
            p_s.append(p_safe)

            if p_gamble > p_safe:
                choice_prob.append(p_gamble)
                choice_pred.append('gamble')
            elif p_safe > p_gamble:
                choice_prob.append(p_safe)
                choice_pred.append('safe')
            else:
                choice_prob.append(0)
                choice_pred.append(0)
            tr.append(trial)

            rep_list.append(rep)


    data = {'rep':rep_list,'tr':tr,'type':trial_list,'choice_pred':choice_pred,'choice_prob':choice_prob,
                       'util_gamble':util_g,'util_safe':util_s,'p_gamble':p_g,'p_safe':p_s,'safe_bet':safe,'high_bet':high,'low_bet':low}
    DF = pd.DataFrame(data)
    
    return DF

