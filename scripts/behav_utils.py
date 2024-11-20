import numpy as np 
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore

def format_all_behav(raw_behav, return_drops=False, drop_bads=False, drop_bads_t1=False,  norm=True,norm_type='zscore'):
    # cont_vars = ['SafeBet','LowBet','HighBet','Profit', 'TotalProfit','GambleEV', 'TrialEV',
    #                                          'CR', 'choiceEV','rpe', 'orthog_rpe','cf','cpe','SafeBet_t1', 'LowBet_t1',                                                        'HighBet_t1', 'Profit_t1','TotalProfit_t1','GambleEV_t1', 
    #                                          'TrialEV_t1', 'CR_t1','choiceEV_t1','rpe_t1', 'cf_t1', 'cpe_t1']
    '''
    Args:
    ------
    raw_behav : (list) List of subject behavior dataframes(task_df).
    cont_vars : (list) List of cols from all_behav to normalize (strings). Default is all continuous variables. 
    '''
    
    all_behav  = []
    drops_dict = {}
    
    for subj_df in raw_behav:
        subj_id = subj_df.subj_id.unique().tolist()[0]
        
        if return_drops:
            drops_dict[subj_id] = {'bad_epochs':    list(subj_df[subj_df.keep_epoch=='drop'].index),
                                   'bad_epochs_t1': list(subj_df[subj_df.keep_epoch_t1=='drop'].index)}
        
        if drop_bads:
            subj_df = subj_df[subj_df.keep_epoch=='keep']
        
        if drop_bads_t1:
            subj_df = subj_df[subj_df.keep_epoch_t1=='keep']

        if norm:
            # make list of continuous variables by getting list of col names and removing non-cont vars
            cont_vars = list(subj_df.columns.drop(['subj_id', 'bdi', 'bdi_thresh', 'TrialOnset', 'ChoiceOnset', 'DecisionOnset', 'FeedbackOnset','CpeOnset','Round','TrialNum', 'RT','TrialType','GambleChoice','Outcome','epoch','logRT','res_type',
                                                   'Round_t1', 'TrialNum_t1', 'RT_t1','TrialType_t1','GambleChoice_t1', 'Outcome_t1',
                                                  'epoch_t1', 'logRT_t1','res_type_t1','keep_epoch','keep_epoch_t1']))
            if norm_type == 'zscore':
                # normalize continuous variables after dropping bad trials 
                for var in cont_vars:
                    var_og_id = ('_').join([var,'raw'])
                    subj_df[var_og_id] = subj_df[var].copy()
                    subj_df[var] = zscore(subj_df[var].values,nan_policy='omit')
            else:
                # normalize continuous variables after dropping bad trials 
                for var in cont_vars:
                    var_og_id = ('_').join([var,'raw'])
                    subj_df[var_og_id] = subj_df[var].copy()
                    subj_df[var] = norm_zscore(subj_df[var].values)
                
        all_behav.append(subj_df.reset_index(drop=True))
        
    all_behav = pd.concat(all_behav).reset_index(drop=True)
    
    if return_drops: # return behav and drops info 
        return all_behav, drops_dict
    else:
        return all_behav


def format_all_mood(raw_mood, all_behav, return_drops=False, drop_bads=False,norm_type='zscore',
                                behav_vars = ['SafeBet', 'LowBet', 'HighBet', 'Profit', 'TotalProfit', 'GambleEV', 
                                              'TrialEV', 'CR', 'choiceEV', 'rpe', 'cf', 'cpe']):
    '''
    Args:
    ------
    raw_mood   : (list) List of subject behavior dataframes(mood_df).
    behav_vars : (list) List of cols from all_behav to include in mood data. Default is all continuous variables. 
    '''
    
    all_mood  = []
    drops_dict = {}
    
    for mood_df in raw_mood:
        subj_id = mood_df.subj_id.unique().tolist()[0]
        behav_df = all_behav[all_behav.subj_id==subj_id].reset_index(drop=True)
        # mood_df['norm_mood'] = norm_zscore(mood_df.Rating)
        mood_df['logRT'] = np.log(mood_df.RT)
        mood_df['MoodChoiceOnset'] = mood_df.RatingOnset + mood_df.RT
        mood_df['epoch'] = np.arange(0,len(mood_df))
        # t-1 trial data
        t1_task_epochs = behav_df.loc[mood_df.epoch_t1_idx]
        # print()
        # t-2 trial data
        t2_task_epochs = behav_df.loc[mood_df.epoch_t2_idx]

        # t-3 trial data
        t3_task_epochs = behav_df.loc[mood_df.epoch_t3_idx]

        for var in behav_vars:
            var_t1_data = t1_task_epochs[var].values
            var_t1 = '_'.join([var,'t1'])
            mood_df[var_t1] = var_t1_data

            var_t2_data = t2_task_epochs[var].values
            var_t2 = '_'.join([var,'t2'])
            mood_df[var_t2] = var_t2_data

            var_t3 = '_'.join([var,'t3'])
            var_t3_data = t3_task_epochs[var].values
            mood_df[var_t3] = var_t3_data

        # get round info for mood ratings, prev behav trial, next behav trial
        mood_round = mood_df.Round
        next_round = mood_df.Round_t3_idx[1:]
        prev_round = mood_df.Round_t1_idx

        # calculate end time of last trial
        mood_starts = mood_df.RatingOnset
        # extract start time of next trial 
        next_starts = [behav_df.loc[behav_df.Round.astype(int)==rnd,'TrialOnset'].values[0] 
                       for rnd in next_round]
        # calculate mood epoch duration
        mood_rts    = mood_df.RT
        end_mood    = mood_starts+mood_rts+0.1
        mood_durs   = end_mood-mood_starts 

        mood_df['mood_epoch_len']   = mood_durs
        mood_df['next_round_start'] = next_starts+[np.nan]
        mood_df.loc[mood_df.mood_epoch_len<1.0,'keep_mood'] = 'drop'
        
        if return_drops:
            # drops = []
            # drops.extend(mood_df[mood_df.RT<0.3].index)
            # drops.extend(mood_df[mood_df.mood_epoch_len<1.0].index)
            # drops_dict[subj_id] = np.unique(drops)
            drops_dict[subj_id] = list(mood_df[mood_df.keep_mood=='drop'].index)

        if drop_bads:
            # mood_df = mood_df[mood_df.RT>=0.3]
            # mood_df = mood_df[mood_df.mood_epoch_len>=1.0]
            mood_df = mood_df[mood_df.keep_mood=='keep']

        if norm_type == 'zscore':
            mood_df['norm_mood'] = zscore(mood_df.Rating,nan_policy='omit')
        else: 
            mood_df['norm_mood'] = norm_zscore(mood_df.Rating)
        
        all_mood.append(mood_df.reset_index(drop=True))

    all_mood = pd.concat(all_mood).reset_index(drop=True)
    
    if return_drops: # return behav and drops info 
        return all_mood, drops_dict
    else:
        return all_mood
    
def norm_zscore(reg_array):
    return (reg_array-np.nanmean(reg_array))/(2*np.nanstd(reg_array))

def fit_mixed_model(df, regressor_vars, outcome_var, rand_eff_var,reml=True):
    # define formula, random effects formula
    formula    = (' + ').join(regressor_vars)
    re_formula = formula
    formula    = f'{outcome_var} ~ 1 + {formula}'
    # fit model
    return smf.mixedlm(formula = formula, re_formula = re_formula,
        data = df, groups=df[rand_eff_var], missing='drop').fit(reml=reml)

def vif_scores(df, regressor_vars):
    
    cov_data_dict = {f'{reg}':[] for reg in regressor_vars}
    
    # check if data is categorical
    for reg in regressor_vars: 
        if pd.api.types.is_numeric_dtype(df[reg]):
            cov_data_dict[reg] = df[reg]
        else: 
            # factorize categorical data into numeric dummy variables 
            cov_data_dict[reg] = pd.factorize(df[reg])[0]
    
    vif_df = pd.DataFrame(cov_data_dict)


    vif_df = vif_df.astype(float)
    vif_df = vif_df.dropna()


    vif_data = pd.DataFrame() 
    vif_data["feature"] = vif_df.columns 

    # calculating VIF for each feature 
    vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) 
                              for i in range(len(vif_df.columns))] 
    return vif_data


