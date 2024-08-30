import numpy as np 
import pandas as pd
import os


def norm_zscore(reg_array):
    return (reg_array-np.nanmean(reg_array))/(2*np.nanstd(reg_array))

def format_all_behav(raw_behav,mood=False,drops_data=False,cont_vars=['SafeBet', 'LowBet', 'HighBet','Profit', 
                                                  'TotalProfit','GambleEV', 'TrialEV', 'CR',
                                                  'choiceEV','rpe', 'cf', 'max_cf', 'cpe', 'max_cpe',
                                                  'SafeBet_t1','LowBet_t1', 'HighBet_t1', 'Profit_t1',
                                                  'TotalProfit_t1','GambleEV_t1', 'TrialEV_t1','CR_t1', 
                                                  'choiceEV_t1', 'rpe_t1','cf_t1', 'max_cf_t1','cpe_t1', 'max_cpe_t1']):
    '''
    Args:
    ------
    raw_behav : (list) List of subject behavior dataframes(task_df).
    cont_vars : (list) List of cols from all_behav to normalize (strings). Default is all continuous variables. 
    '''

    if drops_data:
        drop_epochs_dict = {}
    
    all_behav = []
    
    for subj_df in raw_behav:
        subj_id = subj_df.subj_id.unique().tolist()[0]
        subj_df['CpeOnset'] = subj_df.DecisionOnset + 2.0
        
        if drops_data:
            drop_epochs = [] # these will be redundant 
            drop_epochs.extend(list(subj_df[subj_df.keep_epoch == 'drop'].epoch))
            drop_epochs.extend(list(subj_df.tail(1).epoch))
            drop_epochs.extend(list(subj_df[np.isinf(subj_df.logRT)].epoch))
            drop_epochs.extend(list(subj_df[(subj_df.Round==76)].epoch))
            drop_epochs.extend(list(subj_df[(subj_df.cpe==0.0)].epoch))
            # drop_epochs.append(subj_df[subj_df.keep_epoch_t1 == 'drop'].epoch)
            # drop_epochs.extend(list(subj_df[np.isinf(subj_df.logRT_t1)].epoch))
           
            #### save subj drop epochs 
            drop_epochs_dict[subj_id] = np.unique(drop_epochs)
        if mood: 
            # set bad_epochs values to nan
            for var in cont_vars:
                subj_df.loc[subj_df.keep_epoch == 'drop',var] = np.nan
                subj_df.loc[subj_df.Round == 76,var] = np.nan
                subj_df.loc[subj_df.cpe == 0.0,var] = np.nan
        else:
            # actually drop bad epochs
            subj_df = subj_df[subj_df.keep_epoch != 'drop']
            subj_df = subj_df[subj_df.Round != 76]
            subj_df = subj_df.drop(subj_df.tail(1).index)
        # subj_df = subj_df[subj_df.keep_epoch_t1 != 'drop']
        
        # normalize continuous variables after dropping bad trials 
        for var in cont_vars:
            subj_df[var] = norm_zscore(subj_df[var].values)
            
        all_behav.append(subj_df)
        
    all_behav = pd.concat(all_behav).reset_index(drop=True)
    
    if drops_data: # return behav and drops info 
        return all_behav, drop_epochs_dict
    else:
        return all_behav

# def format_all_behav(raw_behav,drops_data=False,cont_vars=['SafeBet', 'LowBet', 'HighBet','Profit', 
#                                                   'TotalProfit','GambleEV', 'TrialEV', 'CR',
#                                                   'choiceEV','rpe', 'cf', 'max_cf', 'cpe', 'max_cpe',
#                                                   'SafeBet_t1','LowBet_t1', 'HighBet_t1', 'Profit_t1',
#                                                   'TotalProfit_t1','GambleEV_t1', 'TrialEV_t1','CR_t1', 
#                                                   'choiceEV_t1', 'rpe_t1','cf_t1', 'max_cf_t1','cpe_t1', 'max_cpe_t1']):
#     '''
#     Args:
#     ------
#     raw_behav : (list) List of subject behavior dataframes(task_df).
#     cont_vars : (list) List of cols from all_behav to normalize (strings). Default is all continuous variables. 
#     '''

#     if drops_data:
#         drop_epochs_dict = {}
    
#     all_behav = []
    
#     for subj_df in raw_behav:
#         subj_id = subj_df.subj_id.unique().tolist()[0]
#         subj_df['CpeOnset'] = subj_df.DecisionOnset + 2.0
        
#         if drops_data:
#             drop_epochs = [] # these will be redundant 
#             drop_epochs.extend(list(subj_df[subj_df.keep_epoch == 'drop'].epoch))
#             drop_epochs.extend(list(subj_df.tail(1).epoch))
#             drop_epochs.extend(list(subj_df[np.isinf(subj_df.logRT)].epoch))
#             drop_epochs.extend(list(subj_df[(subj_df.Round==76)].epoch))
#             drop_epochs.extend(list(subj_df[(subj_df.cpe==0.0)].epoch))
#             # drop_epochs.append(subj_df[subj_df.keep_epoch_t1 == 'drop'].epoch)
#             # drop_epochs.extend(list(subj_df[np.isinf(subj_df.logRT_t1)].epoch))
           
#             #### save subj drop epochs 
#             drop_epochs_dict[subj_id] = np.unique(drop_epochs)
        
#         # actually drop bad epochs
#         subj_df = subj_df[subj_df.keep_epoch != 'drop']
#         subj_df = subj_df[subj_df.Round != 76]
#         subj_df = subj_df.drop(subj_df.tail(1).index)
#         # subj_df = subj_df[subj_df.keep_epoch_t1 != 'drop']
        
#         # normalize continuous variables after dropping bad trials 
#         for var in cont_vars:
#             subj_df[var] = norm_zscore(subj_df[var].values)
            
#         all_behav.append(subj_df.reset_index(drop=True))
        
#     all_behav = pd.concat(all_behav).reset_index(drop=True)
    
#     if drops_data: # return behav and drops info 
#         return all_behav, drop_epochs_dict
#     else:
#         return all_behav


# def format_all_behav(raw_behav,cont_vars=['SafeBet', 'LowBet', 'HighBet','Profit', 
#                                                   'TotalProfit','GambleEV', 'TrialEV', 'CR',
#                                                   'choiceEV','rpe', 'cf', 'max_cf', 'cpe', 'max_cpe',
#                                                   'SafeBet_t1','LowBet_t1', 'HighBet_t1', 'Profit_t1',
#                                                   'TotalProfit_t1','GambleEV_t1', 'TrialEV_t1','CR_t1', 
#                                                   'choiceEV_t1', 'rpe_t1','cf_t1', 'max_cf_t1','cpe_t1', 'max_cpe_t1']):
#     '''
#     Args:
#     ------
#     raw_behav : (list) List of subject behavior dataframes(task_df).
#     cont_vars : (list) List of cols from all_behav to normalize (strings). Default is all continuous variables. 
#     '''

#     all_behav = []
#     for subj_df in raw_behav:
#         subj_id = subj_df.subj_id.unique().tolist()[0]
#         subj_df = subj_df[subj_df.keep_epoch != 'drop']
#         subj_df = subj_df[subj_df.keep_epoch_t1 != 'drop']
#         subj_df = subj_df.drop(subj_df.tail(1).index)
#         for var in cont_vars:
#             subj_df[var] = norm_zscore(subj_df[var].values)
#         all_behav.append(subj_df.reset_index(drop=True))

#     return pd.concat(all_behav).reset_index(drop=True)
