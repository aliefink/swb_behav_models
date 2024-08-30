import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

class swb_subj_behav(object):

    def __init__(self, subj_id, behav_dir, output='all', save_dir=None, **kwargs):

        '''
        Args:
        - subj_id   : (str) SWB subj_id 
        - behav_dir : (str) Directory for all subject raw behavior files (not individual subj directory)
        - output    : (str) Data to output. Must be 'all','task','Rate','BDI','BAI'. Default is 'all' (outputs all data)
        - save_dir  : (str) Directory location to save preprocessed data (output arg will dictate which data saved). 
                            Default is None, which doesn't save the dataframes. 
        - **kwargs       : (optional)
        '''

        self.subj_id   = subj_id  # single electrode tfr data
        self.behav_dir = behav_dir
        self.output    = output # channel name for single electrode tfr data
        self.save_dir  = save_dir # single subject behav data

    def preprocess_behav(self):
        task_df = self.compute_task_vars(self.format_task_df(self.load_behav_file('task')))

        mood_df = self.format_mood_df(self.load_behav_file('Rate'))
        
        if self.output == 'all': #return task + mood dfs
            if self.save_dir: # save file if defined
                task_df.to_csv(f'{self.save_dir}{self.subj_id}_task_df.csv',index=False)
                mood_df.to_csv(f'{self.save_dir}{self.subj_id}_mood_df.csv',index=False)
            return task_df,mood_df
        elif self.output == 'task':  #return task df only
            if self.save_dir: # save file if defined
                task_df.to_csv(f'{self.save_dir}{self.subj_id}_task_df.csv',index=False)
            return task_df
        elif self.output == 'mood':  #return mood df only 
            if self.save_dir: # save file if defined
                mood_df.to_csv(f'{self.save_dir}{self.subj_id}_mood_df.csv',index=False)
            return mood_df  
        
    def load_behav_file(self,data_type):
        '''        
        Args:
        - data_type : (str) Type of data file to load. Must be one of 'task','Rate','BDI,'BAI'. 
        
        Returns:
        - df : (pd.DataFame) 
        '''
        raw_data_files = os.listdir(f'{self.behav_dir}{self.subj_id}/')

        if data_type =='task':
            file = [file for file in raw_data_files if 'Rate' not in file if 'BDI' not in file if 'BAI' not in file if 'DS' not in file][0]
        else:
            file = [file for file in raw_data_files if data_type in file][0]

        return pd.read_table(f'{self.behav_dir}{self.subj_id}/{file}') 

    def format_mood_df(self,mood_df,round_range=[4,151]):

        if 'Type' in mood_df.columns:
            mood_df = mood_df[['Round','Trial','Type','Rating']]
            mood_df = mood_df.rename(columns={'Trial':'Rating','Type':'RatingOnset','Rating':'RT'})
        
        mood_df = mood_df.drop(mood_df.tail(2).index) #remove empty rows
        # correct epochs check 
        if [int(mood_df.Round[mood_df.first_valid_index()]), int(mood_df.Round[mood_df.last_valid_index()])] != round_range:
            start_drops = np.arange(mood_df.first_valid_index(),mood_df.index[mood_df.Round.astype(int) == round_range[0]].values.astype(int)[0])
            end_drops   = np.arange(mood_df.index[mood_df.Round.astype(int) == round_range[1]].values.astype(int)[0]+1,mood_df.last_valid_index()+1)
            all_drops   = list(start_drops)+list(end_drops)
            mood_df = mood_df.drop(mood_df.loc[all_drops].index).reset_index(drop=True)
        # add bdi info mood_df
        mood_df['bdi'] = self.get_psych_score('BDI')
        mood_df['subj_id'] = self.subj_id
        mood_df['bdi_thresh'] = mood_df['bdi'].apply(lambda x: 'low' if x < 20 else 'high')

        # reordering df cols and removing unnecessary vars 
        mood_df = mood_df[['subj_id','bdi','bdi_thresh','Round','Rating','RatingOnset','RT']]
        mood_df['Round'] = mood_df.Round.astype(int)

        # add t-1,t-2,t-3 round indices to dataframe 
        mood_df['Round_t1_idx'] = mood_df.Round-1
        mood_df['Round_t2_idx'] = mood_df.Round-2
        mood_df['Round_t3_idx'] = mood_df.Round-3
        
        # add epoch (0 indexed round) and add t-1,t-2,t-3 epoch indices to dataframe 
        mood_df['epoch'] = mood_df.Round-1
        mood_df['epoch_t1_idx'] = mood_df.epoch-1
        mood_df['epoch_t2_idx'] = mood_df.epoch-2
        mood_df['epoch_t3_idx'] = mood_df.epoch-3
    
        return mood_df
    
    def format_task_df(self,task_df,keep_cols=None,round_range=[1,150]):
        task_df = task_df.drop(task_df.tail(2).index) #remove empty rows
        task_df = task_df.rename(columns={f'{col}':('').join(col.split(' ')) for col in task_df.columns})
        
        # correct epochs check 
        if [int(task_df.Round[task_df.first_valid_index()]), int(task_df.Round[task_df.last_valid_index()])] != round_range:
            start_drops = np.arange(task_df.first_valid_index(),task_df.index[task_df.Round.astype(int) == round_range[0]].values.astype(int)[0])
            end_drops   = np.arange(task_df.index[task_df.Round.astype(int) == round_range[1]].values.astype(int)[0]+1,task_df.last_valid_index()+1)
            all_drops   = list(start_drops)+list(end_drops)
            task_df = task_df.drop(task_df.loc[all_drops].index).reset_index(drop=True)
        
        task_df['bdi'] = self.get_psych_score('BDI')
        task_df['subj_id'] = self.subj_id
        task_df['bdi_thresh'] = task_df['bdi'].apply(lambda x: 'low' if x < 20 else 'high')
        
        # removing unnecessary data and reordering df 
        if keep_cols:
            task_df = task_df[keep_cols]
        else: 
            task_df = task_df[['subj_id','bdi','bdi_thresh','Round','RT','TrialOnset', 'ChoiceOnset','DecisionOnset', 'FeedbackOnset','ChoicePos','TrialType','SafeBet',
                            'LowBet', 'HighBet','GambleChoice', 'Outcome','Profit', 'TotalProfit']]
            
        return task_df
    
    def get_psych_score(self,psych_type):
    
        psych_df = self.load_behav_file(psych_type)

        return np.max(psych_df.rename(columns={f'{col}':col.split(' ')[-1] for col in psych_df.columns}).Score.values)
    
    def compute_task_vars(self,task_df):
        # add new vars to dataframe 
        task_df['epoch']    = task_df.Round.astype(int)-1
        task_df['logRT']    = np.log(task_df.RT)
        task_df['GambleEV'] = (task_df.LowBet + task_df.HighBet)/2
        task_df['TrialEV']  = (task_df.GambleEV+task_df.SafeBet)/2
        task_df['CR']       = [row['SafeBet'] if row['GambleChoice']=='safe' else 0.0 for ix,row in task_df.iterrows()]
        task_df['choiceEV'] = [row['GambleEV'] if row['GambleChoice']=='gamble' else 0.0 for ix,row in task_df.iterrows()]
        task_df['rpe']      = [row.Profit - row.GambleEV if row['GambleChoice']=='gamble' else 0.0 for ix,row in task_df.iterrows()]
        # start counterfactual computations
        cf_id_dict = {'gamble_good':'SafeBet','gamble_bad':'SafeBet','safe_good':'LowBet','safe_bad':'HighBet'}
        max_cf_id_dict = {'gamble_good':'LowBet','gamble_bad':'HighBet','safe_good':'LowBet','safe_bad':'HighBet'}
        # use dict info to find correct cf value
        task_df['res_type'] = list(map(lambda x,y: '_'.join([x,y]), task_df.GambleChoice.astype(str),task_df.Outcome.astype(str)))
        task_df['res_type'] = [res if res in list(cf_id_dict.keys()) else 'drop' for res in task_df.res_type]
        task_df['cf']       = [row[cf_id_dict[row.res_type]] if row.res_type in list(cf_id_dict.keys()) else 0.0 for ix,row in task_df.iterrows()]
        task_df['max_cf']   = [row[max_cf_id_dict[row.res_type]] if row.res_type in list(max_cf_id_dict.keys()) else 0.0 for ix,row in task_df.iterrows()]
        task_df['cpe']      = task_df['Profit'] - task_df['cf']
        task_df['max_cpe']  = task_df['Profit'] - task_df['max_cf']

        # compute t1 data and add to df  
        t1_var_list = list(task_df.columns.drop(['subj_id','bdi','bdi_thresh','TrialOnset','ChoiceOnset','DecisionOnset','FeedbackOnset']))
        for var in t1_var_list:
            var_t1 = '_'.join([var,'t1'])
            t1_data = task_df[var].tolist()[1:]+[np.nan]
            task_df[var_t1] = t1_data
            
        # mask for which trials to remove later!! 
        task_df['keep_epoch'] = ['keep' if cpe != 0.0 else 'drop' for cpe in task_df.cpe]
        task_df[task_df.RT < 0.3][['keep_epoch']] = 'drop'
        task_df[(task_df.Outcome!='good')&(task_df.Outcome!='bad')][['keep_epoch']] = 'drop'
        task_df[(task_df.GambleChoice!='gamble')&(task_df.GambleChoice!='safe')][['keep_epoch']] = 'drop'

        # mask for which t1 trials to remove later!! 
        task_df['keep_epoch_t1'] = ['keep' if cpe_t1 != 0.0 else 'drop' for cpe_t1 in task_df.cpe_t1]
        task_df[task_df.RT_t1 < 0.3][['keep_epoch_t1']] = 'drop'
        task_df[(task_df.Outcome_t1 !='good')&(task_df.Outcome_t1 !='bad')][['keep_epoch_t1']] = 'drop'
        task_df[(task_df.GambleChoice_t1 !='gamble')&(task_df.GambleChoice_t1 !='safe')][['keep_epoch_t1']] = 'drop'

        return task_df