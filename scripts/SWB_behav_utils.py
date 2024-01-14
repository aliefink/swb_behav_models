import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns



def clean_subj_data(subj_id,swb_dir='/Users/alexandrafink/Documents/GraduateSchool/SaezLab/SWB/'):
    #inputs:
    # subj_id - ID of SWB participant
    # swb_dir - location of behavior_analysis directory

    #get subj raw file names
    raw_data_dir = swb_dir + 'behavior_analysis/behavior_raw/' + subj_id + '/'
    save_dir = swb_dir + 'behavior_analysis/behavior_preprocessed/'
    raw_data_files = os.listdir(raw_data_dir)

    #load task df:
    raw_task_name = [x for x in raw_data_files if 'Rate' not in x if 'BDI' not in x if 'BAI' not in x if 'DS' not in x]
    raw_task = pd.read_table(raw_data_dir+raw_task_name[0])
    task = raw_task.drop(raw_task.tail(2).index) #remove empty rows
    task = task[task.columns.drop(list(task.filter(regex='Unnamed')))] #remove empty column
    task = task.rename(columns={'Feedback Onset': 'FeedbackOnset','Safe Bet':'SafeBet','Low Bet':'LowBet','High Bet':'HighBet',
                                'High Bet Pos':'HighBetPos','Gamble Pos':'GamblePos','Choice Pos':'ChoicePos','Gamble Choice':'GambleChoice'})
    task['GambleEV'] = ((task['LowBet'] + task['HighBet'])/2)

    CR = []
    choiceEV = []
    RPE = []
    nRPE = []
    pRPE = []
    totalCPE = [] #outcome - max possible (only matters for gamble trials)
    decisionCPE = [] #outcome - max possible of alternative decision (only matters for gamble trials)
    totalRegret = []
    totalRelief = []
    decisionRegret = []
    decisionRelief = []
    totalCF = [] #value of counterfactual outcome (max possible)
    decisionCF = [] #value of counterfactual outcome (max of unchosen decision)

    for t in range(len(task)):
        
        if task['GambleChoice'][t]=='gamble':
            CR.append(0)
            choiceEV.append(task['GambleEV'][t])
            RPE.append(task['Profit'][t]-task['GambleEV'][t])
            if task['Outcome'][t]=='good':
                totalRegret.append(0)
                decisionRegret.append(0)
                totalRelief.append(task['Profit'][t] - task['LowBet'][t]) #outcome - worst possible
                decisionRelief.append(task['Profit'][t] - task['SafeBet'][t]) #outcome - worse safe decision
                totalCPE.append(task['Profit'][t] - task['LowBet'][t]) #corrected - should be POSITIVE and profit/high - LOW not high
                decisionCPE.append(task['Profit'][t] - task['SafeBet'][t]) ### this is same as highbet - safebet should be POSITIVE
                totalCF.append(task['LowBet'][t]) # min possible! #### consider whether this should be positive or negative later
                decisionCF.append(task['SafeBet'][t]) # - max of unchosen
                pRPE.append(task['Profit'][t]-task['GambleEV'][t])
                nRPE.append(0)
            elif task['Outcome'][t]=='bad':
                totalRegret.append(task['Profit'][t] - task['HighBet'][t]) #outcome - best possible
                decisionRegret.append(task['Profit'][t] - task['SafeBet'][t]) #outcome - better safe decision
                totalRelief.append(0)
                decisionRelief.append(0)
                totalCPE.append(task['Profit'][t] - task['HighBet'][t]) # should be NEGATIVE same as low - high
                decisionCPE.append(task['Profit'][t] - task['SafeBet'][t]) # this is same as lowbet - safebet should be NEGATIVE
                totalCF.append(task['HighBet'][t])
                decisionCF.append(task['SafeBet'][t])
                pRPE.append(0)
                nRPE.append(task['Profit'][t]-task['GambleEV'][t])
            else: #fail trials
                totalRegret.append(0)
                decisionRegret.append(0)
                totalRelief.append(0)
                decisionRelief.append(0)
                totalCPE.append(0)
                decisionCPE.append(0)
                totalCF.append(0)
                decisionCF.append(0)
                pRPE.append(0)
                nRPE.append(0)

        elif task['GambleChoice'][t]=='safe':
            CR.append(task['SafeBet'][t])
            choiceEV.append(0)
            RPE.append(0)
            pRPE.append(0)
            nRPE.append(0)
            if task['Outcome'][t]=='good':
                totalRegret.append(0)
                decisionRegret.append(0)
                totalRelief.append(task['SafeBet'][t] - task['LowBet'][t]) #outcome - worst possible
                decisionRelief.append(task['SafeBet'][t] - task['LowBet'][t]) #no difference for safe trials  #######THIS WAS WRONG!! SHOULD GIVE POSITIVE VALUE!!!
                totalCPE.append(task['SafeBet'][t] - task['LowBet'][t]) #CORRECTED - GOOD & BAD TRIALS DO NOT HAVE SAME CPE CALCUATION!!!
                decisionCPE.append(task['SafeBet'][t] - task['LowBet'][t])
                totalCF.append(task['LowBet'][t])
                decisionCF.append(task['LowBet'][t])
            elif task['Outcome'][t]=='bad':
                totalRegret.append(task['SafeBet'][t] - task['HighBet'][t]) #outcome - best possible
                decisionRegret.append(task['SafeBet'][t] - task['HighBet'][t]) #no difference for safe trials 
                totalRelief.append(0)
                decisionRelief.append(0)
                totalCPE.append(task['SafeBet'][t] - task['HighBet'][t])
                decisionCPE.append(task['SafeBet'][t] - task['HighBet'][t])
                totalCF.append(task['HighBet'][t])
                decisionCF.append(task['HighBet'][t])
            else: #fail trials
                totalRegret.append(0)
                decisionRegret.append(0)
                totalRelief.append(0)
                decisionRelief.append(0)
                totalCPE.append(0)
                decisionCPE.append(0)
                totalCF.append(0)
                decisionCF.append(0)
        
        else: #fail trials
            CR.append(0)
            choiceEV.append(0)
            RPE.append(0)
            totalCPE.append(0)
            decisionCPE.append(0)
            totalRegret.append(0)
            decisionRegret.append(0)
            totalRelief.append(0)
            decisionRelief.append(0)
            totalCF.append(0)
            decisionCF.append(0)
            pRPE.append(0)
            nRPE.append(0)

    task['CR'] = CR
    task['choiceEV'] = choiceEV
    task['RPE'] = RPE
    task['totalCPE'] = totalCPE
    task['decisionCPE'] = decisionCPE 
    task['totalRegret'] = totalRegret
    task['decisionRegret'] = decisionRegret
    task['totalRelief'] = totalRelief
    task['decisionRelief'] = decisionRelief
    task['totalCF'] = totalCF
    task['decisionCF'] = decisionCF
    task['pRPE'] = pRPE
    task['nRPE'] = nRPE

    task.to_csv(f'{save_dir}{subj_id}_task_data',index=False)


    #mood rating info loading
    raw_mood_name = [x for x in raw_data_files if 'Rate' in x]
    raw_mood = pd.read_table(raw_data_dir+raw_mood_name[0])
    mood = raw_mood.drop(raw_mood.tail(2).index) #remove empty rows
    mood = mood[mood.columns.drop(list(mood.filter(regex='Unnamed')))] #remove empty column
    if (subj_id == 'DA8' or subj_id == 'DA9'): #mistakes in raw data column naming for these 2 subjs
        mood = mood[mood.columns.drop('RT','RatingOnset')]
        mood = mood.rename(columns={'Trial':'Rating','Type':'RatingOnset','Rating':'RT'})
    mood['zscore_mood'] = (mood['Rating']-mood['Rating'].mean())/mood['Rating'].std()
    


    #load swb bdi and bai info to add to mood df
    raw_bdi_name = [x for x in raw_data_files if 'BDI' in x] #not all subj have BDI
    if raw_bdi_name:
        raw_bdi = pd.read_table(raw_data_dir+raw_bdi_name[0]) 
        bdi = raw_bdi['BDI Score'].iloc[-1]   
    else:
        bdi = 0
    mood['bdi'] = bdi
    #load clinical neuropsych bdi info (often different from swb bdi)
    # subj_info = pd.read_excel(f'{swb_dir}SWB_subjects.xlsx', sheet_name='Master_List', usecols=[0,5])
    # bdi_np = subj_info.BDI[np.where(subj_info.PatientID == subj_id)[0]]
    # mood['bdi_neuropsych'] = list(bdi_np)*len(mood)

    #bai loading (not all subj have BAI)
    raw_bai_name = [x for x in raw_data_files if 'BAI' in x]
    if raw_bai_name:
        raw_bai = pd.read_table(raw_data_dir+raw_bai_name[0]) 
        bai = raw_bai['BAI Score'].iloc[-1]   
    else:
        bai = 0
    mood['bai'] = bai


    mood.to_csv(f'{save_dir}{subj_id}_rate_data',index=False)
        

    return task, mood


def norm_gamble_choices(df): #raw or preprocessed df works 
    
    #df is task data for a single subject
    loss_df = df[df.TrialType == 'loss']
    mix_df = df[df.TrialType == 'mix']
    gain_df = df[df.TrialType == 'gain']

    #loss
    loss_dict = {}
    loss_norm = -((loss_df['LowBet'] + loss_df['HighBet'])/2)/loss_df['SafeBet']
    loss_quant = np.quantile(loss_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    loss_x_axis = [np.mean(loss_quant[i:i+2],dtype=np.float64) for i in range(5)]
    loss_dec = loss_df['GambleChoice'].replace(['gamble','safe'],[2,1])
    loss_dec[(loss_dec != 1) & (loss_dec !=2)] = 0
    loss_zip = list(zip(loss_norm,loss_dec))
    loss_dict['loss_norm_evs'] = np.array(loss_norm)
    loss_dict['loss_choices'] = np.array(loss_dec)
    loss_dict['loss_x_axis'] = loss_x_axis
    loss_norm_range = []
    loss_choice_props = []
    for r in range(5):
        loss_ev_range = np.array([loss_quant[r],loss_quant[r+1]])
        loss_gamble_count = [z[1] for z in loss_zip if z[0] >= loss_ev_range[0] and z[0] <= loss_ev_range[1]]
        loss_ev_num = np.count_nonzero(np.array(loss_gamble_count)==2)
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
    mix_dec = mix_df['GambleChoice'].replace(['gamble','safe'],[2,1])
    mix_dec[(mix_dec != 1) & (mix_dec !=2)] = 0
    mix_zip = list(zip(mix_norm,mix_dec))
    mix_dict['mix_norm_evs'] = np.array(mix_norm)
    mix_dict['mix_choices'] = np.array(mix_dec)
    mix_dict['mix_x_axis'] = mix_x_axis
    mix_norm_range = []
    mix_choice_props = []
    for r in range(5):
        mix_ev_range = np.array([mix_quant[r],mix_quant[r+1]])
        mix_gamble_count = [z[1] for z in mix_zip if z[0] >= mix_ev_range[0] and z[0] <= mix_ev_range[1]]
        mix_ev_num = np.count_nonzero(np.array(mix_gamble_count)==2)
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
    gain_dec = gain_df['GambleChoice'].replace(['gamble','safe'],[2,1]) #make safe = 1, gamble = 2, nan/skip/other = 0
    gain_dec[(gain_dec != 1) & (gain_dec !=2)] = 0
    gain_zip = list(zip(gain_norm,gain_dec))
    gain_dict['gain_norm_evs'] = np.array(gain_norm)
    gain_dict['gain_choices'] = np.array(gain_dec)
    gain_dict['gain_x_axis'] = gain_x_axis
    gain_norm_range = []
    gain_choice_props = []
    for r in range(5):
        gain_ev_range = np.array([gain_quant[r],gain_quant[r+1]])
        gain_gamble_count = [z[1] for z in gain_zip if z[0] >= gain_ev_range[0] and z[0] <= gain_ev_range[1]]
        gain_ev_num = np.count_nonzero(np.array(gain_gamble_count)==2)
        gain_ev_prop = gain_ev_num/len(gain_gamble_count)
        gain_norm_range.append(gain_ev_range)
        gain_choice_props.append(gain_ev_prop)
    gain_dict['gain_norm_range'] = gain_norm_range
    gain_dict['gain_choice_props'] = gain_choice_props
    
    return loss_dict, mix_dict, gain_dict

def util_norm_gamble_choices(df):
    
    #df is task data for a single subject
    loss_df = df[df.TrialType == 'loss']
    mix_df = df[df.TrialType == 'mix']
    gain_df = df[df.TrialType == 'gain']

    #loss
    loss_dict = {}
    #replacing EV with utility
    loss_norm = -loss_df['util_g']/loss_df['util_s'] #util_g/util_s
    loss_quant = np.quantile(loss_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    loss_x_axis = [np.mean(loss_quant[i:i+2],dtype=np.float64) for i in range(5)]
    loss_dec = loss_df['GambleChoice'].replace(['gamble','safe'],[2,1])
    loss_dec[(loss_dec != 1) & (loss_dec !=2)] = 0
    loss_zip = list(zip(loss_norm,loss_dec))
    loss_dict['loss_norm_evs'] = np.array(loss_norm)
    loss_dict['loss_choices'] = np.array(loss_dec)
    loss_dict['loss_x_axis'] = loss_x_axis
    loss_norm_range = []
    loss_choice_props = []
    for r in range(5):
        loss_ev_range = np.array([loss_quant[r],loss_quant[r+1]])
        loss_gamble_count = [z[1] for z in loss_zip if z[0] >= loss_ev_range[0] and z[0] <= loss_ev_range[1]]
        loss_ev_num = np.count_nonzero(np.array(loss_gamble_count)==2)
        loss_ev_prop = loss_ev_num/len(loss_gamble_count)
        loss_norm_range.append(loss_ev_range)
        loss_choice_props.append(loss_ev_prop)
    loss_dict['loss_norm_range'] = loss_norm_range
    loss_dict['loss_choice_props'] = loss_choice_props
    
    #mix
    mix_dict = {}
    #replacing EV with utility
    mix_norm = mix_df['util_g'] #can't divide by zero, util gamble has weighted high and low bet already
    mix_quant = np.quantile(mix_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    mix_x_axis = [np.mean(mix_quant[i:i+2],dtype=np.float64) for i in range(5)]
    mix_dec = mix_df['GambleChoice'].replace(['gamble','safe'],[2,1])
    mix_dec[(mix_dec != 1) & (mix_dec !=2)] = 0
    mix_zip = list(zip(mix_norm,mix_dec))
    mix_dict['mix_norm_evs'] = np.array(mix_norm)
    mix_dict['mix_choices'] = np.array(mix_dec)
    mix_dict['mix_x_axis'] = mix_x_axis
    mix_norm_range = []
    mix_choice_props = []
    for r in range(5):
        mix_ev_range = np.array([mix_quant[r],mix_quant[r+1]])
        mix_gamble_count = [z[1] for z in mix_zip if z[0] >= mix_ev_range[0] and z[0] <= mix_ev_range[1]]
        mix_ev_num = np.count_nonzero(np.array(mix_gamble_count)==2)
        mix_ev_prop = mix_ev_num/len(mix_gamble_count)
        mix_norm_range.append(mix_ev_range)
        mix_choice_props.append(mix_ev_prop)
    mix_dict['mix_norm_range'] = mix_norm_range
    mix_dict['mix_choice_props'] = mix_choice_props

    
    #gain
    gain_dict = {}
    #gain_norm = ((gain_df['low_bet'] + gain_df['high_bet'])/2)/gain_df['safe_bet']
    #replacing EV with utility
    gain_norm = gain_df['util_g']/gain_df['util_s'] #util_g/util_s
    gain_quant = np.quantile(gain_norm,q=(0,0.2,0.4,0.6,0.8,1),axis=0)
    gain_x_axis = [np.mean(gain_quant[i:i+2],dtype=np.float64) for i in range(5)]
    gain_dec = gain_df['GambleChoice'].replace(['gamble','safe'],[2,1]) 
    gain_dec[(gain_dec != 1) & (gain_dec !=2)] = 0    
    gain_zip = list(zip(gain_norm,gain_dec))
    gain_dict['gain_norm_evs'] = np.array(gain_norm)
    gain_dict['gain_choices'] = np.array(gain_dec)
    gain_dict['gain_x_axis'] = gain_x_axis
    gain_norm_range = []
    gain_choice_props = []
    for r in range(5):
        gain_ev_range = np.array([gain_quant[r],gain_quant[r+1]])
        gain_gamble_count = [z[1] for z in gain_zip if z[0] >= gain_ev_range[0] and z[0] <= gain_ev_range[1]]
        gain_ev_num = np.count_nonzero(np.array(gain_gamble_count)==2)
        gain_ev_prop = gain_ev_num/len(gain_gamble_count)
        gain_norm_range.append(gain_ev_range)
        gain_choice_props.append(gain_ev_prop)
    gain_dict['gain_norm_range'] = gain_norm_range
    gain_dict['gain_choice_props'] = gain_choice_props
    
    return loss_dict, mix_dict, gain_dict

def get_model_data(subj_id,task_df,rate_df):
    # TO-DO pass in vec of desired param names to eliminate clunky code
    ######### TO DO - UPDATE TO INCLUDE CF CALCULATIONS
    model_data_dict = {}

    #get rating info
    round = rate_df['Round'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:] #need index of last round 1 because some pts have multiple round 1 scores, start after last round 1 index
    rate = rate_df['Rating'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]
    zscore_rate = rate_df['zscore_mood'][max(loc for loc, val in enumerate(rate_df['Round']) if val == 1)+1:]


    cr1 = []
    cr2 = []
    cr3 = []
    ev1 = []
    ev2 = []
    ev3 = []
    rpe1 = []
    rpe2 = []
    rpe3 = []
    tcpe1 = []
    tcpe2 = []
    tcpe3 = []
    dcpe1 = []
    dcpe2 = []
    dcpe3 = []
    treg1 = []
    treg2 = []
    treg3 = []
    dreg1 = []
    dreg2 = []
    dreg3 = []
    trel1 = []
    trel2 = []
    trel3 = []
    drel1 = []
    drel2 = []
    drel3 = []

    for r in round:
        #index for task df
        t3 = r-4 #t-3 trial 
        t2 = r-3 #t-2 trial
        t1 = r-2 #t-1 trial
        
        cr1.append(task_df['CR'][t1])
        cr2.append(task_df['CR'][t2])
        cr3.append(task_df['CR'][t3])
        ev1.append(task_df['choiceEV'][t1])
        ev2.append(task_df['choiceEV'][t2])
        ev3.append(task_df['choiceEV'][t3])
        rpe1.append(task_df['RPE'][t1])
        rpe2.append(task_df['RPE'][t2])
        rpe3.append(task_df['RPE'][t3])
        tcpe1.append(task_df['totalCPE'][t1])
        tcpe2.append(task_df['totalCPE'][t2])
        tcpe3.append(task_df['totalCPE'][t3])
        dcpe1.append(task_df['decisionCPE'][t1])
        dcpe2.append(task_df['decisionCPE'][t2])
        dcpe3.append(task_df['decisionCPE'][t3])
        treg1.append(task_df['totalRegret'][t1])
        treg2.append(task_df['totalRegret'][t2])
        treg3.append(task_df['totalRegret'][t3])
        dreg1.append(task_df['decisionRegret'][t1])
        dreg2.append(task_df['decisionRegret'][t2])
        dreg3.append(task_df['decisionRegret'][t3])
        trel1.append(task_df['totalRelief'][t1])
        trel2.append(task_df['totalRelief'][t2])
        trel3.append(task_df['totalRelief'][t3])
        drel1.append(task_df['decisionRelief'][t1])
        drel2.append(task_df['decisionRelief'][t2])
        drel3.append(task_df['decisionRelief'][t3])

    
    
    
    model_data_dict['subj_id'] = [subj_id]*50
    model_data_dict['round'] = round
    model_data_dict['rate'] = rate
    model_data_dict['zscore_rate'] = zscore_rate
    model_data_dict['cr(t-1)'] = cr1
    model_data_dict['cr(t-2)'] = cr2
    model_data_dict['cr(t-3)'] = cr3
    model_data_dict['choice_ev(t-1)'] = ev1
    model_data_dict['choice_ev(t-2)'] = ev2
    model_data_dict['choice_ev(t-3)'] = ev3
    model_data_dict['rpe(t-1)'] = rpe1
    model_data_dict['rpe(t-2)'] = rpe2
    model_data_dict['rpe(t-3)'] = rpe3
    model_data_dict['totalcpe(t-1)'] = tcpe1
    model_data_dict['totalcpe(t-2)'] = tcpe2
    model_data_dict['totalcpe(t-3)'] = tcpe3
    model_data_dict['decisioncpe(t-1)'] = dcpe1
    model_data_dict['decisioncpe(t-2)'] = dcpe2
    model_data_dict['decisioncpe(t-3)'] = dcpe3
    model_data_dict['totalregret(t-1)'] = treg1
    model_data_dict['totalregret(t-2)'] = treg2
    model_data_dict['totalregret(t-3)'] = treg3
    model_data_dict['decisionregret(t-1)'] = dreg1
    model_data_dict['decisionregret(t-2)'] = dreg2
    model_data_dict['decisionregret(t-3)'] = dreg3
    model_data_dict['totalrelief(t-1)'] = trel1
    model_data_dict['totalrelief(t-2)'] = trel2
    model_data_dict['totalrelief(t-3)'] = trel3
    model_data_dict['decisionrelief(t-1)'] = drel1
    model_data_dict['decisionrelief(t-2)'] = drel2
    model_data_dict['decisionrelief(t-3)'] = drel3

    
    
    return model_data_dict
    
    
def plot_gamble_choices(subj_id,raw_task):
    
    loss,mix,gain = norm_gamble_choices(raw_task)
    gamble_plot_data = {}
    gamble_plot_data['loss_x'] = loss['loss_x_axis']
    gamble_plot_data['loss_y'] = loss['loss_choice_props']

    gamble_plot_data['mix_x'] = mix['mix_x_axis']
    gamble_plot_data['mix_y'] =  mix['mix_choice_props']

    gamble_plot_data['gain_x'] = gain['gain_x_axis']
    gamble_plot_data['gain_y'] = gain['gain_choice_props']

    gamble_plot_data = pd.DataFrame(gamble_plot_data)
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12,5))
    sns.set_theme(style='ticks',font='Arial') #magma_r,gnuplot,paired,cubehelix_r
    sns.regplot(data=gamble_plot_data,ax=ax1,x='loss_x',y='loss_y',ci=None,logistic=True)
    sns.regplot(data=gamble_plot_data,ax=ax2,x='mix_x',y='mix_y',ci=None,logistic=True)
    sns.regplot(data=gamble_plot_data,ax=ax3,x='gain_x',y='gain_y',ci=None,logistic=True)
    ax1.set_ylim(0,1)
    ax1.set_ylabel(None)
    ax1.set_xlabel(None)
    ax1.tick_params(labelsize=12)
    ax1.set_title('Loss Trials',weight='medium',fontsize=15,y=1.02)
    ax2.set_ylim(0,1)
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)
    ax2.tick_params(labelsize=12)
    ax2.set_title('Mix Trials',weight='medium',fontsize=15,y=1.02)
    ax3.set_ylim(0,1)
    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.tick_params(labelsize=12)
    ax3.set_title('Gain Trials',weight='medium',fontsize=15,y=1.02)
    fig.supxlabel('Normalized Expected Value',y=-0.025,weight='medium',fontsize=16)
    fig.supylabel('Proportion of Gamble Choices',x=0.05,weight='medium',fontsize=16)
    fig.suptitle(subj_id,y=1.05,weight='semibold',fontsize=18)
    

    return fig
   
def plot_mood_ratings(subj_id,raw_mood):
    
    
    
    fig,ax = plt.subplots(figsize=(10,6))
    sns.set_theme(style='ticks',font='Arial') #pink_r/pink,rocket_r
    sns.regplot(x=np.arange(0,len(raw_mood['Rating'])), y=raw_mood['Rating'], ci=95, truncate=False)
    ax.set_title(subj_id,weight='semibold',fontsize=18,y=1.025)
    ax.set_xlabel('Round',weight='medium',fontsize=15)
    ax.set_ylabel('Normalized Happiness Rating',weight='medium',fontsize=15,x=0.025)
    ax.tick_params(labelsize=12)

    
    
    return fig






