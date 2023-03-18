# -*- coding: utf-8 -*-
"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 24/10/2022
"""
#%% Import packages and set directories.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import myokit
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
from scipy.stats import spearmanr, pearsonr
import os

# Set directories. Note, the directories need to be set correctly on your own device.
work_dir = os.getcwd() 
work_dir = os.path.join(work_dir, 'fill_in_your_paths', 'IonChannelTraffickingModel')
os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

# Import local functions.    
from TraffickingModelFunctions import single_chan, single_chan_viz, single_sumplot, diff_eq, determ_single_chan, sens_analysis, M_rates, S_rates, temp_APsim, APD_IKr_list, gating_sim, mutant_rates, drug_opt, EP_MT, drug_time, Varke_df, drug_effects, drug_func, APD_original, round_off, hypokalemia_opt, hypok_effects, determ_opt, graphpad_exp

# Set the formatting of the plots.
plt.rc('font', family = 'Arial')
plt.rc('xtick', labelsize = 16)
plt.rc('ytick', labelsize = 16)
plt.rc('axes', titlesize = 18, labelsize = 18)
plt.rc('legend', fontsize = 16)
plt.rc('figure', titlesize = 18) 
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['legend.title_fontsize'] = 18

# Reset default plotting parameters.
#plt.rcParams.update(plt.rcParamsDefault)

# To inspect the plots, you'd need to run the qt statement in the IPython console:
# Plot inline = %matplotlib inline
# Plot outside = %matplotlib qt
#%% Load the experimental data.

# Load the extracted data points from the experimental studies
# Note, Guo et al. has both quantified the protein density
# and the functionality
Ke_exp = pd.read_csv("Data/Ke GDE fig 2C.csv")
Kanner_exp = pd.read_csv("Data/Kanner GDE fig 4H.csv")
Guo_exp = pd.read_csv('Data/Guo GDE fig 4A.csv')
Guo_exp_func = pd.read_csv(('Data/Guo GDE fig 4B.csv'))
Apaja_exp = pd.read_csv('Data/Apaja GDE fig 2C HeLa.csv')
Foo_exp = pd.read_csv('Data/Foo GDE fig 3A.csv')
Shi_exp = pd.read_csv('Data/Shi GDE fig 5.csv')
Osterbur_exp = pd.read_csv('Data/OsterburFig2B.csv')
Osterbur_exp['y'] = Osterbur_exp['y']/max(Osterbur_exp['y'])
Osterbur_exp.to_csv('Data/OsterburFig2B_norm.csv')

# Load the internalization/recycling data from Dennis et al. 2011 Figure 6.
DennisInt = pd.read_csv('Data/Dennis6B.csv')
DennisInt['x'] = round(DennisInt['x'], 0)
DennisInt['y'] = DennisInt['y'] * 100
DennisRec = pd.read_csv('Data/Dennis6C.csv')
DennisRec['x'] = round(DennisRec['x'], 0)
DennisRec['y'] = DennisRec['y'] * 100

DennisInt_SE = pd.read_csv('Data/Dennis6B_SE.csv')
DennisInt_SE['x'] = round(DennisInt_SE['x'], 0)
DennisInt_SE['y'] = DennisInt_SE['y'] * 100
DennisInt_SE['y'] = DennisInt_SE.groupby(['x'])['y'].diff().abs()/2
DennisInt_SE['y'] = DennisInt_SE['y'] * np.sqrt(3) 
DennisInt_SE = DennisInt_SE.dropna(axis = 0).reset_index(drop = True)

DennisRec_SE = pd.read_csv('Data/Dennis6C_SE.csv')
DennisRec_SE['x'] = round(DennisRec_SE['x'], 0)
DennisRec_SE['y'] = DennisRec_SE['y'] * 100
DennisRec_SE['y'] = DennisRec_SE.groupby(['x'])['y'].diff().abs()/2
DennisRec_SE['y'] = DennisRec_SE['y'] * np.sqrt(3) 
DennisRec_SE = DennisRec_SE.dropna(axis = 0).reset_index(drop = True)

# Load the data from Apaja et al. 2013 Figure 4.
ApajaIntHeLa = pd.read_csv('Data/Apaja GDE Fig4A1.csv')
ApajaIntH9C2 = pd.read_csv('Data/Apaja GDE Fig4A2.csv')
ApajaRec = pd.read_csv('Data/Apaja GDE fig4B.csv')

ApajaHelaInt_SE = pd.read_csv('Data/Apaja GDE Fig4A1_SE.csv')
ApajaHelaInt_SE['y'] = ApajaHelaInt_SE.groupby(['x'])['y'].diff().abs()
ApajaHelaInt_SE['y'] = ApajaHelaInt_SE['y'] * np.sqrt(3) 
ApajaHelaInt_SE = ApajaHelaInt_SE.dropna(axis = 0).reset_index(drop = True)

ApajaH9C2Int_SE = pd.read_csv('Data/Apaja GDE Fig4A2_SE.csv')
ApajaH9C2Int_SE['y'] = ApajaH9C2Int_SE.groupby(['x'])['y'].diff().abs()
ApajaH9C2Int_SE['y'] = ApajaH9C2Int_SE['y'] * np.sqrt(3) 
ApajaH9C2Int_SE = ApajaH9C2Int_SE.dropna(axis = 0).reset_index(drop = True)

ApajaRec_SE = pd.read_csv('Data/Apaja GDE Fig4B_SE.csv')
ApajaRec_SE['y'] = ApajaRec_SE.groupby(['x'])['y'].diff().abs()
ApajaRec_SE['y'] = ApajaRec_SE['y'] * np.sqrt(3) 
ApajaRec_SE = ApajaRec_SE.dropna(axis = 0).reset_index(drop = True)

# Import the experimental mutant rates from Kanner et al. 2018.
KannerIntA57P= pd.read_csv('Data/KannerIntA57P.csv')
KannerIntA57P['x'] = round(KannerIntA57P['x'])
KannerIntA57P_SD = pd.read_csv('Data/KannerIntA57P_SD.csv')
KannerFWA57P = pd.read_csv('Data/KannerFWA57P.csv')
KannerFWA57P['x'] = round(KannerFWA57P['x'])
KannerMT_overview = pd.read_csv('Data/KannerMT.csv') 
KannerA57P = KannerMT_overview.iloc[2, :]
Kanner_exp_fw = pd.read_csv('Data/Kanner GDE fig 4G.csv')
Kanner_fw_WT = Kanner_exp_fw.copy()

# Load the experimental data from Varkevisser et al. 2013 & Asahi et al. 2019.
VarkeDof = pd.read_csv('Data/VarkevisserFig4DOF.csv')
VarkeSE = pd.read_csv('Data/VarkevisserFig4DOF_SE.csv')

# Transform into SD, where pentamidine is N = 24 and dofetilide N = 6.
VarkeSE['y'] = VarkeSE.groupby(['x'])['y'].diff().abs()
VarkeSE['y'][1] = VarkeSE['y'][1] * np.sqrt(24)
VarkeSE['y'][2:] = VarkeSE['y'][2:] * np.sqrt(6) 
VarkeSD = VarkeSE.dropna(axis = 0).reset_index(drop = True)

AsahiPA = pd.read_csv('Data/AsahiFig1Penta.csv')
AsahiSD = pd.read_csv('Data/AsahiFig1Penta_SD.csv')
AsahiSD['y'] = AsahiSD.groupby(['x'])['y'].diff().abs()
AsahiSD = AsahiSD.dropna(axis = 0).reset_index(drop = True)

# Load the data from Varkevisser et al. 2013 Figure 3.
VarkeF3 = pd.read_csv('Data/VarkevisserFig3DOF.csv')
VarkeF3_SE = pd.read_csv('Data/VarkevisserFig3DOF_SE.csv')
VarkeF3_SE['y'] = VarkeF3_SE.groupby(['x'])['y'].diff().abs()

# Transform into SD, where pentamidine is N = 24 and dofetilide N = 3.
VarkeF3_SE['y'][1] = VarkeF3_SE['y'][1] * np.sqrt(24)
VarkeF3_SE['y'][2:] = VarkeF3_SE['y'][2:] * np.sqrt(3) 
VarkeF3_SD = VarkeF3_SE.dropna(axis = 0).reset_index(drop = True)

# Load the Q10 data.
Q10_dataframe = pd.read_csv('Data/Q10_cali.csv')

# Load the experimental data from Amin et al. (2018) Figure 5B (N = 7).
AminMean = pd.read_csv('Data/AminFig5BMean.csv')
AminSEM = pd.read_csv('Data/AminFig5BSEM.csv')
AminSEM['y'] = AminSEM.groupby(['x'])['y'].diff().abs()
AminSEM['y'] = AminSEM['y'] * np.sqrt(7) 
AminSD= AminSEM.dropna(axis = 0).reset_index(drop = True)

# Load the temperature data on trafficking.
exp_Zhao37 = pd.read_csv('Data/Zhao fig1 37.csv')
exp_Zhao37['y'] = exp_Zhao37['y']/2207
exp_Zhao40 = pd.read_csv('Data/Zhao fig1 40.csv')
exp_Zhao40['y'] = exp_Zhao40['y']/2207
exp_Foo30 = pd.read_csv('Data/Foo Fig1.csv')
exp_Foo30 = pd.DataFrame({'x': [24], 'y': [exp_Foo30['y'][0]/100]})
exp_Foo41 = pd.read_csv('Data/FooFig5b.csv')
exp_Foo41['y'] = exp_Foo41['y']/100

# Load the hypokalemia data from Guo et al. (2009).
Guo1B = pd.read_csv('Data/GuoFig1BMean.csv')
Guo1C = pd.read_csv('Data/GuoFig1CMean_log.csv')
Guo1C['x'] = np.exp(Guo1C['x'])
Guo1D = pd.read_csv('Data/GuoFig1DMean_log.csv')
Guo1D['x'] = np.exp(Guo1D['x'])
Guo3E_5 = pd.read_csv('Data/GuoFig3EMean_5.csv')
Guo3E_0 = pd.read_csv('Data/GuoFig3EMean_0.csv')

# Load the SEM data from Guo et al. (2009).
Guo1BSEM = pd.read_csv('Data/GuoFig1BSEM.csv')
Guo1BSEM = round_off(Guo1BSEM)
Guo1BSEM['y'] = Guo1BSEM.groupby(['x'])['y'].diff().abs()/2
Guo1BSEM = Guo1BSEM.dropna(axis = 0).reset_index(drop = True)
Guo1B_N = [11, 11, 12, 20, 20, 14, 12, 24, 18]
Guo1B_SD = Guo1BSEM.copy()
Guo1B_SD['y'] = [Guo1B_SD['y'][i] * np.sqrt(Guo1B_N[i]) for i in range(len(Guo1B_SD['y']))]
Guo1B_SD['y'] = (Guo1B_SD['y']/(max(Guo1B['y'])))*100
Guo1B_SD.to_csv('Data/Guo1B_SD.csv', index = False)

Guo1CSEM = pd.read_csv('Data/GuoFig1CSEM.csv')
Guo1CSEM['y'] = Guo1CSEM.groupby(['x'])['y'].diff().abs()/2
Guo1CSEM = Guo1CSEM.dropna(axis = 0).reset_index(drop = True)
Guo1C_N = [7, 9, 15, 14, 13, 14]
Guo1C_SD = Guo1CSEM.copy()
Guo1C_SD['y'] = [Guo1C_SD['y'][i] * np.sqrt(Guo1C_N[i]) for i in range(len(Guo1C_SD['y']))]
Guo1C_SD['y'] = (Guo1C_SD['y']/Guo1C['y'].iloc[-1])*100
Guo1C_SD.to_csv('Data/Guo1C_SD.csv', index = False)

Guo1DSEM = pd.read_csv('Data/GuoFig1DSEM.csv')
Guo1DSEM['y'] = Guo1DSEM.groupby(['x'])['y'].diff().abs()/2
Guo1DSEM = Guo1DSEM.dropna(axis = 0).reset_index(drop = True)
Guo1D_N = [31, 30, 31, 30, 30]
Guo1D_SD = Guo1DSEM.copy()
Guo1D_SD['y'] = [Guo1D_SD['y'][i] * np.sqrt(Guo1D_N[i]) for i in range(len(Guo1D_SD['y']))]
Guo1D_SD['y'] = (Guo1D_SD['y']/Guo1D['y'].iloc[-1])*100
Guo1D_SD.to_csv('Data/Guo1D_SD.csv', index = False)

Guo3E_0SEM = pd.read_csv('Data/GuoFig3E_SEM0.csv')
Guo3E_0SEM['x'] = round(Guo3E_0SEM['x'])
Guo3E_0SEM['y'] = Guo3E_0SEM.groupby(['x'])['y'].diff().abs()/2
Guo3E_0SEM['y'] = Guo3E_0SEM['y'] * np.sqrt(4) 
Guo3E_0SD = Guo3E_0SEM.dropna(axis = 0).reset_index(drop = True)
Guo3E_0SD['y'] = Guo3E_0SD['y']/max(Guo3E_0['y'])*100
Guo3E_0SD.to_csv('Data/Guo3E_0SD.csv', index = False)

# Create an overview of the authors from the experimental studies.
Ke2013 = 'Ke et al. (2013)'
Kanner2018 = 'Kanner et al. (2018)'
Guo2009 = 'Guo et al. (2009)'
Apaja2013 = 'Apaja et al. (2013)'
Foo2019 = 'Foo et al. (2019)'
Osterbur2017 = 'Osterbur Badhey et al. (2017)'
#%% Optimized input data.

# Optimized trafficking rates (alpha, beta, delta, psi).
X = [6.56850075, 2.101663125, 0.599592, 423.26175750000004]

# Assign the rates. 
alpha_opt = X[0]
beta_opt = X[1]
delta_opt = X[2]
psi_opt = X[3]

# Optimized mutant trafficking rates.
MT_opt = [[4.39878871,   1.93333205,   0.65697293, 408.40110484],
          [7.98396124,   2.06770286,   0.81350517, 298.70774597],
          [6.71166945,   2.06613057,   0.97941628, 421.38040913],
          [6.83359678,   2.05937586,   0.62798972, 269.66811756]]


# Optimized drug effect parameters.
drug_results = [2.196566,0.730636,0.329207,6.628938,0.126642,0.525717,6,1]

# Optimnized hypokalemia parameters (overnight and week).
night_final = [7.249733,0.278542,2.895935,0.226791,1.000000,1.000000]
week_final = [7.249733,0.871920,2.691968,0.226791,1.000000,1.000000]
#%% Single channel transitions over time Figure 2A.
# Note, this simulation takes a long time (>20 minutes).

# Define the rate matrix with optimized rates.
mat_opt = np.array([[0, 0, 0], [delta_opt, 0, alpha_opt], [0, beta_opt, 0]])

# Time plan.
total_hr = 12.01
sec_step = 60/5 
dt_min = 1/60
dt_sim = dt_min/sec_step
t_sim_opt = np.arange(0, total_hr, dt_sim)

# Concatenate the steady state distributions and determine the number of channels.
Mocc = np.full(2207, 2)
Socc = np.ones(706)  
MS_conc = np.concatenate([Socc, Mocc])
MS_nch = len(MS_conc)

# Run the single channel simulation. 
sim_opt = determ_single_chan(mat = mat_opt, arr = MS_conc, nch = MS_nch, psi = psi_opt, t = t_sim_opt, dt = dt_sim, 
                             n = 3500, seed = True, plot = True)

# Subset the dataframe.
single_sim_opt = sim_opt['df']

# Create a list with hour indexes.
hr_l = [int(i) for i in np.arange(0, len(single_sim_opt), sec_step*60)]

# Obtain a long-formatted dataframe that contains the transitions and states per hour.
cmap = 'CMRmap_r'
interval = 300
dict_single_viz = single_chan_viz(x = single_sim_opt, interval = interval, hr_list = hr_l, cmap = cmap, bins = True, state_colors = False)

# Subset the total amount of transitions per channel for a lollipop chart.
tran_any = pd.DataFrame(sim_opt['tran_any']).loc[dict_single_viz['channels'].columns -4, 0].reset_index(drop = True)
tran_any = tran_any.rename('Total_tran')

# Merge the dataframe with the total amount of transitions per channel with the corresponding colors of the
# final state.
tran_color = pd.merge(tran_any, dict_single_viz['colors'], left_index = True, right_on = 'Channel')   
tran_color = tran_color.drop(['Channel'], axis = 1)     

# Merge the dataframes for plotting.
df_single_viz = pd.merge(dict_single_viz['tran_state'], tran_any, left_on = 'Channel', right_index = True)

# Map the state titles instead of the numerical values for plotting.
df_single_viz['State'] = pd.Categorical.from_codes(df_single_viz['State'].astype(int), ['Non-exisiting', 'Sub-membrane', 'Membrane'])

# Get the cumulative state distribution per hour.
cum_dist = single_sim_opt.iloc[hr_l, 4:single_sim_opt.shape[-1]:interval]
cum_dist.index = (cum_dist.index*dt_sim).astype(int)

# Initialize a DataFrame that can be used to store the cumulative distribution of states
# per hour.
dist_df = pd.DataFrame(index = cum_dist.index, columns = ['Non-exisiting', 'Sub-membrane', 'Membrane'])
dist_df['Non-exisiting'] = cum_dist[cum_dist == 0].count(axis = 1)
dist_df['Sub-membrane'] = cum_dist[cum_dist == 1].count(axis = 1) 
dist_df['Membrane'] = cum_dist[cum_dist == 2].count(axis = 1) 
dist_df = dist_df.reset_index().rename(columns = {'index':'Time'})
dist_long = pd.melt(dist_df, id_vars = 'Time', var_name = 'States', value_name = 'Value')

# Create a dictionary that contains the markers and the states.
legend_dict = {'Non-exisiting': 'o', 'Sub-membrane': 's', 'Membrane': '^'}
legend_dict = {'NE': 'o', 'S': 's', 'M': '^'}

# Plot the scatterplot with lollipop plot and histogram.
single_sumviz = single_sumplot(x = df_single_viz, total_tran = tran_color, colors = dict_single_viz['cpalette'], 
                               hist_df = dist_long, cmap = cmap, legend_dict = legend_dict, ticks = 10, work_dir = work_dir,
                               bins = True, save = True)      

# Subset the single channel data and index channels 3001 and 3600.
single_data = sim_opt['data']
single_data_time = single_data[:, 0]
single_data_state = single_data[:, 3 + 3001]
single_data_state2 = single_data[:, 3 + 3600]

# Plot the single channel transitions (#3001 and #3009).
fig, ax = plt.subplots(2, 1)
ax[0].plot(single_data_time, single_data_state, 'k')
ax[0].set_yticks([0, 1, 2])
ax[0].set_yticklabels(['NE/D', 'S', 'M'])
ax[0].set_xlabel('Time (hrs)')
ax[0].set_ylabel('States')
ax[0].set_title('Single channel transitions (#3001)')

ax[1].plot(single_data_time, single_data_state2, 'r')
ax[1].set_yticks([0, 1, 2])
ax[1].set_yticklabels(['NE/D', 'S', 'M'])
ax[1].set_xlabel('Time (hrs)')
ax[1].set_ylabel('States')
ax[1].set_title('Single channel transitions (#3600)')
fig.tight_layout()

# Export as csv to graphpad.
single_data = sim_opt['data']
single_data_time = single_data[:, 0]
single_data_state = single_data[:, 3 + 3001]
single_data_state2 = single_data[:, 3 + 3600]
single_data_df = {'time': single_data_time,
                  'state': single_data_state,
                  'state2': single_data_state2}
df_single = pd.DataFrame(single_data_df)
df_single.to_csv('Data/single_sim_data.csv', index = False)
#%% Recycling and internalization calibration Figure 2B. 

# Initialize y-values for both the sub-membrane and membrane states, respectively
# Based on Heijman et al. 2013, doi: https://doi.org/10.1371/journal.pcbi.1003202
y0 = [0, 2200]

# Define the time range in hours.
t_determ = np.arange(0, 240, 1)

# Define the time for the simulation. 
# Note, both the internalization and recycling happen within 1hr in Dennis et al. 2011,
# consequently, we will only model two hours after reaching steady state.
hr_sim = 1.51
dt_min = 1/60
sec_step = 60/3 
dt_sim = dt_min/sec_step
t_sim = np.arange(0, hr_sim, dt_sim)
 
# Obtain the index for the minutes.
min_index = [int(i) for i in np.arange(0, len(t_sim), sec_step)]

# Run the internalization and recycling simulation.
internal_recycle = determ_opt(rate_arr = X, y0 = y0, t_determ = t_determ, t = t_sim, dt = dt_sim, n = 10, 
                                       cont_int = DennisInt, cont_HeLa = ApajaIntHeLa, cont_H9C2 = ApajaIntH9C2,
                                       cont_rec = DennisRec, cont_recA = ApajaRec, min_index = min_index, min_step = sec_step, 
                                       HeLa = True, show = True)

# For convenience the internalization and recycling behaviour are also hard-coded below.
internal = [11.2, 21.9, 21.1, 19.6]
recycle =  [28.4, 54.6, 65.8] 

# Format the data for the visualization.
int_df = {'Percentage': [DennisInt.iloc[0, 1], DennisInt.iloc[1, 1], DennisInt.iloc[2, 1], 0,
                         0, 0, 0, ApajaIntHeLa.iloc[0, 1], 
                         0, 0, 0, ApajaIntH9C2.iloc[0, 1],
                         internal[0], internal[1], 
                         internal[2], internal[3]],
           'Input': ['Dennis et al. (2011)', 'Dennis et al. (2011)', 'Dennis et al. (2011)','Dennis et al. (2011)',
                     'Apaja et al. (2013) [HeLa]', 'Apaja et al. (2013) [HeLa]', 'Apaja et al. (2013) [HeLa]', 'Apaja et al. (2013) [HeLa]', 
                     'Apaja et al. (2013) [H9C2]',  'Apaja et al. (2013) [H9C2]',  'Apaja et al. (2013) [H9C2]',  'Apaja et al. (2013) [H9C2]', 
                     'Model', 'Model', 'Model', 'Model'],
           'Time': [5, 30, 60, 90,
                    5, 30, 60, 90, 
                    5, 30, 60, 90, 
                    5, 30, 60, 90],
           'Error': [DennisInt_SE['y'][0], DennisInt_SE['y'][1], DennisInt_SE['y'][2], 0,
                     0, 0, 0, ApajaHelaInt_SE['y'][0], 
                     0, 0, 0, ApajaH9C2Int_SE['y'][0],
                     0, 0, 0, 0]}

rec_df = {'Percentage': [DennisRec.iloc[0, 1], DennisRec.iloc[1, 1], DennisRec.iloc[2, 1], 
                         0, ApajaRec.iloc[0, 1], ApajaRec.iloc[3, 1],  
                         recycle[0], recycle[1], recycle[2]],
           'Input': ['Dennis et al. (2011)', 'Dennis et al. (2011)', 'Dennis et al. (2011)',
                     'Apaja et al. (2013)', 'Apaja et al. (2013)', 'Apaja et al. (2013)', 'Model', 'Model', 'Model'],
           'Time': [3, 10, 20, 3, 10, 20, 3, 10, 20],
           'Error': [DennisRec_SE['y'][0], DennisRec_SE['y'][1], DennisRec_SE['y'][2],
                     0, ApajaRec_SE['y'][0], ApajaRec_SE['y'][3],
                     0, 0, 0]}

int_rec_df = {'Percentage': [6.5, 14, 21.2,11.2, 22.4, 20.2, 65.1, 65.2, 67.5, 24.9, 56.5, 67.8],
              'Input': ['Dennis et al. (2011)', 'Dennis et al. (2011)', 'Dennis et al. (2011)', 'Model', 'Model', 'Model',
                        'Dennis et al. (2011)', 'Dennis et al. (2011)', 'Dennis et al. (2011)', 'Model', 'Model', 'Model'],
              'Time': ['Int. for 5 min', 'Int. for 30 min', 'Int. for 60 min', 
                       'Int. for 5 min', 'Int. for 30 min', 'Int. for 60 min',
                       'Rec. for 3 min', 'Rec. for 10 min', 'Rec. for 20 min', 
                       'Rec. for 3 min', 'Rec. for 10 min', 'Rec. for 20 min']}

# Create dataframes for internalization and recycling.
df_intrec = pd.DataFrame(int_rec_df)
df_int = pd.DataFrame(int_df)
df_int['Direction'] = 'Internalization'
df_int = df_int.sort_values(['Input', 'Time'])

df_rec = pd.DataFrame(rec_df)
df_rec['Direction'] = 'Recycling'
df_rec = df_rec.sort_values(['Input', 'Time'])

# Re-order the labels because not every time point was measured in the experimental studies.
order = {'Apaja et al. (2013) [H9C2]': 0, 'Model': 1, 'Dennis et al. (2011)': 2, 'Apaja et al. (2013) [HeLa]': 3}
order_int = df_int.sort_values(by=['Input'], key=lambda x: x.map(order))

# Plot the internalization and recycling results.
fig, ax = plt.subplots(1, 2, figsize=(7, 5))
ax1 = sns.barplot(x = 'Time', y = 'Percentage', hue = 'Input', data = order_int, ax = ax[0], 
                  palette = sns.color_palette('CMRmap_r', n_colors = len(np.unique(df_int['Input']))))
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax1.patches]
y_coords = [p.get_height() for p in ax1.patches]
ax[0].legend(loc = 'upper left')
ax[0].errorbar(x=x_coords, y=y_coords, yerr=order_int["Error"], fmt="none", c= "k")
ax[0].set_xlabel('Time [min]', fontweight = 'bold')
ax[0].set_ylabel('% of channels', fontweight = 'bold')
ax[0].set_title('Internalization', fontweight = 'bold')
ax[0].set_ylim(0,)

ax2 = sns.barplot(x = 'Time', y = 'Percentage', hue = 'Input', data = df_rec, ax = ax[1], palette = sns.color_palette('CMRmap_r', n_colors = len(np.unique(df_rec['Input']))))
x_coords2 = [p.get_x() + 0.5*p.get_width() for p in ax2.patches]
y_coords2 = [p.get_height() for p in ax2.patches]
ax[1].legend(loc = 'upper left')
ax[1].errorbar(x=x_coords2, y=y_coords2, yerr=df_rec["Error"], fmt="none", c= "k")
ax[1].set_xlabel('Time [min]', fontweight = 'bold')
ax[1].set_ylabel('% of channels', fontweight = 'bold')
ax[1].set_title('Recycling after 30 min.', fontweight = 'bold')
fig.tight_layout()
fig.savefig(work_dir + '\\Figures\\recintplot.svg', format='svg', dpi=1200, bbox_inches='tight')

# Export csv for graphpad.
df_rec.to_csv('Data/df_rec.csv')
order_int.to_csv('Data/order_int.csv')
#%% Forward trafficking block simulation Figure 2C.

# Load the model.
m_stoch = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Initialize a pacing protocol.
pace = myokit.Protocol()

# Set basic cycle length 100 seconds.
cl = 100*1000

# Create an event schedule.
pace.schedule(1, 20, 0.5, cl, 0)

# Create a simulation object.
sim = myokit.Simulation(m_stoch, pace)

# Set the initial values of M and S.
state = sim.state() 
new_state = state.copy()
Mem_index = m_stoch.get('ikr_trafficking.M').indice()
Sub_index = m_stoch.get('ikr_trafficking.S').indice()
new_state[Mem_index] = 2207
new_state[Sub_index] = 706
sim.set_state(new_state)

# Set psi to zero (block forward trafficking).
sim.set_constant('ikr_trafficking.pr', 0)

# Run the simulation for 24 hrs.
hrs24 = 24 * 3600000
run24 = sim.run(hrs24)

# Normalize the number of channels.
norm_run24M = np.array(run24['ikr_trafficking.M'])/max(run24['ikr_trafficking.M'])

# Repeat the same for total number of channels.
M_channels = np.array(run24['ikr_trafficking.M'])
S_channels = np.array(run24['ikr_trafficking.S'])
T_channels = M_channels + S_channels
norm_run24T = T_channels/max(T_channels)

# Create a marker list for visualization.
markers = ['D', '^', 's', '*', '.', '1']

# Format the dataframes for easier plotting.
all_exp = [Ke_exp, Guo_exp, Apaja_exp, Foo_exp, Shi_exp, Osterbur_exp]
author_list = [Ke2013, Guo2009, Apaja2013, Foo2019, 'Shi et al. (2015)', Osterbur2017]
color_list = ['blue', 'red', 'black', 'orange', 'purple', 'grey', 'pink']
for i in range(len(all_exp)):
    all_exp[i]['Author'] = author_list[i]
    all_exp[i]['Color'] = color_list[i]
df_allexp = pd.concat(all_exp).sort_values(by = 'x').drop_duplicates(subset = 'x').reset_index(drop = True)    
df_allexp['y'] = df_allexp['y']*100
list_to_drop = [Ke2013, Osterbur2017, 'Shi et al. (2015)']
df_onlyM = df_allexp[~df_allexp['Author'].isin(list_to_drop)]

# Create two color dictionaries.
color_dict = {name: color for name, color in zip(df_allexp['Author'], df_allexp['Color'])}
color_dictM = {name: color for name, color in zip(df_onlyM['Author'], df_onlyM['Color'])}

# Plot the results for M (experimental).
plt.figure(figsize = (7, 5))
plt.plot(np.array(run24['engine.time'])/3600000, norm_run24M * 100, color = 'green', label = 'Optimized parameters')
plt.plot(np.array(run24['engine.time'])/3600000, norm_run24T * 100, color = 'pink', label = 'Total')
sns.scatterplot(data = df_allexp, x = 'x', y = 'y', hue = 'Author', palette = color_dict, s = 70)
plt.xlabel('Time [hrs]')
plt.ylabel('% of channels in membrane')
plt.legend()
plt.title("Forward trafficking block (Ψ = 0)", fontweight = 'bold')
plt.tight_layout() 
plt.savefig(work_dir + '\\Figures\\fwblock.svg', format='svg', dpi=1200, bbox_inches='tight') 

# Export to csv for graphpad.
df_allexp.to_csv('Data/determ.csv')
engine_24hr = np.array(run24['engine.time'])/3600000
ind_post = np.arange(0, len(engine_24hr), round(len(engine_24hr)/100))
time_array = engine_24hr[ind_post]
value_array = norm_run24M[ind_post]
opt_data = {'time': time_array,
            'perc': value_array*100}
opti_params  = pd.DataFrame(opt_data)
opti_params.to_csv('Data/fit_determ.csv')
#%% Sensitivity analysis of Figure 3.

# Perform the sensitivity analysis by scaling the rates up and down. 
sens_arr = [4, 2, 1, 0.5, 0.25]
sens_arr_factor = [str(x) + 'x' for x in sens_arr]

# Initialize a list.
sens_list = list()

# Loop through the optimized rates and scale each rate.
for i, x in enumerate(X):
    L = list()
    for y in sens_arr:
        X_scale = X.copy()
        X_scale[i] = x * y
        L.append(X_scale)
    sens_list.append(L)

# Run the sensitivity analysis.
sens_results = sens_analysis(sens_list, model = m_stoch, M = 2207, S = 706, scale = sens_arr_factor, hrs = 240)
sens_alpha = sens_results['alpha'].reset_index(drop = True)
sens_beta = sens_results['beta'].reset_index(drop = True)
sens_delta = sens_results['delta'].reset_index(drop = True)
sens_psi = sens_results['psi'].reset_index(drop = True)

# Adjust the time to hours for the sensitivity results.
sens_alpha['Time'] = sens_alpha['Time']/3600000
sens_beta['Time'] = sens_beta['Time']/3600000
sens_delta['Time'] = sens_delta['Time']/3600000
sens_psi['Time'] = sens_psi['Time']/3600000

# Calculate and plot the change in M and S as a function of the rates.           
M_sens = M_rates(sens_results, sens_arr_factor)
S_sens = S_rates(sens_results, sens_arr_factor)

# Create a color palette.
S_pal = sns.color_palette('CMRmap_r', n_colors = len(np.unique(sens_alpha['Scaling'])))

# Plot the number of channels in the membrane over time for each scaled rate.
fig = plt.figure(figsize = (16,9))   
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 1, 3)

sns.lineplot(data = sens_alpha, x = 'Time', y = 'M', hue = 'Scaling', ax = ax1, palette = S_pal, legend = True)
han, lab = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
ax1.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax1.set_title('Alpha rate')

sns.lineplot(data = sens_beta, x = 'Time', y = 'M', hue = 'Scaling', ax = ax2, palette = S_pal, legend = False)    
ax2.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax2.set_title('Beta rate')

sns.lineplot(data = sens_delta, x = 'Time', y = 'M', hue = 'Scaling', ax = ax3, palette = S_pal, legend = False)  
ax3.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax3.set_title('Delta rate')
 
sns.lineplot(data = sens_psi, x = 'Time', y = 'M', hue = 'Scaling', ax = ax4, palette = S_pal, legend = False)    
ax4.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax4.set_title('Psi rate')

ax5.axis('off')
first_l = ax5.legend(handles = han, loc = "upper center", ncol = 7, frameon = False, title = 'Scaling factors')
fig.suptitle('Membrane-state sensitivity analysis', fontweight = 'bold')
fig.tight_layout()
fig.savefig(work_dir + '\\Figures\\Sens_M.svg', format='svg', dpi=1200, bbox_inches='tight')
        
# Plot the number of channels in the sub-membrane over time for each scaled rate.
fig = plt.figure(figsize = (16,9))   
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 1, 3)

sns.lineplot(data = sens_alpha, x = 'Time', y = 'S', hue = 'Scaling', ax = ax1, palette = S_pal, legend = True)
han, lab = ax1.get_legend_handles_labels()
ax1.get_legend().remove()
ax1.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax1.set_title('Alpha rate')

sns.lineplot(data = sens_beta, x = 'Time', y = 'S', hue = 'Scaling', ax = ax2, palette = S_pal, legend = False)    
ax2.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax2.set_title('Beta rate')

sns.lineplot(data = sens_delta, x = 'Time', y = 'S', hue = 'Scaling', ax = ax3, palette = S_pal, legend = False)  
ax3.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax3.set_title('Delta rate')
 
sns.lineplot(data = sens_psi, x = 'Time', y = 'S', hue = 'Scaling', ax = ax4, palette = S_pal, legend = False)    
ax4.set(xlabel  = 'Time [hrs]', ylabel = 'Number of channels')
ax4.set_title('Psi rate')

ax5.axis('off')
first_l = ax5.legend(handles = han, loc = "upper center", ncol = 7, frameon = False, title = 'Scaling factors')
fig.suptitle('Sub-membrane-state sensitivity analysis', fontweight = 'bold')
fig.tight_layout()
fig.savefig(work_dir + '\\Figures\\Sens_S.svg', format='svg', dpi=1200, bbox_inches='tight')

# Create a list with the scaling factors for export.
scaling_ind = ['4x', '2x', '1x', '0.5x', '0.25x']

# Export to csv for graphpad.
for i in scaling_ind:
    alpha = sens_alpha.loc[sens_alpha['Scaling'] == i].iloc[::25, :]
    beta = sens_beta.loc[sens_beta['Scaling'] == i].iloc[::25, :]
    delta = sens_delta.loc[sens_delta['Scaling'] == i].iloc[::25, :]
    psi = sens_psi.loc[sens_psi['Scaling'] == i].iloc[::25, :]
    
    alpha.to_csv('Data/alpha' + f'{i}' + '.csv', index = False)
    beta.to_csv('Data/beta' + f'{i}' + '.csv', index = False)
    delta.to_csv('Data/delta' + f'{i}' + '.csv', index = False)
    psi.to_csv('Data/psi' + f'{i}' + '.csv', index = False)
#%% Mutant effect optimization Figure 4A and 4B. 

# Initialize y-values for both the sub-membrane and membrane states, respectively
# Based on Heijman et al. 2013, doi: https://doi.org/10.1371/journal.pcbi.1003202
y0 = [0, 2200]

# Time to steady state.
day10 = 240 * 3600000

# Scale the parameters to mimic the effects of a mutation.
psi_mut = psi_opt * KannerA57P['y'] 
delta_mut = delta_opt * 1/KannerA57P['y'] 
alpha_mut = alpha_opt * KannerA57P['y'] 
beta_mut = beta_opt * 1/KannerA57P['y'] 

# We assume that the model behaviour is the ground truth and therefore we scale the MT as a relative
# difference w/r to the model WT on the basis of Kanner et al. (2018) Figure 4. 
MT_int = [(1-KannerIntA57P['y'][i])/(1-Kanner_exp['y'][i]) for i in range(len(Kanner_exp))]
MT_int = MT_int[1:]

# For forward trafficking the difference between the MT and WT is quite stable in Figure 4 from Kanner et al.
# This suggests a single process that is affected (either psi or alpha), consequently we averaged the relative
# difference and used that to decrease the amount of channels being trafficked for the MT.
MT_fw = [KannerFWA57P['y'][i]/Kanner_fw_WT['y'][i] for i in range(len(Kanner_fw_WT))]
MT_fw = MT_fw[1:]
MT_fw_avg = np.average(MT_fw)

# The protocol in Kanner et al. 2018 tracks forward trafficking and internalization for 1 hour.
t_hr_Kanner = 1.01
sec_step_Kanner = 60/5 
dt_Kanner = dt_min/sec_step_Kanner
t_sim_Kanner = np.arange(0, t_hr_Kanner, dt_Kanner)
t_determ = np.arange(0, 240, 1)

# Perform the deterministic simulations to obtain the steady state
sol_ode = odeint(diff_eq, y0, t_determ, args = (X[0], X[1], X[2], X[3]))
sol_ode = pd.DataFrame(np.round(sol_ode, 0))

# Subset the final state distributions to obtain steady-state behaviour.
sub_steady= np.ones(int(sol_ode.iloc[-1][0]))
mem_steady = np.full(int(sol_ode.iloc[-1][1]), 2)

# Concatenate the steady state distributions and determine the number of channels.
arr_conc = np.concatenate([sub_steady, mem_steady])
nch = len(arr_conc)

# Create a new matrix for the WT in the MT simulation.
mat_WT = np.array([[0, 0, 0], [delta_opt, 0, alpha_opt], [0, beta_opt, 0]])

# Run single_sim simulation for the WT condition.
WT_sim = determ_single_chan(mat = mat_WT, arr = arr_conc, nch = nch, psi = psi_opt, t = t_sim_Kanner, dt = dt_Kanner, 
                         n = 500, seed = True, plot = True)

# Subset the dataframe.
single_WT_sim = WT_sim['df']

# Subset the individual channels.
indv_ch = single_WT_sim.iloc[:, 4:]

# Create a list with the indexes at each hour.
min_list = [int(i) for i in np.arange(0, len(t_sim_Kanner), sec_step_Kanner)]

# Check which channels are in membrane state at steady state.
mem_begin = indv_ch.columns[indv_ch.iloc[0] == 2]
subNE_begin = indv_ch.columns[(indv_ch.iloc[0] == 1) | (indv_ch.iloc[0] == 0)]

# The internalization was evaluated according to Kanner et al. 2018.
int_10min = single_WT_sim.loc[min_list[10], mem_begin].index[single_WT_sim.loc[min_list[10], mem_begin] != 2]
int_20min = single_WT_sim.loc[min_list[20], mem_begin].index[single_WT_sim.loc[min_list[20], mem_begin] != 2]
int_40min = single_WT_sim.loc[min_list[40], mem_begin].index[single_WT_sim.loc[min_list[40], mem_begin] != 2]
int_60min = single_WT_sim.loc[min_list[60], mem_begin].index[single_WT_sim.loc[min_list[60], mem_begin] != 2]

# The forward trafficking was evaluated according to Kanner et al. 2018.
fw_10min = single_WT_sim.loc[min_list[10], subNE_begin].index[single_WT_sim.loc[min_list[10], subNE_begin] == 2]
fw_20min = single_WT_sim.loc[min_list[20], subNE_begin].index[single_WT_sim.loc[min_list[20], subNE_begin] == 2]
fw_40min = single_WT_sim.loc[min_list[40], subNE_begin].index[single_WT_sim.loc[min_list[40], subNE_begin] == 2]
fw_60min = single_WT_sim.loc[min_list[60], subNE_begin].index[single_WT_sim.loc[min_list[60], subNE_begin] == 2]

# Create a dataframe for internalization and forward trafficking.
intern_df_WT = pd.DataFrame(data = [[0, 0, 'Internalization begin'],
                                [len(int_10min), 10, 'Internalization for 10 min'],
                                [len(int_20min), 20, 'Internalization for 20 min'],
                                [len(int_40min), 40, 'Internalization for 40 min'],
                                [len(int_60min), 60, 'Internalization for 60 min']],
                         columns = ['Nch', 'Idx', 'Condition'])

forward_df_WT = pd.DataFrame(data = [[0, 0, 'Forward trafficking begin'],
                                 [len(fw_10min), 10, 'Forward trafficking for 10 min'],
                                 [len(fw_20min), 20, 'Forward trafficking for 20 min'],
                                 [len(fw_40min), 40, 'Forward trafficking for 40 min'],
                                 [len(fw_60min), 60, 'Forward trafficking for 60 min']],
                         columns = ['Nch', 'Idx', 'Condition'])
    
# Initialize  WT dataframes for internalization and forward trafficking. 
Kanner_WT_int = Kanner_exp.copy()
Kanner_WT_int['x'] = [0, 10, 20, 40, 60]
Kanner_WT_fw = Kanner_fw_WT.copy()
Kanner_WT_fw['x'] = [0, 10, 20, 40, 60]

# Normalize the MT forward trafficking data with respect to WT.
KannerFWA57P_N = KannerFWA57P.copy()
KannerFWA57P_N['y'] = KannerFWA57P_N['y']/max(Kanner_WT_fw['y'])
Kanner_WT_fw['y'] = Kanner_WT_fw['y']/max(Kanner_WT_fw['y'])

# Calculate the target amount of channels in membrane.
target_MT = round(len(mem_steady) * KannerA57P['y'], 0)

# Initialize a list to store the best fits. 
MT_rates = list()

# Run the optimization and fit the best configuration.
for i in range(len(MT_opt)):
    MT_rates.append(mutant_rates(MT_opt[i], y0 = y0, t_determ = t_determ, t_sim = t_sim_Kanner, sec_step = sec_step_Kanner, dt = dt_Kanner, intern_df_WT = intern_df_WT,
                        forward_df_WT = forward_df_WT, MT_int = MT_int, MT_fw = MT_fw, target = target_MT, Kanner_MT_int = KannerIntA57P, 
                        Kanner_MT_fw = KannerFWA57P_N, Kanner_WT_int = Kanner_WT_int, Kanner_WT_fw = Kanner_WT_fw,
                        int_steady = len(mem_begin), fw_steady = len(subNE_begin), show = True, plot = True, return_df = True, fw_mt = True))

# Initialize a list with parameter names and colors.
param_names = ['a', 'b', 'd', 'p']
MT_colors = sns.color_palette('CMRmap_r', n_colors = len(MT_rates))

# Plot the internalization and forward rates.
fig, ax = plt.subplots(1, 2, figsize=(7, 5))
for i in range(len(MT_rates)):
    ax[0].plot(intern_df_WT['Idx'], (MT_rates[i]['mem_begin'] - MT_rates[i]['intern']['Nch'])/MT_rates[i]['mem_begin'], color = MT_colors[i], label = f"{param_names[i]} = {round(MT_rates[i]['error'], 2)}")
ax[0].plot(intern_df_WT['Idx'], (len(mem_begin) - intern_df_WT['Nch'])/len(mem_begin), 'k', label = 'Model WT') 
ax[0].plot(KannerIntA57P['x'], KannerIntA57P['y'], 'or', label = 'Exp. MT')
ax[0].plot(Kanner_WT_int['x'], Kanner_WT_int['y'], 'ok', label = 'Exp. WT')
ax[0].set_xlabel('Time [mins]')
ax[0].set_ylabel('Fraction')
ax[0].set_title('Membrane stability', fontweight = 'bold')

for i in range(len(MT_rates)):
    ax[1].plot(forward_df_WT['Idx'], MT_rates[i]['forward']['Nch']/max(forward_df_WT['Nch']), color = MT_colors[i], label = f"{param_names[i]} = {round(MT_rates[i]['error'], 2)}")
ax[1].plot(forward_df_WT['Idx'], forward_df_WT['Nch']/max(forward_df_WT['Nch']), 'k', label = 'Model WT') 
ax[1].plot(KannerFWA57P_N['x'], KannerFWA57P_N['y'], 'or', label = 'Exp. MT')
ax[1].plot(Kanner_WT_fw['x'], Kanner_WT_fw['y'], 'ok', label = 'Exp. WT')
ax[1].set_xlabel('Time [mins]')
ax[1].set_ylabel('Fraction')
ax[1].set_title('Forward trafficking', fontweight = 'bold')
ax[1].legend()
fig.tight_layout()

# Export csv for graphpad.
label_mut_inv = ['a', 'b', 'd', 'p']
for i in range(len(MT_rates)):
    time_mem = intern_df_WT['Idx']
    mem_stabMT = (MT_rates[i]['mem_begin'] - MT_rates[i]['intern']['Nch'])/MT_rates[i]['mem_begin']
    mem_stabWT = (len(mem_begin) - intern_df_WT['Nch'])/len(mem_begin)
    
    time_fw = forward_df_WT['Idx']
    fw_trafMT =  MT_rates[i]['forward']['Nch']/max(forward_df_WT['Nch'])
    fw_trafWT =  forward_df_WT['Nch']/max(forward_df_WT['Nch'])
    
    data_stab = {'Time': time_mem,
                 'MT': mem_stabMT,
                 'WT': mem_stabWT}
    
    data_fw = {'Time': time_fw,
               'MT': fw_trafMT,
               'WT': fw_trafWT}
    
    df_stab = pd.DataFrame(data_stab)
    df_traf = pd.DataFrame(data_fw)
    
    df_stab.to_csv('Data/mem_stab' + f'{label_mut_inv[i]}' + '.csv', index = False)
    df_traf.to_csv('Data/fw_traf' + f'{label_mut_inv[i]}' + '.csv', index = False)
    KannerIntA57P.to_csv('Data/Kanner_int_mutinv.csv', index = False)
    KannerFWA57P_N.to_csv('Data/Kanner_fw_mutinv.csv', index = False)
    Kanner_WT_int.to_csv('Data/Kanner_WTINT.csv', index = False)
    Kanner_WT_fw.to_csv('Data/KannerWTFW.csv', index = False)
#%% Total amount of membrane channels for each mutant parameter set and the MT APs as shown in Figures 4C & 4D. 

# Load the model again and the original ORd.
model = myokit.load_model('MMT/ORD_Temp_final.mmt')
model_org = myokit.load_model('MMT/ORD_ORG.mmt')

# Initialize a pacing protocol.
pace_mt = myokit.Protocol()

# Set the basic cycle length.
bcl = 60000

# Create an event schedule.
pace_mt.schedule(1, 20, 0.5, bcl, 0)

# Subset the rates for the MT.
XMT = [MT_opt[0], MT_opt[1], MT_opt[2], MT_opt[3]]

# List with MT parameters used.
MT_names = ['Alpha', 'Beta', 'Delta', 'Psi']

# Visualize the EP consequences of the best fit mutation.
EP_MT_res = EP_MT(XMT = XMT, XWT = X, model = model, model_org = model_org, scalar = 0.65, prot = pace_mt, modeltype = 1, bcl = bcl, MT_names = MT_names, work_dir = work_dir, save = True)

# Export csv for graphpad.
for i in range(len(MT_names)):
    time_org = EP_MT_res['data_org']['engine.time']
    mem_org = EP_MT_res['data_org']['membrane.V']
    
    time_org_down = EP_MT_res['data_org_down']['engine.time']
    mem_org_down = EP_MT_res['data_org_down']['membrane.V']
    
    time_WT = EP_MT_res['data_WT']['engine.time']
    mem_WT = EP_MT_res['data_WT']['membrane.V']
    
    time_MT = EP_MT_res['data_MT'][i]['engine.time']
    mem_MT = EP_MT_res['data_MT'][i]['membrane.V']
    
    org_data = {'Time': time_org,
                'vM': mem_org}
    
    org_down_data = {'Time': time_org_down,
                     'vM': mem_org_down}
    
    WT_data = {'Time': time_WT,
               'vM': mem_WT}
    
    MT_data = {'Time': time_MT,
               'vM': mem_MT}
    
    df_org = pd.DataFrame(org_data)
    df_org_down = pd.DataFrame(org_down_data)
    df_WT = pd.DataFrame(WT_data)
    df_MT = pd.DataFrame(MT_data)
    
    df_org.to_csv('Data/EP_org.csv', index = False)
    df_org_down.to_csv('Data/EP_org_down.csv', index = False)
    df_WT.to_csv('Data/EP_WT.csv', index = False)
    df_MT.to_csv('Data/EP_MT' + f'{MT_names[i]}' + '.csv', index = False)
#%% Model the concentration dependence of dofetilide and pentamidine as shown in Figures 5A and 5B. 

# Visualize the concentration dependence between pentamidine and dofetilide based on experimental data from Asahi et al. 2019 and Varkevisser et al. 2013.
Penta_Dof = drug_opt(x = drug_results, rates = X, PV = 10, DA = 1e-15, varkevisser = VarkeDof, SD_V = VarkeSD,
                     asahi = AsahiPA, SD_A = AsahiSD, TV = 48, TA = 24, TP = 48, y0 = [706, 2207], work_dir = work_dir,
                     total_V = True, total_A = False, SD_error = False, save = True, plot = True, 
                     show = True, return_df = True)

# Export csv for graphpad.
Penta_Dof['df_A'].to_csv('Data/drug_opt_A.csv', index = False)
Penta_Dof['df_V'].to_csv('Data/drug_opt_V.csv', index = False)
#%% Model the temporal (rescueing) effects of dofetilide as shown in Figure 5C.

# Load the model.
model = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Initialize a pacing protocol.
pace_prot = myokit.Protocol()

# Define a bcl.
bcl = 60000

# Create an event schedule.
pace_prot.schedule(1, 20, 0.5, bcl, 0)

# Calculate the temporal dynamics of the drug effects.
bcl_hr = 60000
dt_results = drug_time(x = drug_results, rates = X, model = model, prot = pace_prot, incub = 48, D = 1, P = 10, T = 8, bcl = bcl_hr)
dt_ikr = pd.DataFrame(np.transpose(dt_results['ikr']), columns = ['ikr']).reset_index().rename(columns = {'index':'Time'})
dt_ikr['Time'] = dt_ikr['Time']/60

# Reformat the data to create plot from Varkevisser. 
Varke_p1 = Varke_df(dt_results, ref = 2913, T = [0, 1, 2, 4, 6, 8])

# Concatenate the simulation data with experimetal data.
Varkmem_p1 = pd.concat([pd.DataFrame(Varke_p1['mem']), VarkeF3['y']]).reset_index(drop = True)
Varktot_p1 = pd.concat([pd.DataFrame(Varke_p1['tot']), VarkeF3['y']]).reset_index(drop = True)
Varksd_p1 = pd.concat([pd.DataFrame(np.zeros(6)), VarkeF3_SD['y']]).reset_index(drop = True)

# Create a dataframe for barchart plotting.
sim_p1 = {'Time': [0, 1, 2, 4, 6, 8,
                     0, 1, 2, 4, 6, 8],
            'Mem': Varkmem_p1.iloc[:, 0],
            'Total': Varktot_p1.iloc[:, 0],
            'Data': ['Sim', 'Sim', 'Sim', 'Sim', 'Sim', 'Sim',
                     'Exp', 'Exp', 'Exp', 'Exp', 'Exp', 'Exp'],
            'SD': Varksd_p1.iloc[:, 0]}
data_Vp1 = pd.DataFrame(sim_p1)

# Plot the barchart for the temporal effects of dofetilide.
plt.figure()
ax1 = sns.barplot(x = 'Time', y = 'Total', hue = 'Data', data = data_Vp1,
                  palette = sns.color_palette('CMRmap_r', n_colors = len(np.unique(data_Vp1['Data']))))
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax1.patches]
y_coords = [p.get_height() for p in ax1.patches]
h, l = ax1.get_legend_handles_labels()
plt.legend(h,['Model', 'Varkevisser et al. 2013'] , loc = 'upper left')
plt.errorbar(x = x_coords, y = y_coords, yerr = data_Vp1["SD"], fmt = "none", c = "k")
plt.xlabel('Time [hrs]', fontweight = 'bold')
plt.ylabel('% of channels', fontweight = 'bold')
plt.ylim(0,)

# Export csv for graphpad.
data_Vp1.to_csv('Data/Drug_time.csv', index = False)
#%% Model the acute blocking and trafficking effects of supraphysiological and clinically relevant dosages of dofetilide as in Figures 5D and 5E.

# Create an hour list.
hr_list_dof = [20, 25, 47, 50]

# Simulate and plot the effects of dofetilide on IKr and channels (Suprahysiological 1 umol/L) without pentamidine.
dof_effect = drug_effects(x = drug_results, rates = X, model = model, prot = pace_prot, D = 1, P = 0, T = 72, hr_list = hr_list_dof,
                          bcl = 3600000, work_dir = work_dir, title = 'dof_effect', total = False, plot = True, save = True)

# Simulate and plot the effects of dofetilide on IKr and channels (Suprahysiological 1 umol/L) with pentamidine (5 umol/L).
dof_effect_P = drug_effects(x = drug_results, rates = X, model = model, prot = pace_prot, D = 1, P = 5, T = 72, hr_list = hr_list_dof,
                          bcl = 3600000, work_dir = work_dir, title = 'dof_effect', total = False, plot = True, save = True)

# Perform a simulation which is clinically more relevant.
# Given that we didn't built in a mechanism that can mimic the pharmacokinetics, we decided to take the average of Fig 2a in Allen et al.
# 2002 (doi: https://doi-org.mu.idm.oclc.org/10.1046/j.1365-2125.2000.00243.x) where patients were dosed (500 ug) twice per day. 
Allen1D = pd.read_csv('Data/AllenFig2_1d.csv')
Allen1D_mean = np.mean(Allen1D['y'])
Allen1D_median = np.median(Allen1D['y'])

# Dofetilide molecular weight.
dof_weight = 441.565

# Determine the amount of umol/L.
clinical_dose = round(Allen1D_mean * (1/dof_weight), 4)

# Re-run simulation with this clinical dose.
dof_clinical = drug_effects(x = drug_results, rates = X, model = model, prot = pace_prot, D = clinical_dose, P = 0, T = 72, hr_list = hr_list_dof,
                            bcl = 3600000, work_dir = work_dir, title = 'clinical_dose', total = False, plot = True, save = False)

dof_clinical_P = drug_effects(x = drug_results, rates = X, model = model, prot = pace_prot, D = clinical_dose, P = 5, T = 72, hr_list = hr_list_dof,
                            bcl = 3600000, work_dir = work_dir, title = 'clinical_dose', total = False, plot = True, save = False)

# Exort csv for graphpad.
dof_clinical['df'].to_csv('Data/Drug_effects.csv', index = False)
dof_clinical_P['df'].to_csv('Data/Drug_effectsP.csv', index = False)
dof_effect['df'].to_csv('Data/Drug_effects1um.csv', index = False)
dof_effect_P['df'].to_csv('Data/Drug_effects1umP.csv', index = False)
            
dof_time = graphpad_exp(dof_clinical, 1000)      
dof_timeP = graphpad_exp(dof_clinical_P, 1000)   
dof_time1um = graphpad_exp(dof_effect, 1000)  
dof_time1umP = graphpad_exp(dof_effect_P, 1000)  


for i in range(len(dof_time)):
    dof_time[i].to_csv('Data/dof' + f'{hr_list_dof[i]}' + '.csv', index = False)

for i in range(len(dof_timeP)):
    dof_timeP[i].to_csv('Data/dof_P_' + f'{hr_list_dof[i]}' + '.csv', index = False)
    
for i in range(len(dof_time1um)):
    dof_time1um[i].to_csv('Data/dof1um' + f'{hr_list_dof[i]}' + '.csv', index = False)

for i in range(len(dof_time1umP)):
    dof_time1umP[i].to_csv('Data/dof1um_P_' + f'{hr_list_dof[i]}' + '.csv', index = False)
#%% Temperature dependent-regulation of IKr gating as shown in Figure 6. 

# Format a dataframe for plotting of Q10s and V1/2.
Q10_V = {'Value':[6.25, 4.54, Q10_dataframe.iloc[0, 1], 1.36, 2.56, Q10_dataframe.iloc[0, 2], 
                  3.55, 3.21, Q10_dataframe.iloc[0, 3], 3.65, 3.45, Q10_dataframe.iloc[0, 4], 
                  13.9, 10.3, Q10_dataframe.iloc[0, 5]], 
        'Input':['Zhou et al. (1998)', 'Mauerhöfer & Bauer (2016)', 'Model',
                 'Zhou et al. (1998)', 'Mauerhöfer & Bauer (2016)', 'Model',
                 'Zhou et al. (1998)', 'Mauerhöfer & Bauer (2016)', 'Model',
                 'Zhou et al. (1998)', 'Mauerhöfer & Bauer (2016)', 'Model',
                 'Zhou et al. (1998)', 'Mauerhöfer & Bauer (2016)', 'Model'],
        'Q10': ['Activation', 'Activation', 'Activation', 
                 'Deactivation', 'Deactivation', 'Deactivation', 
                 'Inactivation', 'Inactivation', 'Inactivation', 
                 'Recovery', 'Recovery', 'Recovery',
                 '$V_{1/2}$', '$V_{1/2}$', '$V_{1/2}$'],
        'error': [2.55, 1.24, 0, 0.4, 0.97, 0, 0.87, 0.7, 0, 0.73, 0.62, 0, 0, 0, 0]}

# Create a new dataframe.
Q10df = pd.DataFrame(Q10_V)
Q10df = Q10df.sort_values(['Input', 'Q10'])

# Plot the Q10 and V1/2 calibration.
plt.figure(figsize=(8, 5))
ax = sns.barplot(x = 'Q10', y = 'Value', data = Q10df, hue = 'Input', palette = sns.color_palette('CMRmap_r', n_colors = 3))
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=Q10df["error"], fmt="none", c= "k")
plt.legend()
plt.xlabel('')
plt.tight_layout()

# Load the model.
model = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Create a temperature list to evaluate.
temp_list = [37, 40, 30, 41]
temp_list_amin = [23, 35, 40]

# Create a color palette.
sea_palette = ["#dfba1f", "#e35341", "#471cc7", 'green']

# Define a voltage-step protocol from Amin et al. (2008).
vstep = myokit.Protocol()
vstep.add_step(-80, 500)
vstep.add_step(20, 5000)
vstep.add_step(-50, 3000)
vstep.add_step(-80, 500)

# Total time of the protocol.
t_tot = 9000

# Define a list for trimming. 
trim = [501, 4999, 5501, 3000]

# Run the simulation.
gating = gating_sim(m = model, vprot = vstep, trim = trim, t_tot = t_tot, temp_list = temp_list_amin, color = sea_palette, work_dir = work_dir, save = False)

# Normalize the tails relative to the 35 degrees.
Amin35 = gating['tail'][1]
tail_normAmin = [i/Amin35 for i in gating['tail']]

# Normalize the experimental data from Amin et al. (2008).
Amin35exp = AminMean.iloc[1, 1]
Amin35SD = AminSD.iloc[1, 1]

# Normalize the experimental data from Amin et al. (2008).
tail_normAmin_exp = [i/Amin35exp for i in AminMean['y']]
tail_normAmin_SD = [i/Amin35exp for i in AminSD['y']]

# Reformat to a dataframe.
tail_data = {'tail': tail_normAmin + tail_normAmin_exp,
             'sd': [0, 0, 0] + tail_normAmin_SD,
             'temp': temp_list_amin + temp_list_amin,
             'type': ['Model', 'Model', 'Model', 'Exp.', 'Exp.', 'Exp.']}
tail_df = pd.DataFrame(tail_data)
tail_df['tail'] = tail_df['tail']*100
tail_df['sd'] = tail_df['sd'] * 100

# Plot the relative difference in peak tail current for each temperature.
fig, ax = plt.subplots()
ax1 = sns.barplot(x = 'temp', y = 'tail', hue = 'type', data = tail_df, palette = sea_palette, ax = ax)
x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax1.patches]
y_coords = [p.get_height() for p in ax1.patches]
ax.legend(loc = 'upper left')
ax.errorbar(x=x_coords, y=y_coords, yerr=tail_df["sd"], fmt="none", c= "k")
ax.set_xlabel('Temperature (°C)', fontweight = 'bold')
ax.set_ylabel('IKr peak tail (pA/pF)', fontweight = 'bold')

# Export to graphpad.
time_gating = list()
step_gating = list()
ikr_gating = list()

for i in range(len(gating['data'])):
    time_gating.append(np.asarray(gating['data'][i]['engine.time']))
    step_gating.append(np.asarray(gating['data'][i]['membrane.V']))
    ikr_gating.append(np.asarray(gating['data'][i]['ikr.IKr']))

steps = {'steps': step_gating[0]}

data_gating23 = {'time23': time_gating[0],
               'ikr23': ikr_gating[0]}

data_gating35 = {'time35': time_gating[1],
                 'ikr35': ikr_gating[1]}

data_gating40 = {'time40': time_gating[2],
                 'ikr40': ikr_gating[2]}

gating_df23 = pd.DataFrame(data_gating23)
gating_df35 = pd.DataFrame(data_gating35)
gating_df40 = pd.DataFrame(data_gating40)
steps = pd.DataFrame(steps)

gating_df23.to_csv('Data/gating23.csv', index = False)
gating_df35.to_csv('Data/gating35.csv', index = False)
gating_df40.to_csv('Data/gating40.csv', index = False)
steps.to_csv('Data/steps_gating.csv', index = False)
tail_df.to_csv('Data/tail_temperatures.csv', index = False)
Q10df.to_csv('Data/Q10df.csv')
#%% Simulate the effects of temperature on trafficking as shown in Figure 7A.

# Load the model.
model = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Set cell type (endocardial = 0, default).
model.set_value('cell.mode', 0)

# Initialize a pacing protocol. 
pace = myokit.Protocol()

# Set basic cycle length to 100 seconds.
cl = 100000

# Create an event schedule.
pace.schedule(1, 20, 0.5, cl, 0)

# Run the simulation for different temperatures.
simulation = temp_APsim(x = X, model = model, prot = pace, bcl = cl, temp_list = temp_list, time = 24)

# Subset the simulation and run them for 24 hours.
sim_list = simulation['sim_list']
sim_run = list()
for i in range(len(sim_list)):
    sim_run.append(sim_list[i].run(24 * 3600000, log = ['engine.time', 'ikr_trafficking.S', 'ikr_trafficking.M', 'membrane.V']))

# Subset the sim_run for the membrane channels and the total channels at 41 degrees.
sim_run_mem = sim_run[:3]
sim_run41 = sim_run[3]
temp_list_mem = temp_list[:3]
temp_list41 = temp_list[3]

# Create a list with literature references.
exp_MS = ['Zhao et al. (2016)', 'Zhao et al. (2016)', 'Foo et al. (2019)', 'Foo et al. (2019)']
exp_list = [exp_Zhao37, exp_Zhao40, exp_Foo30, exp_Foo41]

# Calculate the total amount of channels at 41 degrees.
tot_ch41 = np.array(sim_run41['ikr_trafficking.M']) + np.array(sim_run41['ikr_trafficking.S'])

# Reference membrane and total occupation at steady-state.
ref_mem = 2207 
ref_tot = 2913 

# Plot the effects of temperature on trafficking.
plt.figure(figsize = (8, 5))
for i in range(len(temp_list)):
    plt.plot(np.array(sim_run[i]['engine.time'])/3600000, np.array(sim_run[i]['ikr_trafficking.M'])/ref_mem, label = f"{temp_list[i]}°C", color = sea_palette[i], lw = 2)
    plt.plot(exp_list[i]['x'], exp_list[i]['y'], label = f"{exp_MS[i]}", marker = 'o', ls = 'None', color = sea_palette[i])
    if i == 3: # Mature channels (M + S) instead of only membrane channels.
        plt.plot(np.array(sim_run41['engine.time'])/3600000, tot_ch41/ref_tot, label = f"{temp_list[3]}°C", color = 'green')
        plt.plot(exp_Foo41['x'], exp_Foo41['y'], label = 'Foo et al. (2019)', marker = 'o', ls = 'None', color = 'Green')
plt.xlabel('Time [hrs]')
plt.ylabel('Relative # of channels')
plt.tight_layout()
plt.legend(loc = "upper left", bbox_to_anchor=(-0.15, -0.2), ncol =2, title=None, frameon=False)

# Export csv for graphpad.
for i in range(len(temp_list)):
    time = np.array(sim_run[i]['engine.time'])/3600000
    ind = np.arange(0, len(time), round(len(time)/100))
    time_ind = time[ind]
    chan = np.array(sim_run[i]['ikr_trafficking.M']) / sim_run[i]['ikr_trafficking.M'][0]
    if i == 3:
        chan = np.array(sim_run[i]['ikr_trafficking.M']) + np.array(sim_run[i]['ikr_trafficking.S'])
        chan = chan / chan[0]
    chan_ind = chan[ind]
    
    exp = exp_list[i]
    
    data = {'time': time_ind,
            'chan': chan_ind}
    df = pd.DataFrame(data)
    
    df.to_csv("Data/chan" + f'{temp_list[i]}' + '.csv', index = False)
    exp.to_csv("Data/exp" + f'{temp_list[i]}' + '.csv', index = False)
#%% Simulate the effects of temperature over time on IKr and APD as shown in Figures 7B and 7C. 

# Subset the first and last beat. 
first_beat = simulation['first']
last_beat = simulation['last']

# Plot the action potentials of the first and last beat.
plt.figure()
for i in range(len(first_beat)):
    plt.plot(first_beat[i]['engine.time'], first_beat[i]['membrane.V'], label = f"first beat at {temp_list[i]}°C", ls = ':', color = sea_palette[i], lw = 2)
    plt.plot(last_beat[i]['engine.time'], last_beat[i]['membrane.V'], label = f"24h beat at {temp_list[i]}°C", ls = '-', color = sea_palette[i], lw = 2)
plt.legend()
plt.xlim(0, 1000)
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.tight_layout()

# Create a list with the APDs for each temperature and IKr for each temperature.
apd_list = APD_IKr_list(sim = simulation, before_beats = 10, bcl = cl, Chen = True, apd = True)
ikr_list = APD_IKr_list(sim = simulation, before_beats = 10, bcl = cl, Chen = True, apd = False)

# Plot the change in APD/IKr as a function of time and temperature
fig, ax = plt.subplots(1, 2)
for i in range(len(apd_list)):
    ax[0].plot(apd_list[i]['Time'], apd_list[i]['APD'], label = f"{temp_list[i]}°C", color = sea_palette[i], lw = 2)
ax[0].set_xlabel('Time [hrs]')
ax[0].set_ylabel('APD [ms]')
ax[0].set_title('Action potential duration')

for i in range(len(ikr_list)):
    ax[1].plot(ikr_list[i]['Time'], ikr_list[i]['ikr'], label = f"{temp_list[i]}°C", color = sea_palette[i], lw = 2)
ax[1].legend()
ax[1].set_xlabel('Time [hrs]')
ax[1].set_ylabel('Current density [pA/pF]')
ax[1].set_title('IKr')
fig.tight_layout()
sns.move_legend(ax[1], loc = "center", bbox_to_anchor=(-0.2, -0.25), ncol=3, title=None, frameon=False)

# Export csv for graphpad.
for i in range(len(first_beat)):
    fb_time = np.array(first_beat[i]['engine.time'])
    fb = np.array(first_beat[i]['membrane.V'])
    lb_time = np.array(last_beat[i]['engine.time'])
    lb = np.asarray(last_beat[i]['membrane.V'])

    fb_data = {'time': fb_time,
               'fb': fb}
    
    lb_data = {'time': lb_time,
               'lb': lb}

    df_fb = pd.DataFrame(fb_data)
    df_lb = pd.DataFrame(lb_data)
    
    df_fb.to_csv("Data/first_beat" + f'{temp_list[i]}' + '.csv', index = False)
    df_lb.to_csv("Data/last_beat" + f'{temp_list[i]}' + '.csv', index = False)

for i in range(len(ikr_list)):
    df_ikr = ikr_list[i]
    df_apd = apd_list[i]
    
    df_ikr.to_csv("Data/ikr_temp" + f'{temp_list[i]}' + '.csv', index = False)
    df_apd.to_csv("Data/apd_temp" + f'{temp_list[i]}' + '.csv', index = False) 
#%% Hypokalemia simulation and optimization as seen in Figure 8. 

# Load the model.
model = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Set cell type (endocardial = 0, default).
model.set_value('cell.mode', 0)

# Initialize a pacing protocol. 
pace_hypok = myokit.Protocol()

# Set basic cycle length.
cl = 3600000

# Create an event schedule.
pace_hypok.schedule(2, 20, 0.5, cl, 0)

# Create a list with indices for the overnight simulation.
inds_night = [0, 1, 2, 3] 

# Select the correct indices.
hypok_full = np.array(night_final)
sel_night = hypok_full[inds_night]

# Run the simulation and optimize the hypokalemia parameters.  
overnight_hypok = hypokalemia_opt(x = sel_night, xfull = hypok_full, inds = inds_night, rates = X, incub_hr = 12, recov_hr = 24, 
                                  incub = Guo3E_0, recov = Guo1B, conc = Guo1C, conc_week = Guo1D, psi_scale = True,
                                  week = False, showit = True, showerror = True, return_df = True)

# Create a list with indices for week simulation. Here, we first used the fit of the night as a starting point for optimization.
hypok_full_week = np.array(week_final)
inds_week = [1, 2]
sel_week = hypok_full_week[inds_week]

# Re-run the optimalisation for the weekly parameters.
week_hypok = hypokalemia_opt(x = sel_week, xfull = hypok_full_week, inds = inds_week, rates = X, incub_hr = 12, recov_hr = 24, 
                                  incub = Guo3E_0, recov = Guo1B, conc = Guo1C, conc_week = Guo1D, psi_scale= True, 
                                  week = True, showit = True, showerror = True, return_df = True)

# Export to graphpad.
hypok_names = ['conc', 'conc_exp', 'pre', 'pre_exp', 'recov']
for i in range(len(hypok_names)):
    overnight_hypok[hypok_names[i]].to_csv('Data/hypok_night_' + f'{hypok_names[i]}' + '.csv', index = False)
    week_hypok[hypok_names[i]].to_csv('Data/hypok_week_' + f'{hypok_names[i]}' + '.csv', index = False)
#%% Model the acute blocking and trafficking effects of clinically relevant hypokalemia as in Figure 8.

# Initialize an hour list.
hr_list_hypok = [20, 25, 47, 50]

# Run the simulation to first do 24 hours at 5.4 mmol/L K+ then switch to 2.5 mmol/L K+ and then back to 5.4 mmol/L K+ for both the overnight and
# the week parameter sets.
hypok_clinic_overnight = hypok_effects(x = night_final, rates = X, model = model, prot = pace_hypok, T = 72, K_low = 2.5, K_norm = 5.4, hr_list = hr_list_hypok,
                            bcl = 3600000, work_dir = work_dir, title = 'HypoK_overnight', total = False, plot = True, save = False)

# Repeat for the week parameters.
hypok_clinic_week = hypok_effects(x = week_final, rates = X, model = model, prot = pace_hypok, T = 72, K_low = 2.5, K_norm = 5.4, hr_list = hr_list_hypok,
                            bcl = 3600000, work_dir = work_dir, title = 'HypoK_week', total = False, plot = True, save = False)

# Exort csv for Graphpad.
hypok_clinic_overnight['df'].to_csv('Data/hypo_overnight.csv', index = False)
hypok_clinic_week['df'].to_csv('Data/hypo_week.csv', index = False)
night_time = [pd.DataFrame(hypok_clinic_overnight['time'][i]) for i in range(len(hypok_clinic_overnight['time']))]
night_vm = [pd.DataFrame(hypok_clinic_overnight['vM'][i]) for i in range(len(hypok_clinic_overnight['vM']))]
week_time = [pd.DataFrame(hypok_clinic_week['time'][i]) for i in range(len(hypok_clinic_week['time']))]
week_vm = [pd.DataFrame(hypok_clinic_week['vM'][i]) for i in range(len(hypok_clinic_week['vM']))]

for i in range(len(night_time)):
    night_time[i]['vM'] = night_vm[i]

for i in range(len(night_time)):
    night_time[i].to_csv('Data/hypok_night' + f'{hr_list_hypok[i]}' + '.csv', index = False)
    
for i in range(len(week_time)):
    week_time[i]['vM'] = week_vm[i]

for i in range(len(week_time)):
    week_time[i].to_csv('Data/hypok_week' + f'{hr_list_hypok[i]}' + '.csv', index = False)
#%% Graphical abstract figure script.

# Load the model.
model = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Set cell type (endocardial = 0, default).
model.set_value('cell.mode', 0)

# Initialize a pacing protocol. 
pace = myokit.Protocol()

# Set basic cycle length to 100 seconds.
cl = 100000

# Create an event schedule.
pace.schedule(1, 20, 0.5, cl, 0)

# First run the temperature simulation.
graph_temp = temp_APsim(x = X, model = model, prot = pace, bcl = cl, temp_list = [37, 40], time = 24)

# Obtain the IKrs related to the simulation.
s_steady = graph_temp['ikr'][0][0:(100-1)]
s_40 = np.concatenate((s_steady, graph_temp['ikr'][1]))

# Create a dataframe.
df_40 = pd.DataFrame(s_40, columns = ['ikr'])
df_40 = df_40.reset_index().rename(columns = {'index':'Time'})
df_40['Time'] = df_40['Time']*cl/3600000

# Export to graphpad.
df_40.to_csv('Data/df40_abstract.csv', index = False)
