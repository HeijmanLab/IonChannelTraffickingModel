# -*- coding: utf-8 -*-
"""
Author: Stefan Meier
Institute: CARIM, Maastricht University
Supervisor: Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 21/10/2021
Script: Time constants and Q10s
"""
#%% Import packages and set directories
# Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import myokit
from scipy.optimize import minimize, curve_fit 
import os

# Set directories
work_dir = os.getcwd() 
if 'Documents' in work_dir: 
    work_dir = work_dir
else: 
    work_dir = os.path.join(work_dir, 'Documents', 'PhD', 'Python', 'GitHub', 'IonChannelTraffickingModel')
os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)
    
# Import local functions    
from TraffickingModelFunctions import mono_exp, double_exp, Q10, ZhouActTime, ZhouDeactTime, ZhouActVolt, ZhouInactTime, ZhouRecovTime

# To inspect the plots, you'd need to run the qt5 statement in the IPython console
# Plot inline = %matplotlib inline
# Plot outside = %matplotlib qt5

#%% Load the model and obtain some variables for demoting (from a state to ordinary variable)
# The following scripts shows how to apply a step protocol without changing
# the model definition code. Instead, all necessary changes are applied through
# API calls.

# Close all previous plots
plt.close('all') 

# Load the model
m = myokit.load_model('MMT/ORD_TEMP_final.mmt')

# Get pacing variable
p = m.get('engine.pace')

# Remove binding to pacing mechanism before voltage coupling
p.set_binding(None)

# Get membrane potential
v = m.get('membrane.V')
# Demote v from a state to an ordinary variable; no longer dynamic
v.demote()
# right-hand side setting; in this case the actual value doesn't matter because
# it will be linked to the pacing protocol
v.set_rhs(0)
# Bind v's value to the pacing mechanism/protocol
v.set_binding('pace')

# Get intracellular potassium 
ki = m.get('potassium.Ki')
# Demote ki from a state to an ordinary variable; no longer dynamic
ki.demote()
# 110 intracellular K+ is based on Odening et al. (2019) DOI: 10.1093/eurheartj/ehy761, while Zhou et al. (1998) used 130 intra K+ DOI: 10.1016/S0006-3495(98)77782-3.
# We performed all the analyses with 110 intra K+.
ki.set_rhs(110)

#%% Define a protocol that allows to calculate the time constants related to activation

# Time constants protocol, wherein the step duration increases logarithmically from ~10ms to ~7000ms
dur_act = np.logspace(1, 4, num = 20, endpoint = False)
 # Minimum duration 
dur_act_min = min(dur_act) 
# Maximum duration
dur_act_max = max(dur_act)  
# Total duration of the time constants protocol
total_duration = 8500
# The duration of the holding pulse 
hold_pulse = 500

# Initialize a new protocol
p_dur_act = myokit.Protocol()

# Loop through the elements in the duration list 
for d, step in enumerate(dur_act):
    # 500 ms duration of the holding potential
    p_dur_act.add_step(-80, (hold_pulse)) 
    # Depolarize to 0 mV which is the largest current from the original protocol by Zhou et al. 1998 (Figure 6)
    p_dur_act.add_step(0.1, step) 
    # Resume holding potential adjusted for pulse duration
    p_dur_act.add_step(-60, (hold_pulse + (7500 - step))) 
# Maximum time of all events in the protocol 
t_dur_act = p_dur_act.characteristic_time() - 1

# Initialize a simulation protocol to calculate the time constants
s_dur_act = myokit.Simulation(m, p_dur_act)

#%% Calculate the time constants related to the activation with a mono exponential function

# Perform curve fitting at both room and physiological temperature
act23 = ZhouActTime(modeltype = 2, temp = 23, sim = s_dur_act, time = t_dur_act, 
                    tot_dur = total_duration, hold = hold_pulse, t_steps = dur_act, showit = 1, 
                    showcurve = 1, log = 1)

act35 = ZhouActTime(modeltype = 2, temp = 35, sim = s_dur_act, time = t_dur_act, 
                    tot_dur = total_duration, hold = hold_pulse, t_steps = dur_act, showit = 1, 
                    showcurve = 1, log = 1)

# Calculate the Q10 activation between room and physiological temperature
Q10_act = round(Q10(a1 = act23['tau'], a2 = act35['tau'], t1 = 23 , t2 = 35), 2)

# Plot the normalized peak values together with the fitted time constants
plt.figure()
plt.semilogx(dur_act, act23['tail_norm'], 'o-', color = 'black', label = '23°C')
plt.semilogx(dur_act, act35['tail_norm'], 'o-', color = 'red', label = '35°C')
plt.semilogx(dur_act, np.asarray(mono_exp(dur_act, -1, act23['tau'], 1)), '-', color = 'green', label = 'Fitted Tau at 23°C')
plt.semilogx(dur_act, np.asarray(mono_exp(dur_act, -1, act35['tau'], 1)), '-', color = 'orange', label = 'Fitted Tau at 35°C')
plt.hlines(y = 0.5, xmin = min(dur_act)-10, xmax = act23['half_val'], linestyle = '--', color = 'blue')
plt.axvline(act23['half_val'], ymax = 0.5, linestyle = '--', color = 'blue', label = 'T1/2')
plt.axvline(act35['half_val'], ymax = 0.5, linestyle = '--', color = 'blue')
plt.xlabel('Prepulse duration [ms]')
plt.ylabel('Normalized tail current')
plt.suptitle('Normalized tail values')
plt.legend()
plt.tight_layout()
#%% Define a protocol that allows to calculate the time constants related to deactivation

# Deactivation voltage steps from Zhou et al. 1998 table 2 (-30 mV to -100 mV)
steps_deact = np.arange(-30, -110, -10) 

# Initializea deactivation protocol
p_deact = myokit.Protocol()

# Loop through the elements in the voltage steplist 
for k, step_deact in enumerate(steps_deact):
    # 500ms of holding potential
    p_deact.add_step(-80, 500) 
    # Voltage step for 1000 ms
    p_deact.add_step(60, 1000)
    # 1000 ms repolarizing step for tail current
    p_deact.add_step(step_deact, 1000) 
    # resume holding potential for 500ms
    p_deact.add_step(-80, 500)

# Total duration of all voltage steps in the above protocol (3000 ms)
total_deact = 3000

# Maximum time of all events in the protocol
t_deact = p_deact.characteristic_time() - 1

# Initialize a simulation protocol to calculate the time constants
s_deact = myokit.Simulation(m, p_deact)

#%% Calculate the time constants related to the deactivation with a double exponential function

# Perform curve fitting at both room and physiological temperature
deact23 = ZhouDeactTime(modeltype = 2, temp = 23, sim = s_deact, time = t_deact, 
                        tot_dur = total_deact, showit = 1, showcurve = 1)

deact35 = ZhouDeactTime(modeltype = 2, temp = 35, sim = s_deact, time = t_deact, 
                        tot_dur = total_deact, showit = 1, showcurve = 1)

# Note, similar (in trend) to Zhou et al. 1998, the fast component's contribution to the relative
# amplitude increases at more negative values. Nonetheless, the slow component still dominates
# the deactivation rate.
rel_amp_d23 = deact23['rel_amp']
rel_amp_d35 = deact35['rel_amp']

# Plot the relative amplitudes of the fast and slow components to visualize its mono exponential
# characteristics
plt.figure()
ax1 = plt.subplot(2,1,1)
plt.plot(steps_deact, deact35['tau_fast'], c = 'black', ls = '-', label = 'Tau fast at 35°C')
plt.plot(steps_deact, deact35['tau_slow'], c = 'red', ls = '-', label = 'Tau slow at 35°C')
plt.plot(steps_deact, deact23['tau_fast'], c = 'blue', ls = '--', label = 'Tau fast at 23°C')
plt.plot(steps_deact, deact23['tau_slow'], c = 'orange', ls = '--', label = 'Tau slow at 23°C')
plt.legend()
plt.ylabel('Time [ms]')
plt.title('Deactivation time constants')
plt.tight_layout()
ax2 = plt.subplot(2,1,2)
ax1.sharex(ax2)
plt.plot(steps_deact, rel_amp_d23, c = 'black', label = '23°C')
plt.plot(steps_deact, rel_amp_d35, c = 'red', label = '35°C')
plt.ylabel('Relative amplitude')
plt.xlabel('Membrane potential [mV]')
plt.title('Relative amplitudes with respect to afast')
plt.legend()
plt.tight_layout()

# First inialize the estimated parameters. 
p0 = (1, 200, 0)

# Fit the mono exponential approximation of the double exponent, where the input for the double exponential
# estimated parameters are derived from Zhou et al. 1998 table 2 at -50 mV.
# Note, the duration of the tail current was 1000 ms
mono_res, mono_cov = curve_fit(mono_exp, np.arange(0, 1000, 1), double_exp(np.arange(0, 1000, 1), a1 = 0.55, tau1 = 137, a2 = 0.45, tau2 = 1027, c = 0), p0 = p0, maxfev = 3000)

# Visualize the mono exponential approximation of the double exponent
plt.figure()
plt.plot(np.arange(0, 1000, 1), double_exp(np.arange(0, 1000, 1), a1 = 0.55, tau1 = 137, a2 = 0.45, tau2 = 1027, c = 0), color = 'black', label = 'Experimental double exponential')
plt.plot(np.arange(0, 1000, 1), mono_exp(np.arange(0, 1000, 1), *mono_res), color = 'red', label = 'Mono exponential fitting')
plt.legend()
plt.ylabel('Current [pA]')
plt.xlabel('Time [ms]')
plt.title('Mono exponential approximation of the experimental double exponent')
plt.tight_layout()

# Calculate the Q10 rate at -50mV with the weighted taus
# Note, the weighted tau is chosen because the individual taus are not well-defined in the model (no double exponent, but approximate mono exponent)
Q10_deact = round(Q10(a1 = deact23['tau_weight'][2], a2 = deact35['tau_weight'][2], t1 = 23 , t2 = 35), 2)
#%%  Define a protocol that allows to calculate temperature dependent shift in V1/2 related to activation

# Activation voltage steps from Zhou et al. 1998 Figure 3 (-60 mV to 50 mV), however we extended the range with +10 mV due to a typo.
# This does not meaningfully change the results.
steps_actv = np.arange(-60.1, 60, 10) 

# Initializean activation protocol
p_actv = myokit.Protocol()

# Loop through the elements in the voltage steplist 
for k, step_actv in enumerate(steps_actv):
    # 500ms of holding potential
    p_actv.add_step(-80, 500) 
    # Voltage step for 4000 ms
    p_actv.add_step(step_actv, 4000)
    # 5000 ms repolarizing step for tail current
    p_actv.add_step(-50, 5000) 
    # resume holding potential for 500ms
    p_actv.add_step(-80, 500)

# Total duration of all voltage steps in the above protocol (10000 ms)
total_actv = 10000

# Maximum time of all events in the protocol
t_actv = p_actv.characteristic_time() - 1

# Initialize a simulation protocol
s_actv = myokit.Simulation(m, p_actv)
  
#%% Calculate and visualize the temperature-dependent shift in midpoint activation

# Fit the actvolt function to obtain the values to create an IV-curve
actv23 = ZhouActVolt(modeltype = 2, temp = 23, sim = s_actv, time = t_actv, 
                     tot_dur = total_actv, v_steps = steps_actv, showit = 1)

actv35 = ZhouActVolt(modeltype = 2, temp = 35, sim = s_actv, time = t_actv, 
                     tot_dur = total_actv, v_steps = steps_actv, showit = 1)

# Half maximal activation 
hlf_actv23 = actv23['half_val']
hlf_actv35 = actv35['half_val']

# Plot the IV-curve at the different temperatures
plt.figure()
plt.plot(steps_actv, actv23['tail_norm'], color = 'black',  label = '23°C')
plt.plot(steps_actv, actv35['tail_norm'], color = 'red', label = '35°C')
plt.hlines(y = 0.5, xmin = min(steps_actv)-5, xmax = hlf_actv23, linestyle = '--', color = 'blue')
plt.axvline(hlf_actv23, ymax = 0.5, linestyle = '--', color = 'blue', label = 'V1/2')
plt.axvline(hlf_actv35, ymin = 0, ymax = 0.5, linestyle = '--', color = 'blue')
plt.xlim(min(steps_actv)-5)
plt.ylabel('Normalized tail current')
plt.xlabel('Prepulse potential [mV]')
plt.title('Temperature-dependent shift in half maximal activation')
plt.legend()
plt.tight_layout()

# Voltage shifts are often calculated as differences, while rate constants are calculated as ratios.
# The V1/2 from Zhou et al. 1998 were used as comparison, where the V1/2 at 35°C was -28.1 mV and
# at 23°C was -14.2 mV.
hlf_diff = hlf_actv23 - hlf_actv35
hlf_zhou = -14.2 - -28.1
print(hlf_diff)
#%% Inactivation protocol to calculate the inactivation time constants 

# Inactivation voltage steps from Zhou et al. 1998 Figure 7
steps_inact = np.arange(-20.1, 70, 20) 

# Initializean inactivation protocol
p_inact = myokit.Protocol()

# Loop through the elements in the voltage steplist 
for k, step_inact in enumerate(steps_inact):
    # 500ms of holding potential
    p_inact.add_step(-80, 500) 
    # Voltage step for 200ms to activate and inactivate IKr current
    p_inact.add_step(60, 200)
    # 2ms repolarizing step to allow for recovery from 
    # inactivation without sign. deactivation
    p_inact.add_step(-100, 2) 
    # Test step to observe inactivation (based of Zhou et al. 1998 figure 7)
    p_inact.add_step(step_inact, 20)
    # Resume holding potential
    p_inact.add_step(-80, 500)

# Total duration of all voltage steps in the above protocol
total_inact = 1222

# Maximum time of all events in the protocol
t_inact = p_inact.characteristic_time() - 1

# Initialize a simulation protocol
s_inact = myokit.Simulation(m, p_inact)

#%% Calculate the time constants related to the inactivation with a mono exponential function

# Perform curve fitting at both room and physiological temperature
inact23 = ZhouInactTime(modeltype = 2, temp = 23, sim = s_inact, time = t_inact, 
                        tot_dur = total_inact, showit = 1, showcurve = 1)
    
inact35 = ZhouInactTime(modeltype = 2, temp = 35, sim = s_inact, time = t_inact, 
                        tot_dur = total_inact, showit = 1, showcurve = 1)

# Calculate the Q10 inactivation between room and physiological temperature at 0mV
Q10_inact = round(Q10(a1 = inact23['tau'][1], a2 = inact35['tau'][1], t1 = 23 , t2 = 35), 2)
#%% Recovery from inactivation protocol

# Recovery from inactivation voltage steps from Zhou et al. 1998 Figure 7
steps_recin = np.arange(-20, -120, -20) 

# Initialize a recovery from inactivation protocol
p_recin = myokit.Protocol()

# Loop through the elements in the voltage steplist
for k, step_recin in enumerate(steps_recin):
    # 500ms of holding potential
    p_recin.add_step(-80, 500) 
    # Voltage step for 200ms to activate and inactivate IKr current
    p_recin.add_step(60, 200)
    # Produce tail current (duration based on Zhou et al. 1998 figure 7)
    p_recin.add_step(step_recin, 20) 
    # Resume holding potential
    p_recin.add_step(-80, 500)

# Total duration of all voltage steps in the above protocol
total_recin = 1220

# Maximum time of all events in the protocol
t_recin = p_recin.characteristic_time() - 1

# Initialize a simulation protocol
s_recin = myokit.Simulation(m, p_recin)

#%% Calculate the time constants related to the recovery from inactivation with either a mono (>-40 mV) or a double (=<-40 mV) exponential function

# Fit the recovtime function to obtain the time constants related to recovery from inactivation
recov23 = ZhouRecovTime(modeltype = 2, temp = 23, sim = s_recin, time = t_recin, 
                        tot_dur = total_recin, showit = 1, showcurve = 1)

recov35 = ZhouRecovTime(modeltype = 2, temp = 35, sim = s_recin, time = t_recin, 
                        tot_dur = total_recin, showit = 1, showcurve = 1)

# Calculate the Q10 recovery from inactivation between room and physiological temperature at -60mV
Q10_recov = round(Q10(a1 = recov23['tau_fast'][2], a2 = recov35['tau_fast'][2], t1 = 23 , t2 = 35), 2)

#%% Plot the inactivation and recovery from inactivation time constants

# Create a list with the taus (combine mono exponent with tau fast)
# Note, the asteriks is to unpack the variable
recov_l23 = [*recov23['tau_mono'], *recov23['tau_fast']]
recov_l23 = [i for i in recov_l23 if i != 0]
recov_e23 = ['Mono', 'Fast', 'Fast', 'Fast', 'Fast']

recov_l35 = [*recov35['tau_mono'], *recov35['tau_fast']]
recov_l35 = [i for i in recov_l35 if i != 0]
recov_e35 = ['Mono', 'Fast', 'Fast', 'Fast', 'Fast']

# Create a df that contains the taus togeter with a color label
df_recov23 = pd.DataFrame({'Tau':recov_l23, 'Type':recov_e23})
df_recov35 = pd.DataFrame({'Tau':recov_l35, 'Type':recov_e35})

# Plot the membrane potential and the time constants
plt.figure()
plt.plot(steps_recin[:2], df_recov23['Tau'].iloc[:2,], 'o-', color = 'red', label = df_recov23['Type'].iloc[0] + ' 23°C')
plt.plot(steps_recin[1:], df_recov23['Tau'].iloc[1:,], 'o-', color = 'black', label = df_recov23['Type'].iloc[2] + ' 23°C')
plt.plot(steps_recin[:2], df_recov35['Tau'].iloc[:2,], 'o-', color = 'blue', label = df_recov35['Type'].iloc[0] + ' 35°C')
plt.plot(steps_recin[1:], df_recov35['Tau'].iloc[1:,], 'o-', color = 'orange', label = df_recov35['Type'].iloc[2] + ' 35°C')
plt.plot(steps_inact, inact23['tau'], 'o-', color = 'purple', label = 'Mono 23°C')
plt.plot(steps_inact, inact35['tau'], 'o-', color = 'green', label = 'Mono 35°C')
plt.xlabel('Membrane potential [mV]')
plt.ylabel('Time constant [ms]')
plt.title('Time constants for inactivation and recovery from inactivation')
plt.legend()
plt.tight_layout()
#%% Create a function that can be used to optimize the Q10s and the V1/2 shift to mimic the 
# behaviour of Zhou et al. 1998

# Create a new simulation that will be used in the TempOpti function
s_error = myokit.Simulation(m, p_dur_act)

# Weights to scale the error calculations (only when rel_error is True)
weights_rel = [0.9, 6.5, 1, 2, 1.2]

def TempOpti(x, sim, half_zhou, w_rel, rel_error = False):
    """ TempOpti function
    The TempOpti function can be used to minimize the sum of squared differences between
    the Q10s and V1/2 shifts from Zhou et al. 1998 and our model.
    
    Parameters
    ----------
    x : Array
        Array that contains the Q10s and scaling factor from the model
         
    sim : Myokit simulation protocol
         Simulation protocol.
    
    half_zhou : Integer/Float
        Half maximal activation according to Zhou et al. 1998 
        
    w_rel : List
        Weights to scale the error calculations only if rel_error is True.
        
    rel_error : Boolean, default is False
        Relative error calculations (normalized to the maximal value of Zhou et al. 1998)
    
    Returns
    -------
    Array with the optimized Q10s and scaling factors.
    """
    
    # Set the Q10s in the model to the array input values
    sim.set_constant('ikr_MM.Q10_act', x[0])
    sim.set_constant('ikr_MM.Q10_deact', x[1])
    sim.set_constant('ikr_MM.Q10_inact', x[2])
    sim.set_constant('ikr_MM.Q10_recov', x[3])
    sim.set_constant('ikr_MM.temp_dep', x[4])
    sim.set_constant('ikr_MM.temp_dep2', x[5])
      
    # Set the protocol to the one used to calculate the activation time constants 
    sim.set_protocol(p_dur_act)
    # Calculate the activation time constants for both room and physiological temperatures
    act23 = ZhouActTime(modeltype = 2, temp = 23, sim = sim, time = t_dur_act, 
                        tot_dur = total_duration, hold = hold_pulse, t_steps = dur_act, showit = 0, 
                        showcurve = 0, log = 0)
    act35 = ZhouActTime(modeltype = 2, temp = 35, sim = sim, time = t_dur_act, 
                        tot_dur = total_duration, hold = hold_pulse, t_steps = dur_act, showit = 0, 
                        showcurve = 0, log = 0)
    # Calculate the Q10 activation between room and physiological temperature
    Q10_act = round(Q10(a1 = act23['tau'], a2 = act35['tau'], t1 = 23 , t2 = 35), 2)
    
    # Set the protocol to the one used to calculate the deactivation time constants
    sim.set_protocol(p_deact)
    # Calculate the deactivation time constants for both room and physiological temperatures
    deact23 = ZhouDeactTime(modeltype = 2, temp = 23, sim = sim, time = t_deact, 
                            tot_dur = total_deact, showit = 0, showcurve = 0)
    deact35 = ZhouDeactTime(modeltype = 2, temp = 35, sim = sim, time = t_deact, 
                            tot_dur = total_deact, showit = 0, showcurve = 0)
    # Calculate the Q10 deactivation between room and physiological temperature at -50 mV
    Q10_deact = round(Q10(a1 = deact23['tau_weight'][2], a2 = deact35['tau_weight'][2], t1 = 23 , t2 = 35), 2)
    
    # Set the protocol to the one used to calculate the inactivation time constants
    sim.set_protocol(p_inact)
    # Calculate the inactivation time constants for both room and physiological temperatures
    inact23 = ZhouInactTime(modeltype = 2, temp = 23, sim = sim, time = t_inact, 
                            tot_dur = total_inact, showit = 0, showcurve = 0)  
    inact35 = ZhouInactTime(modeltype = 2, temp = 35, sim = sim, time = t_inact, 
                            tot_dur = total_inact, showit = 0, showcurve = 0)
    # Calculate the Q10 inactivation between room and physiological temperature at 0mV
    Q10_inact = round(Q10(a1 = inact23['tau'][1], a2 = inact35['tau'][1], t1 = 23 , t2 = 35), 2)
    
    # Set the protocol to the one used to calculate the recovery from inactivation time constants
    sim.set_protocol(p_recin)
    # Calculate the recovery from inactivation time constants for both room and physiological temperatures
    recov23 = ZhouRecovTime(modeltype = 2, temp = 23, sim = sim, time = t_recin, 
                            tot_dur = total_recin, showit = 0, showcurve = 0)
    recov35 = ZhouRecovTime(modeltype = 2, temp = 35, sim = sim, time = t_recin, 
                            tot_dur = total_recin, showit = 0, showcurve = 0)
    # Calculate the Q10 recovery from inactivation between room and physiological temperature at -60mV
    Q10_recov = round(Q10(a1 = recov23['tau_fast'][2], a2 = recov35['tau_fast'][2], t1 = 23 , t2 = 35), 2)

    # Set the protocol to the one used to calculate the shift in half-maximal activation
    sim.set_protocol(p_actv)
    # Calculate the half maximal activation for both room and physiological temperatures
    actv23 = ZhouActVolt(modeltype = 2, temp = 23, sim = sim, time = t_actv, 
                         tot_dur = total_actv, v_steps = steps_actv, showit = 0)
    actv35 = ZhouActVolt(modeltype = 2, temp = 35, sim = sim, time = t_actv, 
                         tot_dur = total_actv, v_steps = steps_actv, showit = 0)
    # Calculate the difference in half maximal activation between the two temperatures
    half_model = round(actv23['half_val'] - actv35['half_val'], 2)
    
    # Create an array that stores the new Q10s
    Q10_model = np.array([Q10_act, Q10_deact, Q10_inact, Q10_recov])
    
    # Reference array that contains the Q10s from Zhou et al. 1998
    Q10_zhou = np.array([6.25, 1.36, 3.55, 3.65])
    
    # Calculate the sum of squared differences between the Q10s from the model and Zhou et al. 1998
    # Note, scale down the error related to activation by 0.5, due to large SD in Zhou et al. 1998.
    if rel_error is False: 
        err_Q10_A = 0.5 * ((Q10_zhou[0] - Q10_model[0]) * (Q10_zhou[0] - Q10_model[0]))
        err_Q10_D = (Q10_zhou[1] - Q10_model[1]) * (Q10_zhou[1] - Q10_model[1])
        err_Q10_I = (Q10_zhou[2] - Q10_model[2]) * (Q10_zhou[2] - Q10_model[2])
        err_Q10_R = (Q10_zhou[3] - Q10_model[3]) * (Q10_zhou[3] - Q10_model[3])
        
        # Calculate the sum of squared differences between the V1/2 shift from the model and Zhou et al.
        # 1998. Note, this error term is scaled down by 0.2, because the difference in V1/2 has a 
        # disproportionate contribution to the error. An error of 2 mV in V1/2 is much less severe than
        # an error of 1 between the model's Q10 and Zhou's Q10. 
        
        error_half = 0.02 * ((half_zhou - half_model) * (half_zhou - half_model))
    
        # Calculate the total error by adding the above shown error terms
        error_tot =  error_half + err_Q10_A + err_Q10_D + err_Q10_I + err_Q10_R
     
    if rel_error is True:
        # Optimize the relative error difference instead of the absolute error difference to homogenize
        # the error weights a bit more.
  
        err_Q10_A = w_rel[0] * ((Q10_zhou[0] - Q10_model[0])/Q10_zhou[0]) * ((Q10_zhou[0] - Q10_model[0])/Q10_zhou[0])
        err_Q10_D = w_rel[1] * ((Q10_zhou[1] - Q10_model[1])/Q10_model[1]) * ((Q10_zhou[1] - Q10_model[1])/Q10_zhou[1])
        err_Q10_I = w_rel[2] * ((Q10_zhou[2] - Q10_model[2])/Q10_model[2]) * ((Q10_zhou[2] - Q10_model[2])/Q10_zhou[2])
        err_Q10_R = w_rel[3] * ((Q10_zhou[3] - Q10_model[3])/Q10_model[3]) * ((Q10_zhou[3] - Q10_model[3])/Q10_zhou[3])
        
        # Calculate the sum of squared differences between the V1/2 shift from the model and Zhou et al.
        # 1998. Again, normalize to the experimental data to obtain the relative error difference
        error_half = w_rel[4] * ((half_zhou - half_model)/half_zhou) * ((half_zhou - half_model)/half_zhou)
    
        # Calculate the total error by adding the above shown error terms
        error_tot =  error_half + err_Q10_A + err_Q10_D + err_Q10_I + err_Q10_R
    
    # Print the input array, the errors and the corresponding Q10s and V1/2
    print("X: [%f, %f, %f, %f, %f, %f]" % (x[0], x[1], x[2], x[3], x[4], x[5]))
    print("Error: [tot: %f, hlf: %f, A: %f, D: %f, I: %f, R: %f]" % (error_tot, error_half, err_Q10_A, err_Q10_D, err_Q10_I, err_Q10_R))
    print("Q10_model: [%f, %f, %f, %f]; V1/2: [%f]" % (Q10_model[0], Q10_model[1], Q10_model[2], Q10_model[3], half_model))
    return error_tot

# The optimized parameter set.          
x_final = [5.64528884, 1.19795417, 3.41629472, 2.93346223, 1.21461546, 1.11308848]

# Fit the TempOpti function
tempopt_res = TempOpti(x = x_final, sim = s_error, half_zhou = hlf_zhou, w_rel = weights_rel, rel_error = True)

# Subset the resulting model Q10 and export it.
Q10_data = {'Act': [4.74], 'Deact': [1.4], 'Inact': [3.2], 'Recov': [3.86], 'V1/2': [15.04]}
Q10s_halfmax = pd.DataFrame(Q10_data)
Q10s_halfmax.to_csv('Data/Q10_Cali.csv')
