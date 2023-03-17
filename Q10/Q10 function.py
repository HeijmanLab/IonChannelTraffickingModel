# -*- coding: utf-8 -*-
"""
Author: Stefan Meier
Institute: CARIM Maastricht University
Supervisor: Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 06/10/2021
Adapted from Lei, C. L., Clerx, M., Beattie, K. A., Melgari, D., Hancox, J., Gavaghan, D. J., 
Polonchuk, L., Wang, K., Mirams, G. R. (2019). Rapid characterisation of hERG channel kinetics II: 
temperature dependence. Biophysical Journal, 117, 2455-2470.
GitHub link to original paper: https://github.com/CardiacModelling/hERGRapidCharacterisation
Function: Temperature coefficient Q10 function
"""

# Import numpy to calculate standard deviation.
import numpy as np

# This assumes paired comparisons due to single N, otherwise use two N's. 
def Q10func(a1, a2, s1, s2, t1, t2, N1, N2,  decimal, rounding = False):
    """ Q10func
    This function calculates the Q10 values based on the standard error of means and the time constants for
    each of the temperature measurments. This function is adapted from Lei et al. 2019. Rapid characterisation 
    of hERG channel kinetics II: temperature dependence. Biophysical Journal, 117, 2455-2470. 
    GitHub link to original paper: https://github.com/CardiacModelling/hERGRapidCharacterisation
    
    Parameters
    ----------
    a1 : Integer
        Time constant at the first temperature.
        
    a2 : Integer
         Temperature input in degrees Celsius.
         
    s1 : Integer
         Standard error at first temperature.
    
    s2 : Integer
        Standard error at second temperature.
        
    t1 : Integer
        First temperature.
        
    t2 : Integer
        Second temperature.
        
    N1 : Integer
        Sample size corresponding to first temperature.
        
    N2 : Integer
        Sample size corresponding to first temperature.
        
    decimal : Integer
           Round to x decimal.
           
    rounding : Boolean, default is False
           True is rounding is used, default is False.
           
    Returns
    -------
    Tuple with Q10 value and standard deviation.
    """ 
    
    s1 = s1 * np.sqrt(N1)
    s2 = s2 * np.sqrt(N2)
    q10 = (a1 / a2)**(10.0/(t2 - t1))
    std_q10 = np.sqrt((s1 * (10.0/(t2 - t1))/ a1 * q10) ** 2 
                      + (s2 * (10.0/(t2 - t1)) / a2 * q10) ** 2 
                      + (q10 * np.log(a1 / a2) * (10.0/(t2 - t1)) ** 2 / 10.0) ** 2
                      + (q10 * np.log(a1 / a2) * (10.0/(t2 - t1)) ** 2 / 10.0) ** 2)
    if rounding == True:
        q10 = round(q10, decimal)
        std_q10 = round(std_q10, decimal)
    return q10, std_q10


# Zhou et al. (1998). Properties of hERG channels stably expressed in HEK 293 cells studied at physiological temperature.
Q10_activation = Q10func(a1 = 947, a2 = 105, s1 = 87, s2 = 15, t1 = 23, t2= 35, N1 = 6, N2 = 6, rounding = True, decimal = 2)
Q10_deactivation = Q10func(a1 = 216, a2 = 149, s1 = 19, s2 = 27, t1 = 23, t2= 35, N1 = 3, N2 = 3, rounding = True, decimal = 2)
Q10_inactivation = Q10func(a1 = 14.2, a2 = 3.1, s1 = 1.3, s2 = 0.3, t1 = 23, t2= 35, N1 = 3, N2 = 3, rounding = True, decimal = 2)
Q10_recovery = Q10func(a1 = 8.5, a2 = 1.8, s1 = 0.6, s2 = 0.1, t1 = 23, t2= 35, N1 = 3, N2 = 3, rounding = True, decimal = 2)

# Mauerhöfer & Bauer 2016. Effects of Temperature on Heteromeric Kv11.1a/1b and Kv11.3 channels.
# Interestingly, some of the parameters in table 1 result in different Q10s than the ones being calculated here. 
# This is likely due to rounding errors. Note, the Q10 deactivation was based on weighted parameters.
Q10_mbA = Q10func(a1 = 192, a2 = 23.1, s1 = 10.7, s2 = 1.4, t1 = 21, t2 = 35, N1 = 16, N2 = 14, rounding = True, decimal = 2)
Q10_mbD100 = Q10func(a1 = 56.1, a2 = 10.9, s1 = 5.1, s2 = 1.1, t1 = 21, t2 = 35, N1 = 12, N2 = 11, rounding = True, decimal = 2)
Q10_mbD50 = Q10func(a1 = 579, a2 = 155, s1 = 77, s2 = 12, t1 = 21, t2 = 35, N1 = 11, N2 = 11, rounding = True, decimal = 2)
Q10_mbI40 = Q10func(a1 = 9.2, a2 = 2, s1 = 0.7, s2 = 0.3, t1 = 21, t2 = 35, N1 = 15, N2 = 13, rounding = True, decimal = 2)
Q10_mbI0 = Q10func(a1 = 16.9, a2 = 3.3, s1 = 0.6, s2 = 0.2, t1 = 21, t2 = 35, N1 = 15, N2 = 13, rounding = True, decimal = 2)
Q10_mbR = Q10func(a1 = 13.6, a2 = 2.4, s1 = 0.4, s2 = 0.1, t1 = 21, t2 = 35, N1 = 14, N2 = 12, rounding = True, decimal = 2)
