# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:44:38 2020

@author: LocalAdmin
"""

import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from symfit import variables, parameters, Fit, exp, sqrt, pi
from symfit.core.minimizers import BFGS, SLSQP, LBFGSB, BaseMinimizer, GradientMinimizer, HessianMinimizer, ConstrainedMinimizer, MINPACK, ChainedMinimizer, BasinHopping, ScipyMinimize, DifferentialEvolution
from symfit.core.objectives import LeastSquares, LogLikelihood, VectorLeastSquares
# =============================================================================
# Setting
# =============================================================================
# param sweep:
# dBm: power sweep
# Hz: rf_freq sweep
# amps: current sweep
param_sweep = "Hz"

# number of sweeps in the measurement (not counting backsweep) 
number_of_sweeps = 1

# channels that should be fitted
# channels are (normally) (x and y channel of the lock-in)
# only one channel possible
# 2,3 : Triplet
# 4,5 : Voltage
# 6,7 : Singlet
rd_channel = 4
# =============================================================================
# End of settings
# =============================================================================

# path where the measurement data is saved    
path = r"C:\Users\LocalAdmin\Documents\Master\auswertung\global-fit\readout"
# path where the generated data should be saved
newpath = r"C:\Users\LocalAdmin\Documents\Master\auswertung\global-fit\results"

h = 6.62607015*10**(-34)
mub = 9.2740100783*10**(-24)
g_th = 2

def fitfunc(x, y0, A, R, xc, dB1, dB2, a1, a2):
    return y0 + (A/(np.sqrt((dB1**2+a1**2*x**2)*np.pi/2)))*np.exp(-2*((x-xc)/np.sqrt(dB1**2+a1**2*x**2))**2)+(A/(R*(np.sqrt((dB2**2+a2**2*x**2)*np.pi/2))))*np.exp(-2*((x-xc)/np.sqrt(dB2**2+a2**2*x**2))**2)

if not os.path.exists(newpath):
    os.makedirs(newpath)
# finds the files in the given path folder
files = [ f for f in listdir(path) if isfile(join(path,f)) ]
comb_data_array = []
for file in files:
    current_file = path+"/"+file
    file_info=open(current_file,'r')
    lines = []
    for i in range(75):
        lines.append(file_info.readline())
    if param_sweep == "dBm":   
        param_data = lines[4].partition('Hz, ')[2]
        param_data = param_data.partition(' dBm')[0]
        param_data = float(param_data)
        param_data = param_data + 41
    if param_sweep == "amps":
        param_data = lines[74].partition('V ')[2]
        param_data = param_data.partition('A')[0]
        param_data = float(param_data)
        param_data = param_data*1000
    if param_sweep == "Hz":
        param_data = lines[4].partition('settings: ')[2]
        param_data = param_data.partition(' Hz')[0]
        param_data = float(param_data)
        param_data = param_data/(10**6)
        
    file_info.close()
    
    file_data = np.loadtxt(current_file, skiprows = 77)
    file_data = np.array(file_data)
    
    splitted_data = []
    for channel in range(0, len(file_data[0,:])):
        buffer = file_data[:,channel]
        buffer = np.split(buffer, number_of_sweeps*2)
        splitted_data.append(buffer)
        # now the data gets averaged
        # note that the backsweep arrays have to be reversed!
    averaged_data = []
    for channel in range(0, len(file_data[0,:])):
        averaged_data_sc = np.zeros(len(splitted_data[channel][0]))
        for sweep in range(0,len(splitted_data[channel])):
            if (sweep % 2) == 0:
                averaged_data_sc = averaged_data_sc + splitted_data[channel][sweep][::-1]
            else:
                averaged_data_sc = averaged_data_sc + splitted_data[channel][sweep]
            
        averaged_data_sc = averaged_data_sc/(len(splitted_data[channel]))
        averaged_data.append(averaged_data_sc)
    
    comb_data = np.array([param_data, np.split(averaged_data[0],2)[0], np.split(averaged_data[rd_channel],2)[0]])
    comb_data_array.append(comb_data)
comb_data_array = np.array(comb_data_array)

#comb_data_array = sorted(comb_data_array, key=lambda x: x[0])


# create x and y variable strings to feed into the model for symfit

x_list = []
for i in range(len(comb_data_array)):
    buffer = 'x_'+str(i+1)
    x_list.append(buffer)

y_list = []
for i in range(len(comb_data_array)):
    buffer = 'y_'+str(i+1)
    y_list.append(buffer)

xs_str=''
for x_str in x_list:
    if xs_str=='':
        xs_str = xs_str+x_str
    else:
        xs_str = xs_str+', '+x_str
    
ys_str=''
for y_str in y_list:
    if ys_str=='':
        ys_str = ys_str+y_str
    else:
        ys_str = ys_str+', '+y_str

xs = variables(xs_str)
ys = variables(ys_str)

# create the shared and independent parameters for the fit functions

numberofdata = len(comb_data_array)
numberofdata = numberofdata + 1

dB1, dB2, a1, a2 = parameters('dB1, dB2, a1, a2')
y0s = parameters(', '.join('y0_{}'.format(i) for i in range(1,numberofdata)))
As = parameters(', '.join('A_{}'.format(i) for i in range(1, numberofdata)))
Rs = parameters(', '.join('R_{}'.format(i) for i in range(1, numberofdata)))
xcs = parameters(', '.join('xc_{}'.format(i) for i in range(1, numberofdata)))

# starting values for the shared parameters and for the independent parameters

dB1.value = 2.1
dB2.value = 1.2
dB1.min = 1
dB2.min = 0.5
a1.value = 0
a2.value = 0
a1.fixed = True
a2.fixed = True
#a1.max = 0.02
#a2.max = 0.02
#a1.min = 0
#a2.min = 0
#R.min = 0
#R.value = 2
for i in range(0, len(Rs)):
    Rs[i].value = 2
for i in range(0, len(Rs)):
    Rs[i].min = 0
for i in range(0, len(Rs)):
    Rs[i].max = 3
for i in range(0,len(xcs)):
    xcs[i].min = 0
for i in range(0,len(As)):
    As[i].min = 1*10**(-5)
for i in range(0, len(As)):
    As[i].value = 0.1
for i in range(0,len(xcs)):
    freq = comb_data_array[i][0]
    freq = freq * 10**6
    xcs[i].value = h/(g_th*mub)*freq*1000

# creating the fitfunctions in a dictionary so that symfit can use them

model_dict = {
        y: y0 + (A/(sqrt((dB1**2+a1**2*x**2)*pi/2)))*exp(-2*((x-xc)/sqrt(dB1**2+a1**2*x**2))**2)+(A/(R*(sqrt((dB2**2+a2**2*x**2)*pi/2))))*exp(-2*((x-xc)/sqrt(dB2**2+a2**2*x**2))**2)
            for x,y,y0,A,R,xc in zip(xs,ys,y0s,As,Rs,xcs)
}

# creating an array of all x_arrays
x_values_array = []
for comb_data in comb_data_array:
    x_values = comb_data[1]
    x_values_array.append(x_values)

# finding the index where the x_value is zero. This will be used later to 
# equalize the size of the arrays
x_zero_index =[]    
for x_values in x_values_array:
    zero_index = np.where(x_values == 0)[0][0]
    x_zero_index.append(zero_index)

y_values_array = []
for comb_data in comb_data_array:
    y_values = comb_data[2]
    y_values_array.append(y_values)

y_minimums = []

for y_values in y_values_array:
    min_y = np.amin(y_values)
    y_minimums.append(min_y)
    
    
# check for the longest array out of the imported datasets
for i in range(len(x_values_array)):
    if i == 0:
        longest_data = i
    else:
        if len(x_values_array[i])>len(x_values_array[longest_data]):
            longest_data = i

long_x_index = longest_data

for i in range(len(y_values_array)):
    if i ==0:
        longest_data = i
    else:
        if len(y_values_array[i])>len(y_values_array[longest_data]):
            longest_data = i

long_y_index = longest_data

# check if there are any datasets that have less entries than the longest one
# and append the x_arrays with zeros until they have the same length
# this is needed in order for the MINPACK minimizer to work
for i in range(len(x_values_array)):
    len_diff = len(x_values_array[long_x_index])-len(x_values_array[i])
    if len_diff != 0:
        for j in range(len_diff):
            x_values_array[i] = np.append(x_values_array[i],0)
    
# append the y_values with the value at 0mT to get all arrays to the same size
for i in range(len(y_values_array)):
    append_value = y_values_array[i][x_zero_index[i]]
    len_diff = len(y_values_array[long_y_index])-len(y_values_array[i])
    if len_diff != 0:
        for j in range(len_diff):
            y_values_array[i] = np.append(y_values_array[i],append_value)

# restructure the data to feed them into the fit class in symfit
x_data = dict(zip(x_list, x_values_array))
y_data = dict(zip(y_list, y_values_array))    
    
x_data.update(y_data)
data = x_data

# creating the fit, here the minimizer MINPACK is used, which is basically a 
# least squares fit
fit = Fit(model_dict, **data, minimizer=MINPACK)
fit_result = fit.execute()

number_of_it = 4

for i in range(number_of_it):
    dB1.value = fit_result.value(dB1)
    dB2.value = fit_result.value(dB2)
    a1.value = fit_result.value(a1)
    a2.value = fit_result.value(a2)
#    R.value = fit_result.value(R)
    for i in range(len(Rs)):
        Rs[i].value = fit_result.value(Rs[i])
    for i in range(len(As)):
        As[i].value = fit_result.value(As[i])
    for i in range(len(xcs)):
        xcs[i].value = fit_result.value(xcs[i])
    for i in range(len(xcs)):
        y0s[i].value = fit_result.value(y0s[i])

    model_dict = {
            y: y0 + (A/(sqrt((dB1**2+a1**2*x**2)*pi/2)))*exp(-2*((x-xc)/sqrt(dB1**2+a1**2*x**2))**2)+(A/(R*(sqrt((dB2**2+a2**2*x**2)*pi/2))))*exp(-2*((x-xc)/sqrt(dB2**2+a2**2*x**2))**2)
                for x,y,y0,A,R,xc in zip(xs,ys,y0s,As,Rs,xcs)
#                for x,y,y0,A,xc in zip(xs,ys,y0s,As,xcs)
    }
    
    fit = Fit(model_dict, **data, minimizer=MINPACK)
    fit_result = fit.execute()
    

dB1 = fit_result.value(dB1)
dB2 = fit_result.value(dB2)
a1 = fit_result.value(a1)
a2 = fit_result.value(a2)
#R = fit_result.value(R)

data_array_arr = []
j = 0
for comb_array in comb_data_array:
    outputstr = "magnetic field \t"+str(comb_array[0])+"MHz data \t"+str(comb_array[0])+"MHz fit"
    y0 = fit_result.value(y0s[j])
    A = fit_result.value(As[j])
    R = fit_result.value(Rs[j])
    xc = fit_result.value(xcs[j])
    fit_array = fitfunc(comb_array[1], y0, A, R, xc, dB1, dB2, a1, a2)
    data_array = [comb_array[1], comb_array[2], fit_array]
    j = j+1
    data_array_arr.append(data_array)
    plt.plot(data_array[0], data_array[1])
    plt.plot(data_array[0], data_array[2])
    plt.savefig(newpath+"/"+str(comb_array[0])+".png")
    plt.clf()
    out_data = np.array([data_array[0], data_array[1], data_array[2]])
    out_data = np.transpose(out_data)
    out_data_filename = str(comb_array[0])+"MHz.txt"
    path_name = join(newpath, out_data_filename)
    np.savetxt(path_name, out_data, delimiter="\t", header=outputstr)
    
    
print(fit_result)



    
    





