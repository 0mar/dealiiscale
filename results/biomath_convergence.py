#!/usr/bin/env python
# coding: utf-8

# # Convergence of deal.II code
#
# In this notebook we test the performance of our manufacturing solution implementations.
# First, we fix the microscopic problem and solve the macroscopic problem. The approximation should be converging quadratically, using the right elements.
#
# Format:
# ```
# cycle cells dofs    mL2       mH1       ML2       MH1
#     0    16   25 3.409e-01 3.409e-01 7.560e-01 1.815e+00
# ```


import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_convergence(filename,key,errors=None):
    if errors is None:
        errors = ['mL2','mH1','ML2','MH1']
    full_df = pd.read_csv(filename,delim_whitespace=True)
    final_entries = (full_df[full_df.cycle=='cycle'].index - 1).append(pd.Index([len(full_df)-1]))
    df = full_df.loc[final_entries]
    df.dofs = df.dofs.astype('int')
    conv_rates = []
#     fig,axs = plt.subplots(2,2,figsize=(10,10))
    plt.figure()
    for num, column in enumerate(errors):
        df[column] = pd.to_numeric(df[column])
        plt.loglog(df.dofs,df[column],'*-')
        conv_rate = np.log(df[column].iloc[-1]/df[column].iloc[-2])/np.log(0.5)
        conv_rates.append(conv_rate)
#         ax = axs.flatten()[num]
#         ax.loglog(df.dofs,df[column],'*-')

#         ax.set(xlabel="Number of DoF",ylabel="Error",title='%s: Observed order: %.2f'%(column,conv_rate))
    ax = plt.gca()
    ax.set(xlabel="Number of DoF",ylabel="Error",title='%s: Convergence of errors'%key)
    legend = ["%s: %.2f"%(col,error) for col,error in zip(errors,conv_rates)]
    plt.legend(legend)
    plt.savefig(filename.replace('.txt','.pdf'))



folders = {'benchmark':'Benchmark'}
for folder in folders:
    plot_convergence('%s/linear_convergence_table.txt'%folder,'%s (linear map)'%folders[folder])
