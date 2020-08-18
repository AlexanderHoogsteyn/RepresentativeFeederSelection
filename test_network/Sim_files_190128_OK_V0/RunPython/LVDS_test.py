# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:31:29 2020

@author: karpan
"""
import opendssdirect as dss
import numpy as np
import matplotlib.pyplot as plt
from opendssdirect.utils import run_command
import os
import pandas as pd
import seaborn as sns       
import time       
import scipy
import scipy.stats
import chaospy as cp

def fit_distr(ser,upper):
    dist_names=['beta']
    

    df_res = pd.DataFrame(columns=('dist_name', 'p_value', 'params'))
#    dist_name = 'expon'        
    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(ser)
        
        #Applying the Kolmogorov-Smirnov test
        D, p = scipy.stats.kstest(ser, dist_name, args=param);
        
        res_dist = {'dist_name': dist_name, 'p_value': p, 'params': param}
        df_res = df_res.append(res_dist, ignore_index= True)
#    df_res = df_res.reset_index()
    df_res=df_res.loc[[df_res['p_value'].idxmax()]]
    print(df_res)
    return(df_res)
    
#%%   
if __name__ == '__main__':
    dss.run_command('Redirect "C:\\Users\\karpan\\Documents\\load and irradiance study\\Sim_files_190128_OK_V0\\RunDss\\Master.dss"')
    load=dss.utils.loads_to_dataframe()
    transformer=dss.utils.transformers_to_dataframe()
    loadshape=dss.utils.loadshape_to_dataframe()

    df_res_shape=pd.DataFrame()
    for i in range(1,np.shape(loadshape)[0])
        df_res = fit_distr(loadshape.PMult[i],np.max(loadshape.PMult[i]))
        #print(df_res.iloc[0])
        df_res = df_res.reset_index()
        df_res['shape']=loadshape.Name[i]
        #df_res_gb_ts = df_res.groupby(['ts'])
#        df_res_gb_ts = df_res.groupby(['ts'])
#        inds = df_res_gb_ts.p_value.idxmax()
#        df_res=df_res.loc[shape]    
#        df_res=df_res.groupby(['ts'], sort =False)['p_value'].max()
        print(type(df_res))
        max_args = df_res.params.apply(len).max()
     
        # Making max_args cols
        col_names = ['param_' + str(k) for k in range(max_args)]
     
     
        def extend_cols(row):
            sz_cur = len(row.params)
            for k in range(sz_cur):
                row[col_names[k]] = row.params[k]
            return row
         
        df_res = df_res.apply(extend_cols, axis=1)
        df_res = df_res.drop('params', axis=1)
