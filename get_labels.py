# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:26:03 2023

@author: user
"""
import os
# import re
# import pandas as pd


save_file=r"label_02_1-3_8-10.csv"
with open(save_file,"w") as wf:
    wf.write(",".join(["name","type","Q1","Q2","Q3","Q4"]))
    wf.write("\n")
    Y_train=[]
    xuhao=0
    for root, dirs, files in os.walk(r'combined3waves_02_1-3_8-10'):
        for name in files:
            print(xuhao,name)
            xuhao+=1
            label= name.strip().split('-')
            Y_train.append(label)
            # print(Y_train)
            wf.write(",".join([name,label[-5],label[-4].split('_')[-1],label[-3].split('_')[-1],
                               label[-2].split('_')[-1],label[-1].split('_')[-1].split(".")[-2]]))
            wf.write("\n")