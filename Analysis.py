#-*- coding=utf-8 -*-

import numpy as np
import pandas as pd



variance_list=[0.600000,0.800000]
width_utility=0.100000

if width_utility==0.200000:
    seed_list=[19,20,33,38,40,41,55,57,58,59,108,203,204,206,207,208,210,250,270,300]
    string='随机性强'
if width_utility==0.100000:
    seed_list=[19,20,33,38,40,41,55,57,58,59,108,203,204,206,207,208,210,250,270,300]
    string='随机性弱'

Result=np.zeros((len(variance_list),len(seed_list),6))-np.ones((len(variance_list),len(seed_list),6))
varianceID=-1


for variance in variance_list:
    varianceID = varianceID + 1
    seedID = -1
    for seed in seed_list:
        seedID = seedID + 1

        filename=open('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\新确定性和随机性对比\所用样本\%s\随机性%f 确定与否%d 约束%f 种子%d.csv' % (string,width_utility,1,variance,seed))
        df_stochastic= pd.read_csv(filename,header=None,encoding = 'gb18030')
        stochastic_result_temp = np.array(df_stochastic.iloc[:, 0:2])
        for j in range(len(stochastic_result_temp)):
            stochastic_result_temp[j][1]=round(stochastic_result_temp[j][1])

        filename=open('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\新确定性和随机性对比\所用样本\%s\随机性%f 确定与否%d 约束%f 种子%d.csv' % (string,width_utility, 0, variance, seed))
        df_deterministic = pd.read_csv(filename,header=None,encoding = 'gb18030')
        deterministic_result_temp = np.array(df_deterministic.iloc[:, 0:3])
        for j in range(len(deterministic_result_temp)):
            deterministic_result_temp[j][1]=round(deterministic_result_temp[j][1])

        count_node = 0
        dominated_feasible = 0
        dominated_infeasible = 0
        gap = 0

        for i in range(len(stochastic_result_temp)):
            if (stochastic_result_temp[i][1]) in deterministic_result_temp[:, 1]:
                count_node = count_node + 1
                tmp = deterministic_result_temp[:, 1].tolist().index(stochastic_result_temp[i][1])
                if (stochastic_result_temp[i][0] - deterministic_result_temp[tmp][0] > 0.000001) and (
                    deterministic_result_temp[tmp][2] == 1):
                    dominated_feasible = dominated_feasible + 1
                if (deterministic_result_temp[tmp][2] == 0):
                    dominated_infeasible = dominated_infeasible + 1

        Result[varianceID][seedID][0] = seed
        Result[varianceID][seedID][1] = count_node
        Result[varianceID][seedID][2] = dominated_feasible
        Result[varianceID][seedID][3] = dominated_infeasible
        Result[varianceID][seedID][4] = dominated_feasible / count_node
        Result[varianceID][seedID][5] = dominated_infeasible / count_node


pd.DataFrame(Result[0]).to_csv('%s 约束紧.csv' % (string),header=False,index=False)
pd.DataFrame(Result[1]).to_csv('%s 约束松.csv' % (string),header=False,index=False)


