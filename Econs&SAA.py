import numpy as np
import pandas as pd

from gurobipy import *
import time
import matplotlib.pyplot as plt
import os

from matplotlib import rc
rc('text', usetex=True)

plt.rcParams['text.usetex'] = True


if os.path.exists('gurobi.log'):
    os.remove('gurobi.log')
#设置随机数种子方便调试
np.random.seed(12)

#读取效用值矩阵（均值）、交叉效用值矩阵（均值）、模块可选价格、需求权重、需求效用值下限、成本等信息
df_alternativeprice=pd.read_excel('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\数据输入.xlsx',sheet_name='AlternativePrice',header=None)
Alternative_Price=np.array(df_alternativeprice.iloc[:])

df_utilitycenter=pd.read_excel('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\数据输入.xlsx',sheet_name='UtilityCenter',header=None)
Utility_Center=np.array(df_utilitycenter.iloc[:])

Num_Demand=Utility_Center.shape[0]   #模块个数
Num_Module=Utility_Center.shape[1]    #需求维度个数
Num_Price=Alternative_Price.shape[1]     #每个模块的可选价格数量

df_pariutilitycenter=pd.read_excel('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\数据输入.xlsx',sheet_name='PairUtilityCenter',header=None)
PairUtility_Center_Temp=np.array(df_pariutilitycenter.iloc[:])
PairUtility_Center=np.zeros((Num_Demand,Num_Module,Num_Module))
for d in range(Num_Demand):
    PairUtility_Center[d,:,:]=PairUtility_Center_Temp[d*Num_Module:(d+1)*Num_Module,:]


df_Wd=pd.read_excel('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\数据输入.xlsx',sheet_name='Weightofdemand',header=None)
Wd=np.array(df_Wd.loc[:])


df_Cost=pd.read_excel('D:\研究生阶段\科研与项目\毕业生涯\PSS 配置文献\Coding\程序实现\数据输入.xlsx',sheet_name='CostData',header=None)
Cost=np.array(df_Cost.loc[:])


Sensitivity_Center=0.2  #价格敏感性


#标记每类必选模块和可选模块的终止序号（实际值），先必选后可选。
Mandatory=[[1,3],[7,8]]
Optional=[[4,4],[5,6],[9,10]]

#标记产品和服务模块
Product=[1,2,3,4,5,6]
Service=[7,8,9,10]

#标记相容和相斥的模块组合（序号为实际值）。相容的是有先后关系的，先出现的需要后出现的支持；括号里的是同一类模块。相斥的，只是列表中的几项不能同时出现而已。
Inclusive=[[7,[[4,],]]]
Exclusive=[[4,5]]



#定义EpsilonConstraint相关参数
k=0.0001

#定义SAA相关参数
Initial_M=5
Final_M=50
Initial_N=20
Final_N=500
Test_SampleSize=1000
Width_Utility=0.2
Width_Sensitivity=0.1
M_Step=5
N_Step=10
Tolerance = 0.05
Variance=0.8

Max_Iterations=200
N_Enlarge=2


IsStochastic=1


#定义各项函数


def ParaGenerator_Stochastic():
    Utility_tmp = Utility_Center + np.random.uniform(-Width_Utility, Width_Utility, size=(Num_Demand, Num_Module))
    for d in range(Num_Demand):
        for i in Product:
            Utility_tmp[d][i-1]=Utility_Center[d][i-1]

    #为了避免数值问题，所以把最低效用认为设置成小于0.1（最低实际效用）的数值，但是不接受随机调整
    Utility_tmp=np.where(Utility_tmp<0.1,Utility_Center,Utility_tmp)
    Utility_tmp=np.where(Utility_tmp<0,0,Utility_tmp)

    PairUtility_tmp = PairUtility_Center.copy()
    for d in range(Num_Demand):
        PairUtility_tmp[d] = np.triu(PairUtility_tmp[d])
        PairUtility_tmp[d][PairUtility_tmp[d] != 0] = PairUtility_tmp[d][PairUtility_tmp[d] != 0] + np.random.uniform(-Width_Utility, Width_Utility,len(PairUtility_tmp[d][PairUtility_tmp[d] != 0]))
        PairUtility_tmp[d] = PairUtility_tmp[d] + PairUtility_tmp[d].T - np.diag(PairUtility_tmp[d].diagonal())


    for d in range(Num_Demand):
        sumtemp=np.sum(abs(Utility_tmp[d]))+np.sum(abs(PairUtility_tmp[d]))/2
        Utility_tmp[d]=Utility_tmp[d]/sumtemp
        PairUtility_tmp[d]=PairUtility_tmp[d]/sumtemp


    Sensitivity_tmp = Sensitivity_Center + np.random.uniform(-Width_Sensitivity, Width_Sensitivity)
    while Sensitivity_tmp<=0:
        Sensitivity_tmp = Sensitivity_Center + np.random.uniform(-Width_Sensitivity, Width_Sensitivity)



    return [Utility_tmp,PairUtility_tmp,Sensitivity_tmp]


def TestSample_Generator(Test_SampleSize_tmp):
    Utility_test_tmp = np.zeros((Test_SampleSize_tmp, Num_Demand, Num_Module))
    PairUtility_test_tmp = np.zeros((Test_SampleSize_tmp, Num_Demand, Num_Module, Num_Module))
    Sensitivity_test_tmp = np.zeros(Test_SampleSize_tmp)

    for TestID in range(Test_SampleSize_tmp):
        [Utility_test_tmp[TestID], PairUtility_test_tmp[TestID],
         Sensitivity_test_tmp[TestID]] = ParaGenerator_Stochastic()
    return [Utility_test_tmp, PairUtility_test_tmp, Sensitivity_test_tmp]


def ParaGenerator_Deterministic():
    Utility_tmp = Utility_Center
    PairUtility_tmp = PairUtility_Center
    Sensitivity_tmp = Sensitivity_Center

    for d in range(Num_Demand):
        sumtemp=np.sum(abs(Utility_tmp[d]))+np.sum(abs(PairUtility_tmp[d]))/2
        Utility_tmp[d]=Utility_tmp[d]/sumtemp
        PairUtility_tmp[d]=PairUtility_tmp[d]/sumtemp

    return [Utility_tmp, PairUtility_tmp, Sensitivity_tmp]


def Optimizer(NN,Is_Stochastic):
    m = Model()
    # 定义决策变量
    X = m.addVars(Num_Module, vtype=GRB.BINARY, name="X")
    Y = m.addVars(Num_Module, Num_Module, vtype=GRB.BINARY, name="Y")
    Z = m.addVars(Num_Module, Num_Price, vtype=GRB.BINARY, name="Z")
    s = m.addVar(vtype=GRB.CONTINUOUS, name="s")
    Storage=np.zeros((NN,Num_Demand)).tolist()
    Storage_max=np.zeros(NN).tolist()
    Storage_min=np.zeros(NN).tolist()
    for sample in range(NN):
        for d in range(Num_Demand):
            Storage[sample][d]=m.addVar(vtype=GRB.CONTINUOUS)
        Storage_max[sample]=m.addVar(vtype=GRB.CONTINUOUS,name="Storage_max")
        Storage_min[sample]=m.addVar(vtype=GRB.CONTINUOUS,name="Storage_min")

    m.update()

    # 定义和随机项有关的内容，包括目标函数和需求最低满足程度约束
    obj1 = LinExpr()
    for sample in range(NN):
        if Is_Stochastic=='Stochastic':
            [Utility, PairUtility, Sensitivity] = ParaGenerator_Stochastic()
        if Is_Stochastic=='Deterministic':
            [Utility, PairUtility, Sensitivity] = ParaGenerator_Deterministic()


        obj1 = obj1 + quicksum(Z[i, p] * Wd[d] * Utility[d][i] * np.exp(-Sensitivity * Alternative_Price[i][p]) for d in range(Num_Demand) for i in range(Num_Module) for p in range(Num_Price)) \
               + quicksum(Y[i, j] * Wd[d] * PairUtility[d][i][j] / 2 for d in range(Num_Demand) for i in range(Num_Module) for j in range(Num_Module))

        for d in range(Num_Demand):
            m.addConstr(Storage[sample][d]==quicksum(X[i] * Utility[d][i] for i in range(Num_Module)) + quicksum(Y[i, j] * PairUtility[d][i][j] / 2 for i in range(Num_Module) for j in range(Num_Module)))
        m.addConstr(Storage_max[sample]==max_([Storage[sample][d] for d in range(Num_Demand)]))
        m.addConstr(Storage_min[sample]==min_([Storage[sample][d] for d in range(Num_Demand)]))
        m.addConstr((1-Variance)*Storage_max[sample]<=Storage_min[sample])


    if Is_Stochastic == 'Stochastic':
        obj1 = obj1 / NN + k * s
    if Is_Stochastic == 'Deterministic':
        obj1=obj1+k*s
    m.setObjective(obj1, GRB.MAXIMIZE)

    # 定义与随机项无关的内容
    # 处理第二个目标（成为约束）
    obj2 = quicksum(Z[i, p] * Alternative_Price[i][p] for i in range(Num_Module) for p in range(Num_Price)) \
           - quicksum(X[i] * Cost[i] for i in range(Num_Module))
    m.addConstr(obj2 - s >= Epsilon+0.5)


    # X与Y的关系
    for i in range(Num_Module):
        for j in range(Num_Module):
            if (i != j):
                m.addConstr(Y[i, j] == and_(X[i], X[j]))

    # X与Z的关系
    for i in range(Num_Module):
        m.addConstr(X[i] == quicksum(Z[i, p] for p in range(Num_Price)))

    # 必选模块
    for q in Mandatory:
        m.addConstr(quicksum(X[i] for i in range(q[0]-1,q[1])) == 1)

    # 可选模块
    for q in Optional:
        m.addConstr(quicksum(X[i] for i in range(q[0]-1,q[1])) <= 1)

    # 相容
    for p in Inclusive:
        for q in p[1]:
            m.addConstr(X[p[0] - 1] <= quicksum(X[r - 1] for r in q))

    # 相斥
    for q in Exclusive:
        m.addConstr(quicksum(X[z - 1] for z in q) <= 1)


    m.update()
    m.setParam("LogToConsole", 0)

    m.optimize()

    DVX=np.zeros(Num_Module)
    DVY=np.zeros((Num_Module,Num_Module))
    DVZ=np.zeros((Num_Module,Num_Price))
    DVs=0
    obj1tmp=0
    obj2tmp=0
    Flag_of_Infeasible_tmp = 0

    if m.status == GRB.Status.INF_OR_UNBD:
        print('模型无解需重新构建')
        Flag_of_Infeasible_tmp=1
    else:
        for i in range(Num_Module):
            DVX[i] = m.getVarByName('X[%d]' % (i)).x
            for j in range(Num_Module):
                DVY[i][j] = m.getVarByName('Y[%d,%d]' % (i, j)).x
            for p in range(Num_Price):
                DVZ[i][p] = m.getVarByName('Z[%d,%d]' % (i, p)).x

        DVs=m.getVarByName('s').x
        obj1tmp = obj1.getValue()
        obj2tmp = obj2.getValue()


    return [DVX, DVY, DVZ, DVs, obj1tmp, obj2tmp,Flag_of_Infeasible_tmp]


def Cal_Epsilon_LB(NN=Initial_N):
    m = Model()
    # 定义决策变量
    X = m.addVars(Num_Module, vtype=GRB.BINARY, name="X")
    Y = m.addVars(Num_Module, Num_Module, vtype=GRB.BINARY, name="Y")
    Z = m.addVars(Num_Module, Num_Price, vtype=GRB.BINARY, name="Z")
    s = m.addVar(vtype=GRB.CONTINUOUS, name="s")

    m.update()

    # 定义和随机项有关的内容，包括目标函数和需求最低满足程度约束
    obj1 = LinExpr()
    for sample in range(NN):
        [Utility, PairUtility, Sensitivity] = ParaGenerator_Stochastic()

        obj1 = obj1 + quicksum(Z[i, p] * Wd[d] * Utility[d][i] * np.exp(-Sensitivity * Alternative_Price[i][p]) for d in range(Num_Demand) for i in range(Num_Module) for p in range(Num_Price)) \
               + quicksum(Y[i, j] * Wd[d] * PairUtility[d][i][j] / 2 for d in range(Num_Demand) for i in range(Num_Module) for j in range(Num_Module))

    obj1 = obj1 / NN
    m.setObjective(obj1, GRB.MAXIMIZE)

    # X与Y的关系
    for i in range(Num_Module):
        for j in range(Num_Module):
            if (i != j):
                m.addConstr(Y[i, j] == and_(X[i], X[j]))

    # X与Z的关系
    for i in range(Num_Module):
        m.addConstr(X[i] == quicksum(Z[i, p] for p in range(Num_Price)))

    # 必选模块
    for q in Mandatory:
        m.addConstr(quicksum(X[i] for i in range(q[0]-1,q[1])) == 1)

    # 可选模块
    for q in Optional:
        m.addConstr(quicksum(X[i] for i in range(q[0]-1,q[1])) <= 1)

    #相容
    for p in Inclusive:
        for q in p[1]:
            m.addConstr(X[p[0] - 1] <= quicksum(X[r - 1] for r in q))

    # 相斥
    for q in Exclusive:
        m.addConstr(quicksum(X[z - 1] for z in q) <= 1)


    m.update()
    m.setParam("LogToConsole", 0)
    m.optimize()

    if m.status==GRB.Status.INF_OR_UNBD:
        print('Epsilon下限无法求得')
        return 0

    DVX=np.zeros(Num_Module)
    DVZ=np.zeros((Num_Module,Num_Price))

    for i in range(Num_Module):
        DVX[i] = m.getVarByName('X[%d]' % (i)).x
        for p in range(Num_Price):
            DVZ[i][p] = m.getVarByName('Z[%d,%d]' % (i, p)).x

    profit=sum(DVZ[i][p] * Alternative_Price[i][p] for i in range(Num_Module) for p in range(Num_Price)) \
    - sum(DVX[i] * Cost[i] for i in range(Num_Module))

    return profit


def Cal_Epsilon_UB(NN=Initial_N):
    m = Model()
    # 定义决策变量
    X = m.addVars(Num_Module, vtype=GRB.BINARY, name="X")
    Y = m.addVars(Num_Module, Num_Module, vtype=GRB.BINARY, name="Y")
    Z = m.addVars(Num_Module, Num_Price, vtype=GRB.BINARY, name="Z")
    s = m.addVar(vtype=GRB.CONTINUOUS, name="s")


    m.update()


    obj2 = quicksum(Z[i, p] * Alternative_Price[i][p] for i in range(Num_Module) for p in range(Num_Price)) \
           - quicksum(X[i] * Cost[i] for i in range(Num_Module))

    m.setObjective(obj2, GRB.MAXIMIZE)

    # X与Y的关系
    for i in range(Num_Module):
        for j in range(Num_Module):
            if (i != j):
                m.addConstr(Y[i, j] == and_(X[i], X[j]))

    # X与Z的关系
    for i in range(Num_Module):
        m.addConstr(X[i] == quicksum(Z[i, p] for p in range(Num_Price)))

    # 必选模块
    for q in Mandatory:
        m.addConstr(quicksum(X[i] for i in range(q[0]-1,q[1])) == 1)

    # 可选模块
    for q in Optional:
        m.addConstr(quicksum(X[i] for i in range(q[0]-1,q[1])) <= 1)

    # 相容
    for p in Inclusive:
        for q in p[1]:
            m.addConstr(X[p[0] - 1] <= quicksum(X[r - 1] for r in q))

    # 相斥
    for q in Exclusive:
        m.addConstr(quicksum(X[z - 1] for z in q) <= 1)


    m.update()
    m.setParam("LogToConsole", 0)
    m.optimize()
    if m.status==GRB.Status.INF_OR_UNBD:
        print('Epsilon上限无法求得')
        return 0

    return m.getObjective().getValue()


def Evaluation(DVX,DVY,DVZ,DVs,SampleSize,Utilitytmp,PairUtilitytmp,Sensitivitytmp):
    Objectivetmp = 0
    for TestID in range(SampleSize):
        Objectivetmp = Objectivetmp + sum(DVZ[i][p] * Wd[d] * Utilitytmp[TestID][d][i] * np.exp(-Sensitivitytmp[TestID] * Alternative_Price[i][p]) for d in range(Num_Demand) for i in range(Num_Module) for p in range(Num_Price)) \
                       + sum(DVY[i][j] * Wd[d] * PairUtilitytmp[TestID][d][i][j] / 2 for d in range(Num_Demand) for i in range(Num_Module) for j in range(Num_Module))
    Satisfactiontmp = Objectivetmp / SampleSize
    Objectivetmp = Satisfactiontmp + k * DVs
    Profittmp = sum(DVZ[i][p] * Alternative_Price[i][p] for i in range(Num_Module) for p in range(Num_Price)) \
                - sum(DVX[i] * Cost[i] for i in range(Num_Module))

    return [Objectivetmp,Satisfactiontmp, Profittmp]


def Judge_Feasibility(DVX,DVY,SampleSize,Utilitytmp,PairUtilitytmp):
    Feasible=1
    for TestID in range(SampleSize):
        tmptmp=[]
        for d in range(Num_Demand):
            tmptmp.append(sum(DVX[i] * Utilitytmp[TestID][d][i] for i in range(Num_Module)) \
                   + sum(DVY[i][j] * PairUtilitytmp[TestID][d][i][j] / 2 for i in range(Num_Module) for j in range(Num_Module)))

        if ((1-Variance)*max(tmptmp)>min(tmptmp)):
            Feasible = 0
            break

    return Feasible


def SAA(M,N,N_test):

    while((M<Final_M)and(N<Final_N)):
        print('M=', M, 'N=', N)
        DecisionVarX = np.zeros((M, Num_Module))
        DecisionVarY = np.zeros((M, Num_Module, Num_Module))
        DecisionVarZ = np.zeros((M, Num_Module, Num_Price))
        DecisionVars=  np.zeros((M))
        Objective1 = np.zeros(M)
        Objective2 = np.zeros(M)
        Solution_Feasibility=np.zeros(M)

        trials=0
        Iterations=0
        while((trials<M) and (Iterations<Max_Iterations)):
            [DecisionVarX[trials], DecisionVarY[trials], DecisionVarZ[trials], DecisionVars[trials], Objective1[trials],Objective2[trials], Flag_of_Infeasible] = Optimizer(N, 'Stochastic')
            if(Flag_of_Infeasible==0):
                trials=trials+1
            else:
                Iterations = Iterations + 1
        if(Iterations==Max_Iterations):
            print('实验样本下模型无解，程序异常停止')
            return [0,0]

        for trials in range(M):
            Solution_Feasibility[trials] = Judge_Feasibility(DecisionVarX[trials], DecisionVarY[trials], N_test, Utility_test,PairUtility_test)

        if(np.sum(Solution_Feasibility)==0):
            if(N*N_Enlarge>Final_N):
                print('当前N的范围内，无法找到测试样本下的可行解，程序异常终止')
                return [0,0]
            else:
                print('未发现测试样本下的可行解，临时调大N')
                N = N * N_Enlarge
                continue


        UpperBound = Objective1[Solution_Feasibility==1].mean()

        Objective_Test = np.zeros(M)
        Gap_Test = np.zeros(M) - 1
        Satisfaction_Test = np.zeros(M)
        Profit_Test = np.zeros(M)
        for TrialID in range(M):
            if (Solution_Feasibility[TrialID] == 1):
                [Objective_Test[TrialID], Satisfaction_Test[TrialID], Profit_Test[TrialID]] = Evaluation(
                    DecisionVarX[TrialID], DecisionVarY[TrialID], DecisionVarZ[TrialID], DecisionVars[TrialID], N_test,
                    Utility_test, PairUtility_test, Sensitivity_test)
                Gap_Test[TrialID] = abs((UpperBound - Objective_Test[TrialID])) / UpperBound


        LowerBound=np.max(Objective_Test[Solution_Feasibility==1])


        Gap = abs((UpperBound - LowerBound)) / UpperBound

        print('下限为：',LowerBound,'上限为：',UpperBound)
        print('Gap=',Gap)
        if(Gap<=Tolerance):
            break
        else:
            M = M + M_Step
            N = N + N_Step


    SelectedSol_ID=np.argmax(Objective_Test)

    print('最优方案对应的试验序号为', SelectedSol_ID + 1)
    print('最优方案的gap、满意度、利润分别为：',Gap_Test[SelectedSol_ID],Satisfaction_Test[SelectedSol_ID],Profit_Test[SelectedSol_ID])
    print(DecisionVarX[SelectedSol_ID])
    print(DecisionVarY[SelectedSol_ID])
    print(DecisionVarZ[SelectedSol_ID])
    print(DecisionVars[SelectedSol_ID])
    print(DecisionVarX)
    print(DecisionVars)
    print('各方案的目标值为：', Objective_Test)
    print('各方案的gap为：',Gap_Test)
    print('各方案的满意度为：', Satisfaction_Test)
    print('各方案的利润为：', Profit_Test)

    return [Satisfaction_Test[SelectedSol_ID],Profit_Test[SelectedSol_ID]]


def Deterministic_Output(N_test):
    DecisionVarX_Deterministic = np.zeros((Num_Module))
    DecisionVarY_Deterministic = np.zeros((Num_Module, Num_Module))
    DecisionVarZ_Deterministic = np.zeros((Num_Module, Num_Price))
    [DecisionVarX_Deterministic, DecisionVarY_Deterministic, DecisionVarZ_Deterministic, DecisionVars_Deterministic, Objective1_Deterministic,Objective2_Deterministic,Flag_of_Infeasible] = Optimizer(1, 'Deterministic')

    if(Flag_of_Infeasible==1):
        print('确定性问题无解，需调整参数表')
        return [0,0,0]


    Feasible_Flag = Judge_Feasibility(DecisionVarX_Deterministic, DecisionVarY_Deterministic, N_test, Utility_test,PairUtility_test)
    [Objective_Test, Satisfaction_Test, Profit_Test] = Evaluation(DecisionVarX_Deterministic,
                                                                  DecisionVarY_Deterministic,
                                                                  DecisionVarZ_Deterministic,
                                                                  DecisionVars_Deterministic, N_test, Utility_test,
                                                                  PairUtility_test, Sensitivity_test)

    if(Feasible_Flag==0):
        print('该解在随机情况下不可行！')

    print('满意度为：', Satisfaction_Test, '利润为：', Profit_Test)
    print(DecisionVarX_Deterministic)
    print(DecisionVarZ_Deterministic)
    print(DecisionVars_Deterministic)

    return [Satisfaction_Test,Profit_Test,Feasible_Flag]





#主程序开始
[Utility_test,PairUtility_test,Sensitivity_test]=TestSample_Generator(Test_SampleSize)

# 其实这里的Epsilon上下限应该是f2的
Epsilon_LB=Cal_Epsilon_LB()
Epsilon_UB=Cal_Epsilon_UB()
print('e的下限为：',Cal_Epsilon_LB())
print('e的上限为：',Cal_Epsilon_UB())


Epsilon = Epsilon_LB-1
Satisfaction_Result=[]
Profit_Result=[]
Feasible_for_Deterministic=[]
time_start=time.time()


while (Epsilon < Epsilon_UB):
    print('Epsilon=',Epsilon)
    Past_Epsilon=Epsilon
    if(IsStochastic==1):
        [Satisfaction_Temp, Profit_Temp] = SAA(Initial_M, Initial_N, Test_SampleSize)
    else:
        [Satisfaction_Temp,Profit_Temp,Feasible_Temp]=Deterministic_Output(Test_SampleSize)
        Feasible_for_Deterministic.append(Feasible_Temp)

    Epsilon = Profit_Temp
    Current_Epsilon=Epsilon
    if(Past_Epsilon>=Current_Epsilon):
        print('帕累托前沿面出现异常情况，程序停止')
        break
    Satisfaction_Result.append(Satisfaction_Temp)
    Profit_Result.append(Profit_Temp)



time_end=time.time()
print('Finished')
print('共计用时：',np.rint(time_end-time_start),'s')

print('双目标优化结果如下：')
for i in range(len(Satisfaction_Result)):
    print(Satisfaction_Result[i],Profit_Result[i])
    if((IsStochastic!=1)and(Feasible_for_Deterministic[i]==0)):
        print('该解在随机情况下不可行！')



