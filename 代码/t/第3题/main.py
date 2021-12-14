import geatpy as ea
from Problem3Pool import Problem3
import random
import numpy as np
import pandas as pd
def calc_problem(dataSet = 'A', initial = 0):
    """===============================实例化问题对象==========================="""
    problem = Problem3(dataSet)  # 生成问题对象
    prophetPop = None
    """=================================种群设置=============================="""
    Encoding = 'RI'  # 编码方式
    if dataSet == 'A':
        NIND = 200  # 种群规模
    else:
        NIND = 100
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 15  # 最大进化代数
    myAlgorithm.maxTrappedCount = 20
    if dataSet == 'A':
        myAlgorithm.MAXTIME = 60 * 60
    else:
        myAlgorithm.MAXTIME = 2 * 60 * 60
    #myAlgorithm.mutOper.Pm = 0.5  # 变异概率
    myAlgorithm.logTras = 5  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    if initial == 1:
        prophetChrom = np.random.randint(1,problem.crewNum + 1,size=(NIND, problem.flightNum * 2)) 
        for j in range(4000, problem.flightNum):
            prophetChrom[:,j * 2 + 0] = 0
            prophetChrom[:,j * 2 + 1] = 1
        """==========================初始种群========================"""
        prophetPop = ea.Population(Encoding, Field, NIND, prophetChrom)
        myAlgorithm.call_aimFunc(prophetPop)
    """==========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run(prophetPop)  # 执行算法模板，得到最优个体以及最后一代种群
    fileName = 'Resultap4000/' + dataSet +str(initial)
    BestIndi.save(fileName)  # 把最优个体的信息保存到文件中
    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
    else:
        print('没找到可行解。')

if __name__ == '__main__':
    calc_problem('B', 1)
    #calc_problem('B', 1)
