import numpy as np
import geatpy as ea
from pre_process import *
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy

class Problem1(ea.Problem):  # 继承Problem父类
    def __init__(self, dataSet = 'A'):
        name = 'Problem1'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        self.is_cpt, self.is_fo, self.base, self.duty_cost, self.pair_cost, self.stn_pool = read_crew(dataSet)
        self.crewNum = len(self.is_cpt)
        self.dptrDate, self.arrvDate, self.td, self.ta, self.pd, self.pa, self.stn_list, self.can_return_base = read_flight(dataSet, 1)
        self.flightNum = len(self.pd)
        self.MinCT = 40
        self.MinRest = 660
        maxormins = [-1, 1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = self.flightNum * 2  # 初始化Dim（决策变量维数）xij 以及 环开始和结束航班
        self.singleDim = self.flightNum
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub =  [self.crewNum] * Dim  # 决策变量上界
        #ub.append(14842093) #14842094表示最大的可能的基地环起点和终点配对数量
        # 0：不在飞机上，1：正机长，2：副机长，3：替补副机长，4：乘机
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def can_take(self, crew_id, flight_id, table):
        '''
        描述：决定
        输入：
        '''
        if (self.pd[flight_id] == table[crew_id].CurStn) and (self.td[flight_id] - table[crew_id].ArrvTime >= self.MinCT) and (self.can_return_base[table[crew_id].base][flight_id] == 1):
            return True
        return False
    def update(self, crew_id, flight_id, table):
        table[crew_id].ArrvTime = self.ta[flight_id]
        table[crew_id].CurStn = self.pa[flight_id]
        table[crew_id].cur_flight = flight_id
    def crew_type(self, i):
        t = 0
        if self.is_cpt[i] == 1 and self.is_fo[i] == 1:
            t = 2
        elif self.is_cpt[i] == 1 and self.is_fo[i] == 0:
            t = 0
        else:
            t = 1
        return t
    def choose(self, choose_cpt, choose_fo, choose_cf, cpt_id = 0, fo_id = 0, cf_id1 = -1, cf_id2 = -1, flag = 0):
        t1, t2 = -1, -1
        if len(choose_cpt) > 0 and len(choose_fo) > 0:
            if flag == 0:
                cpt_id = choose_cpt[cpt_id % len(choose_cpt)]
                fo_id = choose_fo[fo_id % len(choose_fo)]
            t1 = 0
            t2 = 1
        elif len(choose_fo) > 0 and len(choose_cf) > 0:
            if flag == 0:
                cpt_id = choose_cf[cpt_id % len(choose_cf)]
                fo_id = choose_fo[fo_id % len(choose_fo)]
            else:
                cpt_id = cf_id1
            t1 = 2
            t2 = 1
        elif len(choose_cpt) > 0 and len(choose_cf) > 0:
            if flag == 0:
                cpt_id = choose_cpt[cpt_id % len(choose_cpt)]
                fo_id = choose_cf[fo_id % len(choose_cf)]
            else:
                fo_id = cf_id1
            t1 = 0
            t2 = 2
        elif len(choose_cf) > 1:
            if flag == 0:
                cpt_id = choose_cf[cpt_id % len(choose_cf)]
                if choose_cf[(fo_id) % len(choose_cf)] == cpt_id:
                    fo_id = choose_cf[(fo_id + 1) % len(choose_cf)]
                else:
                    fo_id = choose_cf[(fo_id) % len(choose_cf)]
            else:
                cpt_id = cf_id1
                fo_id = cf_id2
            t1 = 2
            t2 = 2
        return t1, t2, cpt_id, fo_id
    def aimFunc(self, pop):
        Vars = pop.Phen
        N = pop.sizes
        x = Vars[:,].astype(np.int32)
        obj = np.zeros((N, 3))
        path = np.zeros((N, self.crewNum * self.flightNum), np.int8)
        circle_constr = np.zeros((N, 1))
        constraint = []
        for ind in range(N):
            pool = deepcopy(self.stn_pool)
            for stn in self.stn_list:
                if (pool.get(stn) == None):
                    pool[stn] = []
                    for k in range(3):
                        pool[stn].append(set())
                    pool[stn].append(0)
            table = [item(self.base[i]) for i in range(self.crewNum)]
            table = np.array(table)
            open_list = set()
            for j in range(self.flightNum):
                if pool[self.pd[j]][3] == 0:
                    continue
                choose_cpt = []
                choose_fo = []
                choose_cf = []
                cpt_id = x[ind, j * 2 + 0]
                fo_id = x[ind, j * 2 + 1]
                if (cpt_id == fo_id):
                    continue
                for i in pool[self.pd[j]][0]:
                    if self.can_take(i, j, table):
                        choose_cpt.append(i)
                for i in pool[self.pd[j]][1]:
                    if self.can_take(i, j, table):
                        choose_fo.append(i)
                for i in pool[self.pd[j]][2]:
                    if self.can_take(i, j, table):
                        choose_cf.append(i)
                t1, t2, cpt_id, fo_id = self.choose(choose_cpt, choose_fo, choose_cf, cpt_id, fo_id)
                if t1 != -1 and  t2 != -1:
                    #由遗传算法决定是否带走已经处于基地的人
                    '''
                    if ((table[cpt_id].base == table[cpt_id].CurStn and table[cpt_id].cur_flight != -1) or \
                        (table[fo_id].base == table[fo_id].base == table[fo_id].CurStn and table[fo_id].cur_flight != -1)) and \
                        (x[ind, j * 2 + 0] == 0 or x[ind, j * 2 + 1] == 0):
                        continue
                    '''
                    open_list.add(j)
                    obj[ind, 0] += 1
                    obj[ind, 2] += t2 == 2
                    pool[self.pd[j]][t1].remove(cpt_id)
                    pool[self.pd[j]][3] -= 1
                    pool[self.pa[j]][t1].add(cpt_id)
                    pool[self.pa[j]][3] += 1
                    pool[self.pd[j]][t2].remove(fo_id)
                    pool[self.pd[j]][3] -= 1
                    pool[self.pa[j]][t2].add(fo_id)
                    pool[self.pa[j]][3] += 1
                    path[ind][cpt_id + j * self.crewNum] = 1
                    if t2 == 2:
                        path[ind][fo_id + j * self.crewNum] = 3
                    else:
                        path[ind][fo_id + j * self.crewNum] = 2
                    self.update(cpt_id, j, table)
                    self.update(fo_id, j, table)
            cnt = 0
            for stn in self.stn_list:
                if pool[stn][3] == 0:
                    continue
                base_pool = dict()
                for base in ['HOM', 'TGD', 'NKX']:
                    if (base_pool.get(base) == None):
                            base_pool[base] = []
                            for k in range(3):
                                base_pool[base].append(set())
                for k in range(3):
                    for i in pool[stn][k]:
                        if (stn == table[i].base):
                            continue               
                        base_pool[table[i].base][k].add(i)
                for base in ['HOM', 'TGD', 'NKX']:       
                    choose_cpt = []
                    choose_fo = []
                    choose_cf = []
                    min_cpt = -1
                    min_fo = -1
                    min_cf1 = -1
                    min_cf2 = -1
                    for i in base_pool[base][0]:
                        choose_cpt.append(i)
                        if min_cpt == -1 or table[i].cur_flight < table[min_cpt].cur_flight:
                            min_cpt = i
                    for i in base_pool[base][1]:
                        choose_fo.append(i)
                        if min_fo == -1 or table[i].cur_flight < table[min_fo].cur_flight:
                            min_fo = i
                    for i in base_pool[base][2]:
                        choose_cf.append(i)
                        if min_cf1 == -1 or table[i].cur_flight < table[min_cf1].cur_flight:
                            min_cf2 = min_cf1
                            min_cf1 = i
                        if (min_cf2 == -1 or table[min_cf2].cur_flight > table[i].cur_flight) and min_cf1 != i:
                            min_cf2 = i
                    t1, t2, cpt_id, fo_id = self.choose(choose_cpt, choose_fo, choose_cf, min_cpt, min_fo, min_cf1, min_cf2, 1)
                    if t1 != -1 and t2 != -1:
                        last = max(table[cpt_id].cur_flight, table[fo_id].cur_flight)
                        people_list = [cpt_id, fo_id]
                        for j in range(last + 1, self.flightNum):
                            if stn != self.pd[j] or j in open_list:
                                continue
                            can = True
                            for i in people_list:
                                can &= self.can_take(i, j, table)
                            if not can:
                                continue
                            last = j
                            obj[ind, 0] += 1
                            open_list.add(j)
                            for i in people_list:
                                self.update(i, j, table)
                                t = self.crew_type(i)
                                pool[self.pd[j]][t].remove(i)
                                pool[self.pd[j]][3] -= 1
                                pool[self.pa[j]][t].add(i)
                                pool[self.pa[j]][3] += 1
                                if i == cpt_id:
                                    path[ind, i + j * self.crewNum] = 1
                                elif  i == fo_id:
                                    if self.is_cpt[fo_id] == 1:
                                        path[ind, i + j * self.crewNum] = 3
                                        obj[ind, 2] += 1
                                    else:
                                        path[ind, i + j * self.crewNum] = 2
                            if(self.pa[j] == base):
                                break
            #####
            for stn in self.stn_list:
                if pool[stn][3] == 0:
                    continue      
                choose_cpt = []
                choose_fo = []
                choose_cf = []
                min_cpt = -1
                min_fo = -1
                min_cf1 = -1
                min_cf2 = -1
                for i in pool[stn][0]:
                    if stn == table[i].base:
                        continue
                    choose_cpt.append(i)
                    if min_cpt == -1 or table[i].cur_flight < table[min_cpt].cur_flight:
                        min_cpt = i
                for i in pool[stn][1]:
                    if stn == table[i].base:
                        continue
                    choose_fo.append(i)
                    if min_fo == -1 or table[i].cur_flight < table[min_fo].cur_flight:
                        min_fo = i
                for i in pool[stn][2]:
                    if stn == table[i].base:
                        continue
                    choose_cf.append(i)
                    if min_cf1 == -1 or table[i].cur_flight < table[min_cf1].cur_flight:
                        min_cf2 = min_cf1
                        min_cf1 = i
                    if (min_cf2 == -1 or table[min_cf2].cur_flight > table[i].cur_flight) and min_cf1 != i:
                        min_cf2 = i
                t1, t2, cpt_id, fo_id = self.choose(choose_cpt, choose_fo, choose_cf, min_cpt, min_fo, min_cf1, min_cf2, 1)
                if t1 != -1 and t2 != -1:
                    last = max(table[cpt_id].cur_flight, table[fo_id].cur_flight)
                    people_list = [cpt_id, fo_id]
                    for j in range(last + 1, self.flightNum):
                        if stn != self.pd[j] or j in open_list:
                            continue
                        can = True
                        for i in people_list:
                            can &= self.can_take(i, j, table)
                        if not can:
                            continue
                        last = j
                        obj[ind, 0] += 1
                        open_list.add(j)
                        for i in people_list:
                            self.update(i, j, table)
                            t = self.crew_type(i)
                            pool[self.pd[j]][t].remove(i)
                            pool[self.pd[j]][3] -= 1
                            pool[self.pa[j]][t].add(i)
                            pool[self.pa[j]][3] += 1
                            if i == cpt_id:
                                path[ind, i + j * self.crewNum] = 1
                            elif  i == fo_id:
                                if self.is_cpt[fo_id] == 1:
                                    path[ind, i + j * self.crewNum] = 3
                                    obj[ind, 2] += 1
                                else:
                                    path[ind, i + j * self.crewNum] = 2
                        if(self.pa[j] == base):
                            break
            ####
            for stn in self.stn_list:
                if pool[stn][3] == 0:
                    continue
                for k in range(3):
                    for i in pool[stn][k]: 
                        if (table[i].CurStn == table[i].base):
                            continue
                        last = table[i].cur_flight
                        for j in range(last + 1, self.flightNum):
                            if (j not in open_list) or (not self.can_take(i, j, table)):
                                continue
                            if self.can_take(i, j, table):
                                last = j
                                path[ind, i + j * self.crewNum] = 4
                                self.update(i, j, table)
                                obj[ind, 1] += 1
                                if(self.pa[j] == table[i].base):
                                    break
                        if(self.pa[last] != table[i].base):
                            cnt += 1
            circle_constr[ind] += cnt
        pop.ObjV = obj
        pop.CV = circle_constr
        pop.Vars = path
class item:
    def __init__(self, base):
        self.ArrvTime = -10000
        self.CurStn = base
        self.base = base
        self.cur_flight = -1
