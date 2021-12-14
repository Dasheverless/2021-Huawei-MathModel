import numpy as np
import geatpy as ea
from pre_process import *
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy

class Problem2(ea.Problem):  # 继承Problem父类
    def __init__(self, dataSet = 'A'):
        name = 'Problem2'  # 初始化name（函数名称，可以随意设置）
        M = 5  # 初始化M（目标维数）
        self.is_cpt, self.is_fo, self.base, self.duty_cost, self.pair_cost, self.stn_pool = read_crew(dataSet)
        self.crewNum = len(self.is_cpt)
        self.dptrDate, self.arrvDate, self.td, self.ta, self.pd, self.pa, self.stn_list, self.can_return_base = read_flight(dataSet, 1)
        self.flightNum = len(self.pd)
        self.MinCT = 40
        self.MaxBlk = 600
        self.MaxDP = 720
        self.MinRest = 660
        maxormins = [-1, 1, 1, 1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = self.flightNum * 2  # 初始化Dim（决策变量维数）xij 以及 环开始和结束航班
        self.singleDim = self.flightNum
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
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
        描述：
            根据机组人员crew_id的信息决定是否乘坐航班flight_id
        输入：
            crew_id: 机组人员编号
            fligh_id: 航班编号
            table: 机组人员信息表
        输出:
            True：机组人员crew_id可以搭乘航班flight_id
            False: 机组人员crew_id不可以搭乘航班flight_id
        '''
        if (self.pd[flight_id] != table[crew_id].CurStn) or (self.td[flight_id] - table[crew_id].ArrvTime < self.MinCT) \
            or (self.can_return_base[table[crew_id].base][flight_id] != 1):
            return False
        if (self.dptrDate[flight_id] > table[crew_id].duty_date) and (self.td[flight_id] - table[crew_id].ArrvTime >= self.MinRest):
            return True
        elif (self.dptrDate[flight_id] == table[crew_id].duty_date) and \
            (table[crew_id].duty_time + self.ta[flight_id] - table[crew_id].ArrvTime <= self.MaxDP)\
                 and (table[crew_id].fly_time + self.ta[flight_id] - self.td[flight_id] <= self.MaxBlk):
            return True
        return False
    def update(self, crew_id, flight_id, table):
        '''
        描述：
            更新机组人员crew_id搭乘航班flight_id后的信息
        输入：
            crew_id: 机组人员编号
            fligh_id: 航班编号
            table: 机组人员信息表
        输出：
            无
        '''
        #执勤日期是同一天的情况
        if (table[crew_id].duty_date == self.dptrDate[flight_id]):
            table[crew_id].all_duty_time += self.ta[flight_id] - table[crew_id].ArrvTime
            table[crew_id].duty_time += self.ta[flight_id] - table[crew_id].ArrvTime
            table[crew_id].fly_time += self.ta[flight_id] - self.td[flight_id]
            table[crew_id].duty_cost += (self.ta[flight_id] - table[crew_id].ArrvTime) * self.duty_cost[crew_id] / 60
        #执勤日期不是同一天则创建新的执勤
        else:
            table[crew_id].duty_date = self.dptrDate[flight_id]
            table[crew_id].all_duty_time += self.ta[flight_id] - self.td[flight_id]
            table[crew_id].duty_time = self.ta[flight_id] - self.td[flight_id]
            table[crew_id].fly_time = self.ta[flight_id] - self.td[flight_id]
            table[crew_id].duty_cost += (self.ta[flight_id] - self.td[flight_id]) * self.duty_cost[crew_id] / 60
        table[crew_id].ArrvTime = self.ta[flight_id]
        table[crew_id].CurStn = self.pa[flight_id]
        table[crew_id].cur_flight = flight_id
        '''
        调试信息
        table[crew_id].sol.append(flight_id)
        if self.pa[flight_id] == table[crew_id].base:
            table[crew_id].flight_in_base.append(flight_id)
        '''
    def crew_type(self, crew_id):
        '''
        描述：
            获取机组人员crew_id的类别
        输入：
            cre_id: 机组人员编号
        输出：
            0: 仅能当机长(简称机长)
            1：仅能当副机长
            2：既可以当机长也能当副机长(全能机长)
        '''
        t = 0
        if self.is_cpt[crew_id] == 1 and self.is_fo[crew_id] == 1:
            t = 2
        elif self.is_cpt[crew_id] == 1 and self.is_fo[crew_id] == 0:
            t = 0
        else:
            t = 1
        return t
    def choose(self, choose_cpt, choose_fo, choose_cf, cpt_id = 0, fo_id = 0, cf_id1 = -1, cf_id2 = -1, flag = 0):
        '''
        描述：
            根据给定的机长列表，副机长列表，全能列表以及对应id确定具体当机长和副机长的人员
        输入：
            choose_cpt: 待选非全能机长(简称机长)列表
            choose_fo: 待选副机长列表
            choose_fo: 待选全能机长列表
            cpt_id: 备选机长id
            fo_id: 备选副机长id
            cf_id1: 备选全能机长id1
            cf_id2： 备选全能机长id2
            flag: flag = 1: 备选由遗传算法给定， flag = 0: 备选由程序算法决定
        输出：
            cpt_id: 选定当机长的id
            fo_id: 选定副机长的id
            t1: 选定机长的类型
            t2: 选定副机长的类型
        '''
        t1, t2 = -1, -1
        #优先选副机长保证替补次数最少
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
        '''
        描述：
            计算遗传算法种群的目标函数及约束， 每次迭代执行一次
        输入：
            pop: 种群类，包含表现型， 基因型， 目标函数， 约束等变量
        输出：
            无
        '''
        Vars = pop.Phen#决策变量
        N = pop.sizes#种群大小
        x = Vars[:,].astype(np.int32)
        obj = np.zeros((N, 5))#目标函数值
        #状态变量，负责记录机组人员与航班的配置关系， 
        # 0代表不搭乘, 1代表担任机长， 2代表担任副机长，3代表担任替补副机长，4代表乘机
        path = np.zeros((N, self.crewNum * self.flightNum), np.int8)
        circle_constr = np.zeros((N, 1))#是否成环的约束，其值大小代表有多少人不成环
        for ind in range(N):       
            #pool[stn][t]是一个set(), 存放机场stn中类型为t的机组人员，t = 3则代表总人数
            #stn_pool是pool的初始状态，所有人都在其基地
            pool = deepcopy(self.stn_pool)
            #pool初始化
            for stn in self.stn_list:
                if (pool.get(stn) == None):
                    pool[stn] = []
                    for k in range(3):
                        pool[stn].append(set())
                    pool[stn].append(0)
            table = [item(self.base[i]) for i in range(self.crewNum)]
            table = np.array(table)
            open_list = set()
            #第0步：按时间顺序遍历航班，从其所在机场的人员中选择机长和副机长起飞
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
                    #如果机组人员i和航班j匹配才加入列表中
                    if self.can_take(i, j, table):
                        choose_cpt.append(i)
                for i in pool[self.pd[j]][1]:
                    if self.can_take(i, j, table):
                        choose_fo.append(i)
                for i in pool[self.pd[j]][2]:
                    if self.can_take(i, j, table):
                        choose_cf.append(i)
                #根据各类型列表选定机长和副机长
                t1, t2, cpt_id, fo_id = self.choose(choose_cpt, choose_fo, choose_cf, cpt_id, fo_id)
                #如果存在可以令航班起飞的机长和副机长
                if t1 != -1 and  t2 != -1:
                    #对于第2问和第3问的B数据集，由于约束复杂可能存在机组人员本来已经成环，航班j又将他带离机场的情况，
                    #这种情况发生在航班起飞时间较后时
                    #这里让遗传算法决定是否放弃带离已经成环的机组人员
                    if ((table[cpt_id].base == table[cpt_id].CurStn and table[cpt_id].cur_flight != -1) or \
                        (table[fo_id].base == table[fo_id].base == table[fo_id].CurStn and table[fo_id].cur_flight != -1)) and \
                        (x[ind, j * 2 + 0] == 0 or x[ind, j * 2 + 1] == 0):
                        continue
                    #加入可以起飞的航班列表
                    open_list.add(j)
                    #可以起飞的航班数 + 1
                    obj[ind, 0] += 1
                    #替补数加1
                    obj[ind, 4] += t2 == 2
                    #机场池移动，从出发机场删除对应人员，并加入到达机场
                    pool[self.pd[j]][t1].remove(cpt_id)
                    pool[self.pd[j]][3] -= 1
                    pool[self.pa[j]][t1].add(cpt_id)
                    pool[self.pa[j]][3] += 1
                    pool[self.pd[j]][t2].remove(fo_id)
                    pool[self.pd[j]][3] -= 1
                    pool[self.pa[j]][t2].add(fo_id)
                    pool[self.pa[j]][3] += 1
                    #更新航班与机组人员的配置信息
                    path[ind][cpt_id + j * self.crewNum] = 1
                    if t2 == 2:
                        path[ind][fo_id + j * self.crewNum] = 3
                    else:
                        path[ind][fo_id + j * self.crewNum] = 2
                    #更新机组人员的信息
                    self.update(cpt_id, j, table)
                    self.update(fo_id, j, table)
            cnt = 0
            #返回基地方法第一步
            #此步将各机场人员的滞留人员按基地匹配，若能成组(有足够的机长和副机长)
            #则去开未起飞的飞机(称为点亮)，让剩余的人乘机选择更多
            #若有很多选择，则选所有机组人员最后搭乘的航班中， 起飞航班最早的人，这样可以点亮更多的航班
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
                        #如果已经在基地，则跳过
                        if stn == table[i].base:
                            continue               
                        base_pool[table[i].base][k].add(i)
                for base in ['HOM', 'TGD', 'NKX']:       
                    choose_cpt = []
                    choose_fo = []
                    choose_cf = []
                    min_cpt = -1
                    min_fo = -1
                    #若从全能池中选择，就需要记录起飞航班最早和次早
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
                        #选中的两个人均不能往回飞，故需要从两人最后起飞时间最晚的航班后面开始遍历
                        last = max(table[cpt_id].cur_flight, table[fo_id].cur_flight)
                        people_list = [cpt_id, fo_id]
                        for j in range(last + 1, self.flightNum):
                            #若该航班不在当前机场出发，或航班已经在列表里了就跳过
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
                                        obj[ind, 4] += 1
                                    else:
                                        path[ind, i + j * self.crewNum] = 2
                            if(self.pa[j] == base):
                                break
            #返回基地方法第二步
            #第一步中存在基地相同但无法成组，不同基地的人却能成组的情况
            #第二步就是处理这种情况
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
                    #如果已经在基地，则跳过
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
                                    obj[ind, 4] += 1
                                else:
                                    path[ind, i + j * self.crewNum] = 2
                        if(self.pa[j] == base):
                            break
            #返回基地方法第三步
            #not_in_base = []，调试数据，不在基地的机组人员
            #最后一步再次遍历各机场滞留人员， 由于无法成组，此时只能搭乘已经起飞的航班
            for stn in self.stn_list:
                if pool[stn][3] == 0:
                    continue
                for k in range(3):
                    for i in pool[stn][k]: 
                        #如果已经在基地，则跳过
                        if (stn == table[i].base):
                            continue
                        last = table[i].cur_flight
                        for j in range(last + 1, self.flightNum):
                            if (j not in open_list) or (not self.can_take(i, j, table)):
                                continue
                            last = j
                            path[ind, i + j * self.crewNum] = 4
                            self.update(i, j, table)
                            obj[ind, 2] += 1
                            if(self.pa[j] == table[i].base):
                                break
                        if(self.pa[last] != table[i].base):
                            #尝试搭乘所有已经起飞的航班后仍不能回到基地
                            cnt += 1
                            #not_in_base.append(i)， 调试数据，不在基地的机组人员
            crew_duty_time = np.zeros(self.crewNum)
            #目标函数计算
            for i in range(self.crewNum):
                obj[ind, 1] += table[i].duty_cost
                crew_duty_time[i] = table[i].all_duty_time
            obj[ind, 3] = np.std(crew_duty_time)
            circle_constr[ind] += cnt
            #这里不给权重，最后在所有最优种群中再按优先级选择
            weight = [1] * 5#[1e8, 1e2, 1e5, 1, 1]
            for i in range(5):
                obj[ind, i] *= weight[i]
        pop.ObjV = obj
        pop.CV = circle_constr
        pop.Vars = path
class item:
    '''
    描述：
        记录机组人员的信息
    '''
    def __init__(self, base):
        self.ArrvTime = -10000#表示到达当前机场的时间
        self.CurStn = base #当前机场，初始在基地机场
        self.base = base#所在基地
        self.cur_flight = -1#搭乘的最后一趟航班
        self.duty_date = -60#当前执勤的日期
        self.duty_time = 0#一次执勤的执勤时长
        self.all_duty_time = 0#执勤总时长
        self.fly_time = 0#一次执勤的飞行时长
        self.duty_cost = 0#执勤总成本
        '''
        调试信息
        self.sol = [] #机组人员搭乘的航班列表
        self.flight_in_base = [] # 机组人员搭乘过的回到过基地的航班
        '''
