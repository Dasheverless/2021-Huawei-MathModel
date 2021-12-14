import numpy as np
import pandas as pd
MinCT = 40
def read_crew(dataSet):
    #读取表格数据
    crew_sheet = pd.read_csv('../data/机组排班Data ' + dataSet + '-Crew.csv')
    crew_num = crew_sheet.shape[0]
    crew = crew_sheet.values
    captain = np.arange(crew_num)
    FirstOfficer = np.arange(crew_num)
    base = []
    DutyCost = np.arange(crew_num)
    ParingCost = np.arange(crew_num)
    stn_pool = dict()
    for i in range(crew_num):
        if crew[i][1] == 'Y':
            captain[i] = 1
        else:
            captain[i] = 0
        if crew[i][2] == 'Y':
            FirstOfficer[i] = 1
        else:
            FirstOfficer[i] = 0
        base.append(crew[i][4])
        if stn_pool.get(crew[i][4]) == None:
            stn_pool[crew[i][4]] = []
            for k in range(3):
                stn_pool[crew[i][4]].append(set())
            stn_pool[crew[i][4]].append(0)
        if (captain[i] == 1 and FirstOfficer[i] == 0):
            stn_pool[crew[i][4]][0].add(i)
            stn_pool[crew[i][4]][3] += 1
        elif (captain[i] == 0 and FirstOfficer[i] == 1):
            stn_pool[crew[i][4]][1].add(i)
            stn_pool[crew[i][4]][3] += 1
        else:
            stn_pool[crew[i][4]][2].add(i)
            stn_pool[crew[i][4]][3] += 1
        DutyCost[i] = crew[i][5]
        ParingCost[i] = crew[i][6]
    return captain, FirstOfficer, base, DutyCost, ParingCost, stn_pool

def read_flight(dataSet, flag = 0):
    if(flag == 1):
        flight_sheet = pd.read_csv('../data/flight_info_' + dataSet + '.csv')
        flight_list = flight_sheet.values
        can_return_to_base = dict()
        if dataSet == 'A':
            base_list = ['NKX']
        else:
            base_list = ['HOM', 'TGD']
        for stn in base_list:
            data = pd.read_csv('../data/' + stn + 'return_base.csv', header = None)
            data = data.values
            can_return_to_base[stn] = data
        stn_list = set()
        for i in range(len(flight_list)):
            stn_list.add(flight_sheet['DptrStn'].values[i])
            stn_list.add(flight_sheet['ArrvStn'].values[i])
        return flight_sheet['DptrDate'].values, flight_sheet['ArrvDate'].values, \
            flight_sheet['DptrTime'].values, flight_sheet['ArrvTime'].values, \
            flight_sheet['DptrStn'].values, flight_sheet['ArrvStn'].values, \
            stn_list, can_return_to_base
    flight_sheet = pd.read_csv('../data/机组排班Data ' + dataSet + '-Flight.csv')
    flight_num = flight_sheet.shape[0]
    flight_list = flight_sheet.values
    class item:
        def __init__(self):
            self.FltNum = ""
            self.DptrTime = 0
            self.ArrvTime = 0
            self.DptrStn = ""
            self.ArrvStn = ""
    flight = [item() for i in range(flight_num)]

    #将日期和时刻整理成时间戳
    DptrDate = np.arange(flight_num)
    ArrvDate = np.arange(flight_num)
    DptrClock = np.arange(flight_num)
    ArrvClock = np.arange(flight_num)
    for i in range(flight_num):
        DptrList = flight_list[i][1]
        if DptrList[4] == '/':
            date = DptrList[2:4]
        else:
            date = DptrList[2]
        DptrDate[i] = int(date)
        ArrvList = flight_list[i][4]
        if ArrvList[4] == '/':
            date = ArrvList[2:4]
        else:
            date = ArrvList[2]
        ArrvDate[i] = int(date)

        DptrList = flight_list[i][2]
        if DptrList[1] == ':':
            hour = int(DptrList[0])
            minute = int(DptrList[2:])
        else:
            hour = int(DptrList[0:2])
            minute = int(DptrList[3:])
        DptrClock[i] = hour*60 + minute
        ArrvList = flight_list[i][5]
        if ArrvList[1] == ':':
            hour = int(ArrvList[0])
            minute = int(ArrvList[2:])
        else:
            hour = int(ArrvList[0:2])
            minute = int(ArrvList[3:])
        ArrvClock[i] = hour * 60 + minute
    DptrStart = np.min(DptrDate)
    ArrvStart = np.min(ArrvDate)

    for i in range(flight_num):
        flight[i].FltNum = flight_list[i][0]
        flight[i].DptrStn = flight_list[i][3]
        flight[i].ArrvStn = flight_list[i][6]
        flight[i].DptrTime = (DptrDate[i]-DptrStart)*24*60 + DptrClock[i]
        flight[i].ArrvTime = (ArrvDate[i]-ArrvStart)*24*60 + ArrvClock[i]
    flight = sorted(flight, key = lambda x:(x.DptrTime, x.ArrvTime))
    flt_num = []
    dptr_stn = []
    arrv_stn = []
    dptr_time = np.arange(flight_num)
    arrv_time = np.arange(flight_num)
    for i in range(flight_num):
        flt_num.append(flight[i].FltNum)
        dptr_stn.append(flight[i].DptrStn)
        arrv_stn.append(flight[i].ArrvStn)
        dptr_time[i] = flight[i].DptrTime
        arrv_time[i] = flight[i].ArrvTime
    return flt_num, dptr_stn, arrv_stn, dptr_time,arrv_time

def adjacent_list(flight, dataSet):
    flight_num = len(flight)
    adjacent_port = [[] for i in range(flight_num)]
    adjacent_port = [[] for i in range(flight_num)]
    for i in range(flight_num):
        for j in range(i + 1, flight_num):
            if flight[i].ArrvStn == flight[j].DptrStn and flight[j].DptrTime - flight[i].ArrvTime >= MinCT:
                adjacent_port[i].append(j)
    is_connected = np.zeros((flight_num, flight_num), np.int8)
    is_visited = np.zeros(flight_num, np.bool)
    def dfs(i):
        is_visited[i] = True
        for j in adjacent_port[i]:
            if(is_visited[j]):
                is_connected[i] = (is_connected[i] | is_connected[j])
                continue
            is_connected[i, j] = True
            dfs(j)
    for i in range(flight_num):
        dfs(i)
    is_connected = pd.DataFrame(is_connected)
    is_connected.to_csv('../data/连通表'+dataSet+'.csv', header = 0, index = 0)
def pair_gen(dataSet):
    is_connected = pd.read_csv('../data/连通表'+dataSet+'.csv',header=None)
    flight = pd.read_csv('../data/flight_info_'+dataSet+'.csv')
    is_connected = is_connected.values
    d = flight['DptrStn'].values
    a = flight['ArrvStn'].values
    n = len(flight)
    pair_list = dict()
    pair_len = dict()
    if dataSet == 'A':
        base_list = ['NKX']
    else:
        base_list = ['HOM', 'TGD']
    can_return_base = dict()
    can_return_base_len = dict()
    for stn in base_list:
        pair_list[stn] = - np.ones((n, n), np.int8)
        pair_len[stn] = np.zeros(n, np.int32)
        can_return_base_len[stn] = np.zeros(n, np.int32)
        can_return_base[stn] = np.zeros(n, np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if a[j] == stn and is_connected[i][j] != 0:
                    can_return_base[stn][j] = 1
                    can_return_base[stn][i] = 1
                    can_return_base_len[stn][i] += 1
                    if(d[i] == stn):
                        pair_list[stn][i, pair_len[stn][i]] = j
                        pair_len[stn][i] += 1
    for stn in base_list:
        #data = pd.DataFrame(pair_list[stn])
        #data.to_csv('../data/' + stn + '.csv', header = None, index = None)
        #data = pd.DataFrame(pair_len[stn])
        #data.to_csv('../data/' + stn + 'len.csv', header = None, index = None)
        data = pd.DataFrame(can_return_base[stn])
        data.to_csv('../data/' + stn + 'return_base.csv', header = None, index = None)
        data = pd.DataFrame(can_return_base_len[stn])
        data.to_csv('../data/' + stn + 'return_base_len.csv', header = None, index = None)
def max_pair():
    l = 0
    base_list = ['HOM', 'TGD', 'NKX']
    for stn in base_list:
        data = pd.read_csv('../data/' + stn + '.csv', header=None)
        data = data.values
        l = max(l, len(data))
        print(len(data))
def read_pair(stn):
    data1 = pd.read_csv('../data/' + stn + '.csv', header=None)
    data2 = pd.read_csv('../data/' + stn + 'len.csv', header=None)
    return data1.values, data2.values
def read_conn(dataSet):
    data = pd.read_csv('../data/连通表'+dataSet+'.csv',header=None)
    return data.values
def test_var(prob, dataSet):
    is_cpt, is_fo, base, duty_cost, pair_cost, stn_pool = read_crew(dataSet)
    crewNum = len(is_cpt)
    dptrDate, arrvDate, td, ta, _pd, pa, stn_list, can_return_base = read_flight(dataSet, 1)
    x = pd.read_csv('../第' + str(prob) + '题/Result/' + dataSet + '/Vars.csv',header=None)
    x = x.values
    N = x.shape[0]
    cnt = 0
    if dataSet == 'A':
        crewNum = 21
        flightNum = 206
    else:
        crewNum = 465
        flightNum = 13954
    for ind in range(N):
        flag = 0
        obj1 = 0
        obj2 = 0
        obj3 = 0
        for i in range(crewNum):
            last = -1
            for j in range(flightNum):
                if x[ind, i + j * crewNum] == 0:
                    continue
                if last == -1:
                    flag |= _pd[j] != base[i]
                    last = j
                    continue
                flag |= (td[j] - ta[last]) < MinCT
                if flag == 1:
                    print(ind, i, j)
                    ok = 1
                flag |= pa[last] != _pd[j]
                last = j
            flag |= pa[last] != base[i]
            if flag == 1:
                print(ind, i, j)
                ok = 1
        for j in range(flightNum):
            sumC = 0
            sumO = 0
            sub = 0
            ride = 0
            for i in range(crewNum):
                if x[ind, i + j * crewNum] == 0:
                    continue
                sumC += x[ind, i + j * crewNum] == 1
                if (x[ind, i + j * crewNum] == 1 and is_cpt[i] == 0):
                    flag |= 1
                if (x[ind, i + j * crewNum] == 2 or x[ind, i + j * crewNum] == 3) and is_fo[i] == 0:
                    flag |= 1
                sumO += x[ind, i + j * crewNum] == 2 or x[ind, i + j * crewNum] == 3
                sub += x[ind, i + j * crewNum] == 3
                ride += x[ind, i + j * crewNum] == 4
            if (sumC == 0 and sumO != 0) or (sumC != 0 and sumO == 0) or (sumC > 1 or sumO > 1):
                flag |= 1
            if flag == 1:
                print(ind, i, j)
                ok = 1
            obj1 += (sumC == 1 and sumO == 1)
            obj2 += ride
            obj3 += sub
        #print(obj1, obj2, obj3)
        cnt += flag
    return cnt <= 0
def test1():
    print('x')
if __name__ == '__main__':
    test_var('A')
    print(test_var('A'))