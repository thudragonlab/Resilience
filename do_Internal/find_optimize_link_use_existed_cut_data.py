#!usr/bin/env python
# _*_ coding:utf8 _*_
import multiprocessing
from typing import Dict, List, Set, NewType, Tuple, Iterator, TypeVar, Union
import numpy as np
from do_Internal.data_analysis import as_rela_txt_dont_save
import json
import os
from do_Internal.cal_break_link import monitor_break
from collections import Counter
import time
from multiprocessing import Pool
import copy
from do_Internal.create_rtree_after_optimize import add_npz_and_monitor_cut
from do_Internal.use_monitor_data_as_weakPoint import make_weak_point
from util import mkdir, record_launch_time, record_launch_time_and_param

as_rel = {}
as_customer = {}
as_peer = {}
numberAsns = {}
gl_num_list = []

ForwardPointType = NewType('ForwardPointType', List[str])
BackwardPointType = NewType('BackwardPointType', List[str])
AsnType = TypeVar('AsnType', str, int)
ASRelationType = NewType('ASRelationType', Dict[int, List[AsnType]])
UserWeightType = NewType('UserWeightType', int)
DomainWeightType = NewType('DomainWeightType', int)


class cut_week_point():

    def __init__(self, file_name: str) -> None:

        self.file_name: str = file_name
        self.graph: Dict[AsnType, Tuple[ForwardPointType, BackwardPointType]] = {}
        self.res = {}
        pass

    def from_npz_create_graph(self):
        '''
        存routingTree 【【前向点】【后向点】】 后向点为空说>明就脱离了routingTree
        '''
        m = np.load(self.file_name)
        self.row: List[str] = [str(i) for i in m['row']]
        self.col: List[str] = [str(i) for i in m['col']]
        link = list(zip(self.row, self.col))
        for l in link:
            a, b = l
            if a not in self.graph:
                self.graph[a] = [[], []]
            if b not in self.graph:
                self.graph[b] = [[], []]
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        #monitor_cut_node(self.graph, list(self.graph.keys())[2])
        self.res[''] = len(self.graph)

    def monitor_cut_node(self, queue: List[int]):
        res = []
        queue = list(map(str, queue))
        n = len(queue)
        for i in range(n - 1, -1, -1):
            node = queue[i]
            if node not in self.graph:
                queue.remove(node)
                continue

            for i in self.graph[node][1]:
                self.graph[i][0].remove(node)

            self.graph[node][1] = []
        while queue:
            n = queue.pop(0)
            res.append(n)
            if n not in self.graph:
                continue

            for i in self.graph[n][0]:
                if i in self.graph:
                    self.graph[i][1].remove(n)
                    if len(self.graph[i][1]) == 0:
                        queue.append(i)
            del self.graph[n]
        return res

    def yield_cur_link(self, depth=None) -> Iterator:
        begin_as = self.file_name.split('/')[-1].split('.')[0]
        if not depth:
            for _as in self.graph:
                yield str(_as)
                # for end_as in self.graph[begin_as][1]:
                #     yield int(begin_as), int(end_as)
        else:
            _as = self.file_name.split('/')[-1].split('.')[0]
            if not _as[0].isdigit():
                _as = _as[9:]
            stack = [[_as]]
            s = set()
            s.add(_as)
            for _ in range(depth):
                cur = stack[-1]
                stack.append([])
                for _cur_as in cur:
                    for line in self.graph[_cur_as]:
                        for _as in line:
                            if _as not in s:
                                s.add(_as)
                                stack[-1].append(_as)
                                yield str(_as)

    def cal_state(self, _as):
        '''
        1. /
        2. (/) -
        3. (/ - )\ 

        反向关系结果
        1’.\\
        2’. - (\)
        3’. / (- \)

        无谷匹配规则
        1: 1’,2’,3’
        2: 1’,3’
        3: 1’
        '''
        global as_rel
        s = set()
        state_num = {'c2p': 1, 'p2p': 2, 'p2c': 3}
        # file_name = os.path.join(file_path, 'dcomplete'+str(begin_as)+'.npz')
        # m = np.load(file_name)
        # row = [str(i) for i in m['row']]
        # col = [str(i) for i in m['col']]
        begin_index = 0
        while _as in self.row[begin_index:]:
            index = self.row.index(_as, begin_index)
            s.add(state_num[as_rel[str(self.col[index]) + ' ' + str(self.row[index])]])
            begin_index = index + 1
        return s


class FindOptimizeLink():
    '''
    1、遍历集合A中每个AS的路由树中的AS‘，弄一个hash，记录AS’在路由树里面的出现次数，并根据AS’的链接种类，记录他的state: 1(中/p2p) 2(上+[中]/c2p) 3([上]+[中]+下/p2c)
    2、对于不在AS’中的AS’’，计算他的路由树中AS有几个在集合B里面。根据集合B的链接种类，记录state:  1(中/p2p) 2(上+[中]/c2p) 3([上]+[中]+下/p2c)
    state 1后跟1或2或3，2后跟2
    所以目前原则是前面的as走向优先向上，后面的as走向优先向下
    '''

    # def __init__(self, rtpath, break_link, week_point, raw_graph, node_index, dsn_path) -> None:
    def __init__(self, rtpath: str,as_topo, dsn_path: str,cc2as_list_path:str) -> None:
        '''
        rtpath: 存放npz的路径
        break_link: [[begin_as, end_as],...]
        '''
        self.rtpath: str = rtpath
        self.file_name = os.listdir(self.rtpath)
        # self.break_link: List[Tuple[str, str]] = break_link
        # self.week_point: List[int] = week_point
        self.dsn_path: str = dsn_path
        with open(cc2as_list_path,'r') as ff:
            self.all_as_list = json.load(ff)
        self.as_topo = as_topo
        self.graph = self.create_graph(as_topo)
        
    
    def create_graph(self,as_topo):
        graph = {}
        for _as in as_topo:
            peer, customer = as_topo[_as]
            if _as not in graph:
                graph[_as] = [[],customer]
            for i in customer:
                if i not in graph:
                    graph[i] = [[],[]]
                graph[i][0].append(_as)

                if i not in graph[_as][1]:
                    graph[_as][1].append(i)

            for pi in peer:
                if pi not in graph:
                    graph[pi] = [[],[]]
                if pi < _as:
                    graph[_as][1].append(pi)
                    graph[pi][0].append(_as)
                else:
                    graph[_as][0].append(pi)
                    graph[pi][1].append(_as)

        return graph

    def break_link_begin_rtree_frequency(self, depth=None):
        hash_dict: Dict[str, ASRelationType] = {}
        for begin_as in self.all_as_list:
            if begin_as in hash_dict:
                continue
            if str(begin_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(self.rtpath, str(begin_as) + '.npz'))
            elif 'dcomplete' + str(begin_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(self.rtpath, 'dcomplete' + str(begin_as) + '.npz'))
            else:
                continue

            cwp.from_npz_create_graph()
            # cwp.monitor_cut_node(copy.deepcopy(self.week_point))

            global total
            hash_dict[begin_as] = {1: [], 2: []}
            '''
            for _as in cwp.yield_cur_link(depth = depth):
            # m = np.load(os.path.join(self.rtpath, str(begin_as)+'.npz'))
            # link = list(zip(m['row'], m['col']))
            # for as1, as2 in link:
                hash_dict[begin_as][self.cal_state(begin_as,_as,left_perference = 2,right_perference = 3)].append(int(_as))            
            '''
            '''
            1. /   2. (/) - 或 (/ - )\ 

            反向关系结果  1’.\\   2’. -(\)  或 /(- \)

            无谷匹配规则
            1: 1’,2’
            2: 1’
        

            匹配优先state: 1 - 2
            '''
            for _as in cwp.yield_cur_link(depth=depth):  # 遍历以beginAS为根结点的路由树生成graph破坏后的所有节点
                if str(begin_as) == str(_as):
                    hash_dict[begin_as][1].append(int(_as))
                    hash_dict[begin_as][2].append(int(_as))
                else:
                    state = cwp.cal_state(_as)  # 计算_as在topo数据(txt)里面的所有的关系
                    if 1 in state:
                        state = 1
                    else:
                        state = 2
                    hash_dict[begin_as][state].append(int(_as))

        self.begin_hash_dict = hash_dict
        with open(self.dsn_path + '.begin_hash_dict.json', 'w') as f:
            json.dump(self.begin_hash_dict, f)
        # hash_dict[begin_as][1]表示和begin_as是p2c关系的AS列表
        # hash_dict[begin_as][2]表示和begin_as是p2p关系的AS列表

    def break_link_end_rtree_frequency(self, depth=None):
        hash_dict: Dict[str, ASRelationType] = {}
        for end_as in self.all_as_list:
            # print('end_as',_, end_as)
            if end_as in hash_dict:
                continue
            if str(end_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(self.rtpath, str(end_as) + '.npz'))
            elif 'dcomplete' + str(end_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(self.rtpath, 'dcomplete' + str(end_as) + '.npz'))
            else:
                continue

            cwp.from_npz_create_graph()
            # cwp.monitor_cut_node(copy.deepcopy(self.week_point))

            hash_dict[end_as] = {1: [], 2: []}

            global total
            '''
            for _as in cwp.yield_cur_link(depth = depth):
            # m = np.load(os.path.join(self.rtpath, str(begin_as)+'.npz'))
            # link = list(zip(m['row'], m['col']))
            # for as1, as2 in link:
                hash_dict[end_as][self.cal_state(_as,end_as,left_perference = 3,right_perference = 2)].append(int(_as))
                total += b-a
            '''
            '''
            1. /   2. (/) -  或 (/ - )\ 

            反向关系结果  1’.\\   2’. - (\) 或 / (- \)

            无谷匹配规则
            1: 1’,2’
            2: 1’

            匹配优先state: 1' - 3' - 2'
            '''
            for _as in cwp.yield_cur_link(depth=depth):
                if str(end_as) == str(_as):
                    hash_dict[end_as][1].append(int(_as))
                    hash_dict[end_as][2].append(int(_as))
                else:
                    state = cwp.cal_state(_as)
                    if 1 in state:
                        state = 1
                    else:
                        state = 2
                    hash_dict[end_as][state].append(int(_as))

        self.end_hash_dict = hash_dict
        # hash_dict[end_as][1]表示和end_as是p2c关系的AS列表
        # hash_dict[end_as][2]表示和end_as是p2p关系的AS列表

        
        with open(self.dsn_path + '.end_hash_dict.json', 'w') as f:
            json.dump(self.end_hash_dict, f)

    # def find_opt_link(self):
    #     '''
    #     设计不同策略，找到需要建立链接的link
    #     1、数量
    #     2、数量+金额
    #     3、数量+金额+距离

    #     贪心搜索策略 输入：被破坏链接左/右集合、as连通集合
    #     1、贪心搜索左边：找到一个左边as 能链接到最多 被破坏链接左集合 的as
    #     2、贪心搜索右边：同步骤1左边as连通后 恢复最多数量的被破坏链接
    #     3、重复步骤1-2，直到所有破坏链接均被覆盖
    #     '''
    #     '''
    #     1. /   2. (/) -  或 (/ - )\ 

    #     反向关系结果  1’.\\   2’. - (\)  或 / (- \)

    #     无谷匹配规则
    #     1: 1’,2’
    #     2: 1’

    #     匹配优先state: 1' - 2'
    #     '''

    #     def cal_node_value(_as):
    #         if isinstance(_as, int):
    #             _as = str(_as)
    #         if NODE_VALUE != 'basic':
    #             if _as in as_importance_weight:
    #                 if NODE_VALUE == 'user':
    #                     return as_importance_weight[_as][0]
    #                 else:
    #                     return as_importance_weight[_as][1]
    #             else:
    #                 return as_importance_weight_min
    #         return 0.5

    #     def cal_cost(_as, _as2, begin_state, end_state):
    #         global numberAsns, as_peer
    #         _as, _as2 = str(_as), str(_as2)
    #         cost = float('inf')
    #         for relation in state[begin_state][end_state]:
    #             if _as not in numberAsns:
    #                 numberAsns[_as] = 1
    #             if _as2 not in numberAsns:
    #                 numberAsns[_as2] = 1
    #             if relation == 'p2p':
    #                 if 0.5 <= float(numberAsns[_as]) / float(numberAsns[_as2]) <= 2:
    #                     cost = min(cost, a)
    #                 else:
    #                     if _as in as_peer and numberAsns[_as2] <= as_peer[_as][0]*1.2 and \
    #                         numberAsns[_as2] >= as_peer[_as][1]*0.8:
    #                         cost = min(cost, b)
    #                     elif _as2 in as_peer and numberAsns[_as] <= as_peer[_as2][0]*1.2 and \
    #                         numberAsns[_as] >= as_peer[_as2][1]*0.8:
    #                         cost = min(cost, b)
    #             elif relation == 'p2c' and numberAsns[_as] >= numberAsns[_as2] * 0.95:
    #                 cost = min(cost, c)
    #             elif relation == 'c2p' and numberAsns[_as2] >= numberAsns[_as] * 0.95:
    #                 cost = min(cost, c)
    #         return cost
        
    #     # state = {'1':{'1':['p2c']}, '2':{'1':['p2c']}}
    #     # state = {'1':{'1':['p2p']}}
    #     # state = {'1': {'1': ['c2p'],'2':['c2p']}}
    #     state = {'1':{'1':['p2p', 'p2c', 'c2p'],'2':['c2p']}, \
    #         '2':{'1':['p2c']}}
        # with open(self.dsn_path + '.begin_hash_dict.json', 'r') as f:
        #     self.begin_hash_dict = json.load(f)
        # with open(self.dsn_path + '.end_hash_dict.json', 'r') as f:
        #     self.end_hash_dict = json.load(f)

    #     country_name = os.path.basename(self.dsn_path).split('_')[0]
    #     if NODE_VALUE != 'basic':
    #         with open(os.path.join(as_importance_path, country_name + '.json'), 'r') as f:
    #             as_importance_weight: List[Tuple[str, int, int]] = json.load(f)
    #         as_importance_weight: Dict[str, Tuple[UserWeightType, DomainWeightType]] = {
    #             line[0]: [line[1], line[2]]
    #             for line in as_importance_weight
    #         }
    #         if NODE_VALUE == 'user':
    #             as_importance_weight_min = min([as_importance_weight[k][0] for k in as_importance_weight])  # 最小的user权重值
    #         else:
    #             as_importance_weight_min = min([as_importance_weight[k][1] for k in as_importance_weight])  # 最小的domain权重值
    #     else:
    #         as_importance_weight_min = 0.5

    #     res = []
    #     res.append(['', '', self.week_point, copy.deepcopy(self.break_link), 0, 0])
    #     while self.break_link:
    #         max_benefit_all, opt_left_as, opt_right_as, opt_begin_state, opt_end_state = float("-inf"), '', '', '', ''
    #         for begin_state in state:
    #             for end_state in state[begin_state]:
    #                 benefit, left_as, right_as = 0, '', ''

    #                 count_dict: Dict[AsnType, int] = {}  # 所有和被破坏连接有关的AS的value
    #                 for begin_as, end_as in self.break_link:  # 遍历这个国家下所有被模拟破坏路由树的所有被破坏链接的左节点
    #                     if str(begin_as) not in self.begin_hash_dict:
    #                         continue
    #                     v = (cal_node_value(begin_as) + cal_node_value(end_as))
    #                     nodes = self.begin_hash_dict[str(begin_as)][begin_state]
    #                     for _nodes in nodes:
    #                         if _nodes not in count_dict:
    #                             count_dict[_nodes] = 0
    #                         count_dict[_nodes] += v

    #                 #     for _as in set(self.begin_hash_dict[str(begin_as)][begin_state]):
    #                 #         if _as not in count_dict: count_dict[_as] = 0
    #                 #         count_dict[_as]+=1
    #                 left_as, left_max_benefit = '', -1

                    
    #                 # 寻找价值最大的能连上begin_as的左节点
    #                 if count_dict:
    #                     for _nodes, _value in count_dict.items():
    #                         if _value > left_max_benefit:
    #                             left_as, left_max_benefit = _nodes, _value
    #                 else:
    #                     continue
    #                 count_dict: Dict[AsnType, int] = {}
    #                 for begin_as, end_as in self.break_link:
    #                     if left_as in self.begin_hash_dict[str(begin_as)][begin_state]:  # 如果价值最大的左节点能连上begin_as
    #                         if str(end_as) not in self.end_hash_dict:
    #                             continue
    #                         for _as in set(self.end_hash_dict[str(end_as)][end_state]):
    #                             if _as == left_as:
    #                                 continue
    #                             if _as not in count_dict:
    #                                 count_dict[_as] = 0
    #                             count_dict[_as] += (cal_node_value(begin_as) + cal_node_value(end_as))


    #                 right_max_benefit, right_as = float('-inf'), ''
    #                 for _as in count_dict:
    #                     cost = cal_cost(left_as, _as, begin_state, end_state)
    #                     if '%s %s' % (left_as,_as) in as_rel:
    #                         continue
    #                     if right_max_benefit <= count_dict[_as] - cost:
    #                         right_max_benefit, right_as = count_dict[_as] - cost, _as

    #                 cost = cal_cost(left_as, right_as, begin_state, end_state)
    #                 n = len(self.break_link)
    #                 for i in range(n):
    #                     begin_as, end_as = self.break_link[i]
    #                     if str(begin_as) not in self.begin_hash_dict or str(end_as) not in self.end_hash_dict:
    #                         continue
    #                     if left_as in self.begin_hash_dict[str(begin_as)][begin_state] and \
    #                             right_as in self.end_hash_dict[str(end_as)][end_state]:
    #                         benefit += (cal_node_value(begin_as) + cal_node_value(end_as))

    #                 benefit -= cost

    #                 if benefit > max_benefit_all:
    #                     max_benefit_all, opt_left_as, opt_right_as, opt_begin_state, opt_end_state, opt_cost = \
    #                         benefit, left_as, right_as, begin_state, end_state, cost

    #         if opt_right_as == '' or opt_left_as == '':
    #             break
    #         n = len(self.break_link)
    #         opt_re_link:Tuple[str,str] = []
    #         for i in range(n - 1, -1, -1):
    #             begin_as, end_as = self.break_link[i]
    #             if str(begin_as) not in self.begin_hash_dict or str(end_as) not in self.end_hash_dict:
    #                 continue
    #             if opt_left_as in self.begin_hash_dict[str(begin_as)][opt_begin_state] and \
    #                     opt_right_as in self.end_hash_dict[str(end_as)][opt_end_state]:
    #                 opt_re_link.append(self.break_link[i])
    #                 del self.break_link[i]
    #             elif opt_right_as in self.begin_hash_dict[str(begin_as)][opt_end_state] and \
    #                     opt_left_as in self.end_hash_dict[str(end_as)][opt_begin_state]:
    #                 opt_re_link.append([self.break_link[i][1], self.break_link[i][0]])
    #                 del self.break_link[i]

    #         res.append([[opt_left_as, opt_right_as], [opt_begin_state, opt_end_state, opt_cost], opt_re_link, n, len(self.break_link)]) 
    #          #          [优化链接左节点, 优化链接右节点], [优化链接左节点连begin_as关系,优化链接右节点连end_as关系,优化成本],重新连接的链接,原来断开链接数,优化后断开链接数
    #     return res


    def new_find_opt_link(self):
            max_num = max(gl_num_list)
            state = {'1':{'1':['p2p', 'p2c', 'c2p'],'2':['c2p']}, \
                '2':{'1':['p2c']}}

            country_name = os.path.basename(self.dsn_path).split('_')[0]
            if NODE_VALUE != 'basic':
                with open(os.path.join(as_importance_path, country_name + '.json'), 'r') as f:
                    as_importance_weight: List[Tuple[str, int, int]] = json.load(f)
                as_importance_weight: Dict[str, Tuple[UserWeightType, DomainWeightType]] = {
                    line[0]: [line[1], line[2]]
                    for line in as_importance_weight
                }
                if NODE_VALUE == 'user':
                    as_importance_weight_min = min([as_importance_weight[k][0] for k in as_importance_weight])  # 最小的user权重值
                else:
                    as_importance_weight_min = min([as_importance_weight[k][1] for k in as_importance_weight])  # 最小的domain权重值
            else:
                as_importance_weight_min = 0.5

            res = []
            def cal_node_value(_as):
                if isinstance(_as, int):
                    _as = str(_as)
                if NODE_VALUE != 'basic':
                    if _as in as_importance_weight:
                        if NODE_VALUE == 'user':
                            return as_importance_weight[_as][0]
                        else:
                            return as_importance_weight[_as][1]
                    else:
                        return as_importance_weight_min
                return 0.5

            def cal_cost(_as, _as2, begin_state, end_state):
                global numberAsns, as_peer
                _as, _as2 = str(_as), str(_as2)
                cost = float('inf')
                for relation in state[begin_state][end_state]:
                    if _as not in numberAsns:
                        numberAsns[_as] = 1
                    if _as2 not in numberAsns:
                        numberAsns[_as2] = 1
                    if relation == 'p2p':
                        if 0.5 <= float(numberAsns[_as]) / float(numberAsns[_as2]) <= 2:
                            cost = min(cost, a)
                        else:
                            if _as in as_peer and numberAsns[_as2] <= as_peer[_as][0]*1.2 and \
                                numberAsns[_as2] >= as_peer[_as][1]*0.8:
                                cost = min(cost, b)
                            elif _as2 in as_peer and numberAsns[_as] <= as_peer[_as2][0]*1.2 and \
                                numberAsns[_as] >= as_peer[_as2][1]*0.8:
                                cost = min(cost, b)
                    elif relation == 'p2c' and numberAsns[_as] >= numberAsns[_as2] * 0.95:
                        cost = min(cost, c)
                    elif relation == 'c2p' and numberAsns[_as2] >= numberAsns[_as] * 0.95:
                        cost = min(cost, c)
                return cost
            

            # for _ in self.break_link:
            for begin_state in state:
                for end_state in state[begin_state]:
                    
                    count_dict: Dict[AsnType, int] = {}  # 所有和被破坏连接有关的AS的value
                    for _as in self.as_topo:  # 遍历这个国家下所有被模拟破坏路由树的所有被破坏链接的左节点
                        if begin_state == '1' and end_state == '1': # p2p
                            my_index = 0
                        else:
                            my_index = 1
                        for _nodes in self.as_topo[_as][my_index]:
                        # for _nodes in set(self.begin_hash_dict[str(begin_as)][begin_state]):
                            
                            if _nodes not in count_dict:
                                count_dict[_nodes] = 0
                            if _as not in count_dict:
                                count_dict[_as] = 0
                            count_dict[_nodes] += cal_node_value(_as)
                            count_dict[_as] += cal_node_value(_nodes)


                    # # right_count_dict: Dict[AsnType, int] = {}
                    # for end_as in self.begin_hash_dict: 
                    #     # if str(end_as) not in self.end_hash_dict:
                    #     #     continue
                    #     for _as in set(self.end_hash_dict[str(end_as)][end_state]):
                    #         if _as not in count_dict:
                    #             count_dict[_as] = 0
                    #         count_dict[_as] += cal_node_value(end_as)


                    for left_as in self.all_as_list:
                        


                        for right_as in self.all_as_list:
                            
                            #如果left和right一样或者在原来的topo里面存在 就跳过
                            if left_as == right_as or '%s %s' % (left_as,right_as) in as_rel :
                                continue

                            
                            cost = cal_cost(left_as, right_as, begin_state, end_state)
                            if left_as not in count_dict: # cc2as里面的某个节点不在input的topo数据中
                                left_as_value = 0 #价值设置为0
                            else:
                                left_as_value = count_dict[left_as]
                            if right_as not in count_dict: # cc2as里面的某个节点不在input的topo数据中
                                right_as_value = 0 #价值设置为0
                            else:
                                right_as_value = count_dict[right_as]

                            benefit =left_as_value + right_as_value - cost
                            res.append([[left_as, right_as], [begin_state, end_state],benefit]) 
                            #[优化链接左节点, 优化链接右节点], [优化链接左节点连begin_as关系,优化链接右节点连end_as关系,优化成本],收益
                    # for begin_as, end_as in self.break_link: 
                    #     cost = cal_cost(begin_as, end_as, begin_state, end_state)
                    #     benefit =cal_node_value(begin_as) + cal_node_value(end_as) - cost
                    #     res.append([[begin_as, end_as], [begin_state, end_state],benefit]) 


            res.sort(key=lambda x:x[2],reverse=True)   
            # res = res[:max_num]
            print('len(res)',len(res))
            return res

@record_launch_time_and_param(1)
def find_optimize_link_pool(output_path,m, cname):
    global as_rel, as_customer, as_peer, numberAsns
    # q, cname = s.split(' ')
    _dsn_path = os.path.join(output_path,m)
    optimize_link_path = os.path.join(_dsn_path, 'optimize_link')
    dsn_path = os.path.join(optimize_link_path, 'floyed/')
    rtree_path = os.path.join(_dsn_path, 'rtree')
    old_break_dsn_path = dsn_path
    opt_add_link_rich_path =os.path.join(dsn_path,'opt_add_link_rich',cname) 
    hash_dict_path = os.path.join(dsn_path,'hash_dict')
    mkdir(optimize_link_path)
    mkdir(dsn_path)
    mkdir(opt_add_link_rich_path)
    mkdir(hash_dict_path)

    # os.popen('mkdir '+dsn_floyed_path)
    if cname in ['BR', 'US', 'RU']:
        return

    if not os.path.exists(os.path.join(rtree_path, cname, 'as-rel.txt')):
        return
    Res_set:Set[str] = set()
    json_data = as_rela_txt_dont_save(os.path.join(rtree_path, cname, 'as-rel.txt'))
     

    fol = FindOptimizeLink(os.path.join(rtree_path, cname),json_data,
                            os.path.join(hash_dict_path, cname),os.path.join(gl_cc2as_path,'%s.json' % cname))
    res = fol.new_find_opt_link() 
    for line in res:
        state_list = line[1]
        l_as,r_as = line[0]
        if state_list == ['1','1'] and str([[r_as,l_as],state_list]) in Res_set:
            continue
        Res_set.add(str(line))

    
    res_list = list(map(lambda x:eval(x) , Res_set))
    res_list.sort(key=lambda x:x[2],reverse=True)
    with open(os.path.join(dsn_path, cname + '.opt_add_link_rich.json'), 'w') as f:
        json.dump(res_list, f)
    

    add_npz_and_monitor_cut(output_path, m, cname, gl_num_list,os.path.join(gl_cc2as_path,'%s.json' % cname),gl_numberAsns)


NODE_VALUE = 'basic'  # 'basic' 'user' 'domain'
as_importance_path = ''

# sample_num = '3'

# [按受影响的节点数量倒序取前多少名,破坏节点数量]
# sample_num_dict = {'1': [400, 1], '2': [
#         180, 2], '3': [100, 3], '4': [80, 4]}

# cost
a, b, c = 0, 0, 0


@record_launch_time
def find_optimize_link(txt_path, output_path,_type, cone_path, cc_list, _as_importance_path,num_list,cc2as_path):
    global as_importance_path
    global gl_num_list
    global gl_cc2as_path
    global gl_numberAsns
    gl_cc2as_path = cc2as_path
    gl_num_list = num_list
    as_importance_path = _as_importance_path
    # input = []
    with open(cone_path, 'r') as f:
        numberAsns = json.load(f)
        gl_numberAsns = numberAsns

    with open(txt_path) as fp:
        line = fp.readline().strip()
        while line:
            if line[0] == '#' or line[0] == '(':
                line = fp.readline().strip()
                continue
            data = line.split('|')
            if len(data) != 3:
                continue
            if data[-1][-1] == '1':
                as_rel[data[0] + ' ' + data[1]] = 'p2c'
                as_rel[data[1] + ' ' + data[0]] = 'c2p'
                if data[0] not in as_customer:
                    as_customer[data[0]] = 0
                as_customer[data[0]] += 1
            else:
                as_rel[data[0] + ' ' + data[1]] = 'p2p'
                as_rel[data[1] + ' ' + data[0]] = 'p2p'
                if data[0] not in as_peer:
                    as_peer[data[0]] = [float('-inf'), float('inf')]
                if data[1] not in as_peer:
                    as_peer[data[1]] = [float('-inf'), float('inf')]
                value = numberAsns[data[1]] if data[1] in numberAsns else 1
                as_peer[data[0]] = [max(as_peer[data[0]][0], value), min(as_peer[data[0]][1], value)]
                value = numberAsns[data[0]] if data[0] in numberAsns else 1
                as_peer[data[1]] = [max(as_peer[data[1]][0], value), min(as_peer[data[1]][1], value)]

            line = fp.readline().strip()
    pool = Pool(multiprocessing.cpu_count())
    for cname in cc_list:
        find_optimize_link_pool (
              output_path,
            _type,
            cname,
        )
        # pool.apply_async(find_optimize_link_pool, (
        #     output_path,
        #     _type,
        #     cname,
        # ))
    pool.close()
    pool.join()
