#!usr/bin/env python
# _*_ coding:utf8 _*_
import numpy as np
import json
import os
from do_Internal.cal_break_link import monitor_break
from collections import Counter
import time
from multiprocessing import Pool
import copy
from other_script.util import mkdir

as_rel = {}
as_customer = {}
as_peer = {}
numberAsns = {}


class cut_week_point():
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.graph = {}
        self.res = {}
        pass

    def from_npz_create_graph(self):
        '''
        存routingTree 【【前向点】【后向点】】 后向点为空说>明就脱离了routingTree
        '''
        m = np.load(self.file_name)
        self.row = [str(i) for i in m['row']]
        self.col = [str(i) for i in m['col']]
        link = list(zip(self.row, self.col))
        for l in link:
            a, b = l
            if a not in self.graph:
                self.graph[a] = [[], []]
            if b not in self.graph:
                self.graph[b] = [[], []]
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        self.res[''] = len(self.graph)

    def monitor_cut_node(self, queue):
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

    def yield_cur_link(self, depth=None):
        begin_as = self.file_name.split('/')[-1].split('.')[0]
        if not depth:
            for _as in self.graph:
                yield str(_as)
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
        begin_index = 0
        while _as in self.row[begin_index:]:
            index = self.row.index(_as, begin_index)
            s.add(
                state_num[as_rel[str(self.col[index]) + ' ' + str(self.row[index])]])
            begin_index = index + 1
        if len(s) == 0:
            print(self.file_name, _as, _as in self.row, _as in self.col)
        return s


class FindOptimizeLink():
    '''
    1、遍历集合A中每个AS的路由树中的AS‘，弄一个hash，记录AS’在路由树里面的出现次数，并根据AS’的链接种类，记录他的state: 1(中/p2p) 2(上+[中]/c2p) 3([上]+[中]+下/p2c)
    2、对于不在AS’中的AS’’，计算他的路由树中AS有几个在集合B里面。根据集合B的链接种类，记录state:  1(中/p2p) 2(上+[中]/c2p) 3([上]+[中]+下/p2c)
    state 1后跟1或2或3，2后跟2
    所以目前原则是前面的as走向优先向上，后面的as走向优先向下
    '''

    def __init__(self, rtpath, break_link, week_point, dsn_path, file_names) -> None:
        '''
        rtpath: 存放npz的路径
        break_link: [[begin_as, end_as],...]
        '''
        self.rtpath = rtpath
        self.file_name = file_names
        self.break_link = break_link
        self.week_point = week_point
        self.dsn_path = dsn_path

    def break_link_begin_rtree_frequency(self, depth=None):
        hash_dict = {}
        for begin_as, _ in self.break_link:
            if begin_as in hash_dict:
                continue
            if str(begin_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(
                    self.rtpath, str(begin_as) + '.npz'))
            elif 'dcomplete' + str(begin_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(
                    self.rtpath, 'dcomplete' + str(begin_as) + '.npz'))
            else:
                continue

            cwp.from_npz_create_graph()
            print(self.week_point)
            cwp.monitor_cut_node(copy.deepcopy(self.week_point))

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
            for _as in cwp.yield_cur_link(depth=depth):
                if str(begin_as) == str(_as):
                    hash_dict[begin_as][1].append(int(_as))
                    hash_dict[begin_as][2].append(int(_as))
                else:
                    state = cwp.cal_state(_as)
                    if 1 in state:
                        state = 1
                    else:
                        state = 2
                    hash_dict[begin_as][state].append(int(_as))

        self.begin_hash_dict = hash_dict

    def break_link_end_rtree_frequency(self, depth=None):
        hash_dict = {}
        for _, end_as in self.break_link:
            if end_as in hash_dict:
                continue
            if str(end_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(
                    self.rtpath, str(end_as) + '.npz'))
            elif 'dcomplete' + str(end_as) + '.npz' in self.file_name:
                cwp = cut_week_point(os.path.join(
                    self.rtpath, 'dcomplete' + str(end_as) + '.npz'))
            else:
                print(str(end_as) + '.npz not exist in ')
                continue

            cwp.from_npz_create_graph()
            cwp.monitor_cut_node(copy.deepcopy(self.week_point))

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

        with open(self.dsn_path + '.begin_hash_dict.json', 'w') as f:
            json.dump(self.begin_hash_dict, f)
        with open(self.dsn_path + '.end_hash_dict.json', 'w') as f:
            json.dump(self.end_hash_dict, f)

    def find_opt_link(self):

        '''
        设计不同策略，找到需要建立链接的link
        1、数量
        2、数量+金额
        3、数量+金额+距离

        贪心搜索策略 输入：被破坏链接左/右集合、as连通集合
        1、贪心搜索左边：找到一个左边as 能链接到最多 被破坏链接左集合 的as
        2、贪心搜索右边：同步骤1左边as连通后 恢复最多数量的被破坏链接
        3、重复步骤1-2，直到所有破坏链接均被覆盖
        '''

        '''
        1. /   2. (/) -  或 (/ - )\ 

        反向关系结果  1’.\\   2’. - (\)  或 / (- \)

        无谷匹配规则
        1: 1’,2’
        2: 1’

        匹配优先state: 1' - 2'
        '''

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
                    print(_as, ' not in')
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
                        if _as in as_peer and numberAsns[_as2] <= as_peer[_as][0] * 1.2 and \
                                numberAsns[_as2] >= as_peer[_as][1] * 0.8:
                            cost = min(cost, b)
                        elif _as2 in as_peer and numberAsns[_as] <= as_peer[_as2][0] * 1.2 and \
                                numberAsns[_as] >= as_peer[_as2][1] * 0.8:
                            cost = min(cost, b)
                elif relation == 'p2c' and numberAsns[_as] >= numberAsns[_as2] * 0.95:
                    cost = min(cost, c)
                elif relation == 'c2p' and numberAsns[_as2] >= numberAsns[_as] * 0.95:
                    cost = min(cost, c)
            return cost

        state = {'1': {'1': ['p2p', 'p2c', 'c2p'], '2': ['c2p']}, \
                 '2': {'1': ['p2c']}}
        with open(self.dsn_path + '.begin_hash_dict.json', 'r') as f:
            self.begin_hash_dict = json.load(f)
        with open(self.dsn_path + '.end_hash_dict.json', 'r') as f:
            self.end_hash_dict = json.load(f)

        country_name = os.path.basename(self.dsn_path).split('_')[0]
        if NODE_VALUE != 'basic':
            with open(os.path.join(as_importance_path, country_name + '.json'), 'r') as f:
                as_importance_weight = json.load(f)
            as_importance_weight = {line[0]: [line[1], line[2]] for line in as_importance_weight}
            if NODE_VALUE == 'user':
                as_importance_weight_min = min(
                    [as_importance_weight[k][0] for k in as_importance_weight])
            else:
                as_importance_weight_min = min(
                    [as_importance_weight[k][1] for k in as_importance_weight])
        else:
            as_importance_weight_min = 0.5

        def search_end_state(opt_right_as, data):
            if opt_right_as in data:
                return True
            return False

        res = []
        res.append(['', '', self.week_point,
                    copy.deepcopy(self.break_link), 0, 0])
        while self.break_link:
            max_benefit_all, opt_left_as, opt_right_as, opt_begin_state, opt_end_state = float(
                "-inf"), '', '', '', ''
            for begin_state in state:
                for end_state in state[begin_state]:
                    benefit, left_as, right_as = 0, '', ''

                    count_dict = {}
                    for begin_as, _ in self.break_link:
                        if str(begin_as) not in self.begin_hash_dict:
                            continue
                        v = cal_node_value(begin_as)
                        nodes = self.begin_hash_dict[str(
                            begin_as)][begin_state]
                        for _nodes in nodes:
                            if _nodes not in count_dict:
                                count_dict[_nodes] = 0
                            count_dict[_nodes] += v

                    left_as, left_max_benefit = '', -1

                    if count_dict:
                        for _nodes, _value in count_dict.items():
                            if _value > left_max_benefit:
                                left_as, left_max_benefit = _nodes, _value
                    else:
                        continue

                    count_dict = {}
                    for begin_as, end_as in self.break_link:
                        if left_as in self.begin_hash_dict[str(begin_as)][begin_state]:
                            if str(end_as) not in self.end_hash_dict:
                                continue
                            for _as in set(self.end_hash_dict[str(end_as)][end_state]):
                                if _as == left_as:
                                    continue
                                if _as not in count_dict:
                                    count_dict[_as] = 0
                                count_dict[_as] += (cal_node_value(begin_as) +
                                                    cal_node_value(end_as))

                    right_max_benefit, right_as = float('-inf'), ''
                    for _as in count_dict:
                        cost = cal_cost(left_as, _as, begin_state, end_state)
                        if right_max_benefit <= count_dict[_as] - cost:
                            right_max_benefit, right_as = count_dict[_as] - cost, _as

                    cost = cal_cost(left_as, right_as, begin_state, end_state)
                    n = len(self.break_link)
                    for i in range(n):
                        begin_as, end_as = self.break_link[i]
                        if str(begin_as) not in self.begin_hash_dict or str(end_as) not in self.end_hash_dict:
                            continue
                        if left_as in self.begin_hash_dict[str(begin_as)][begin_state] and \
                                right_as in self.end_hash_dict[str(end_as)][end_state]:
                            benefit += (cal_node_value(begin_as) +
                                        cal_node_value(end_as))

                    benefit -= cost

                    if benefit > max_benefit_all:
                        max_benefit_all, opt_left_as, opt_right_as, opt_begin_state, opt_end_state, opt_cost = \
                            benefit, left_as, right_as, begin_state, end_state, cost

            if opt_right_as == '' or opt_left_as == '':
                break
            n = len(self.break_link)
            opt_re_link = []
            for i in range(n - 1, -1, -1):
                begin_as, end_as = self.break_link[i]
                if str(begin_as) not in self.begin_hash_dict or str(end_as) not in self.end_hash_dict:
                    continue
                if opt_left_as in self.begin_hash_dict[str(begin_as)][opt_begin_state] and \
                        opt_right_as in self.end_hash_dict[str(end_as)][opt_end_state]:
                    opt_re_link.append(self.break_link[i])
                    del self.break_link[i]
                elif opt_right_as in self.begin_hash_dict[str(begin_as)][opt_end_state] and \
                        opt_left_as in self.end_hash_dict[str(end_as)][opt_begin_state]:
                    opt_re_link.append([self.break_link[i][1], self.break_link[i][0]])
                    del self.break_link[i]

            res.append([[opt_left_as, opt_right_as], [opt_begin_state,
                                                      opt_end_state, opt_cost], opt_re_link, n, len(self.break_link)])
        return res


def find_optimize_link_pool(_dsn_path, cname):
    global as_rel, as_customer, as_peer, numberAsns
    optimize_link_path = os.path.join(_dsn_path, 'optimize_link')
    dsn_path = os.path.join(optimize_link_path, 'floyed/')
    rtree_path = os.path.join(_dsn_path, 'rtree')
    old_break_dsn_path = dsn_path

    mkdir(optimize_link_path)
    mkdir(dsn_path)

    if cname in ['BR', 'US', 'RU']:
        return

    if not os.path.exists(os.path.join(rtree_path, cname, 'as-rel.txt')):
        print(os.path.join(rtree_path, cname, 'as-rel.txt') + ' as-rel not exist')
        return
    if os.path.exists(os.path.join(dsn_path, cname + '.opt_add_link_rich.json')):
        print(os.path.join(dsn_path, cname + '.opt_add_link_rich.json'))
        print(cname + ' exist')
        return
    if os.path.exists(os.path.join(old_break_dsn_path, cname) + '.break_link.json'):
        with open(os.path.join(old_break_dsn_path, cname) + '.break_link.json', 'r') as f:
            week_point_and_break_link = json.load(f)
    else:
        mb = monitor_break()
        week_point_and_break_link = mb.main_2(os.path.join(
            rtree_path, cname), os.path.join(dsn_path, cname))

    week_point_and_break_link = list(week_point_and_break_link.items())
    Res = {}
    range_num = len(week_point_and_break_link)
    file_names = os.listdir(os.path.join(rtree_path, cname))
    for i in range(range_num):
        if isinstance(week_point_and_break_link[i][0], int):
            week_point = [int(week_point_and_break_link[i][0])]
        elif isinstance(week_point_and_break_link[i][0], str):
            week_point = list(
                map(int, week_point_and_break_link[i][0].split(' ')))
        else:
            week_point = list(week_point_and_break_link[i][0])
        break_link = week_point_and_break_link[i][1]
        if len(break_link) == 0:
            break
        else:
            print('第 %s组 %s len => %s' % (str(i), cname, len(break_link)))

        fol = FindOptimizeLink(os.path.join(rtree_path, cname),
                               break_link, week_point, os.path.join(dsn_path, cname + '_' + str(i)), file_names)
        fol.break_link_begin_rtree_frequency()
        fol.break_link_end_rtree_frequency()
        res = fol.find_opt_link()
        with open(os.path.join(dsn_path, cname + '.' + str(i) + '.opt_add_link_rich.json'), 'w') as f:
            json.dump(res, f)
        for line in res:
            if str(line[0]) + ' ' + str(line[1]) not in Res:
                Res[str(line[0]) + ' ' + str(line[1])] = []
            Res[str(line[0]) + ' ' + str(line[1])] += line[2]
    with open(os.path.join(dsn_path, cname + '.opt_add_link_rich.json'), 'w') as f:
        json.dump(Res, f)


NODE_VALUE = 'basic'  # 'basic' 'user' 'domain'
as_importance_path = ''

# [按受影响的节点数量倒序取前多少名,破坏节点数量]
# cost
a, b, c = 0, 0, 50


def find_optimize_link(txt_path, _dsn_path, cone_path, cc_list, _as_importance_path):
    global as_importance_path
    as_importance_path = _as_importance_path
    with open(cone_path, 'r') as f:
        numberAsns = json.load(f)

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
                value = numberAsns[data[1]
                ] if data[1] in numberAsns else 1
                as_peer[data[0]] = [max(as_peer[data[0]][0], value), min(
                    as_peer[data[0]][1], value)]
                value = numberAsns[data[0]
                ] if data[0] in numberAsns else 1
                as_peer[data[1]] = [max(as_peer[data[1]][0], value), min(
                    as_peer[data[1]][1], value)]

            line = fp.readline().strip()
    for cname in cc_list:
        find_optimize_link_pool(_dsn_path, cname)
