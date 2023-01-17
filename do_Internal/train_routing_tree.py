#!usr/bin/env python
# _*_ coding:utf8 _*_
import copy
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
from typing import Any, Dict, List, Union
from scipy import stats, sparse
from statsmodels.stats.multicomp import MultiComparison
import numpy as np
import time
import json
from do_Internal.data_analysis import as_rela_txt_dont_save
from do_Internal.sort_rank import groud_truth_based_anova_for_single_country, country_internal_rank, internal_survival
from multiprocessing import Pool
from other_script.util import mkdir, record_launch_time, record_launch_time_and_param

'''
3、量化并排序 (internal_security.py)
'''

SUFFIX = 'optimize_link'
RESULT_SUFFIX = SUFFIX + '/new_optimize_result'
SORT_DSN_PATH_SUFFIX = RESULT_SUFFIX + '/anova'


def add_link_to_npz(add_link_file, old_npz_file, relas_file, dsn_npz_file, add_link_relas_file, add_link_num):
    state = {'c2p': 0, 'p2p': 1, 'p2c': 2, 0: 'c2p', 1: 'p2p', 2: 'p2c'}
    match_state = {'1': {'1': 'p2p', '2': 'c2p'}, '2': {'1': 'p2c'}}

    graph = {}
    with open(relas_file) as fp:
        line = fp.readline().strip()
        while line:
            if line[0].isdigit():
                line = line.split('|')
                if line[0] not in graph:
                    graph[line[0]] = {'p2p': [], 'p2c': [], 'c2p': []}
                if line[1] not in graph:
                    graph[line[1]] = {'p2p': [], 'p2c': [], 'c2p': []}
                if line[-1][-1] == '1':
                    graph[line[0]]['p2c'].append(line[1])
                    graph[line[1]]['c2p'].append(line[0])
                else:
                    graph[line[0]]['p2p'].append(line[1])
                    graph[line[1]]['p2p'].append(line[0])
            line = fp.readline().strip()

    npz_file = np.load(old_npz_file)
    row = [str(i) for i in npz_file['row']]
    col = [str(i) for i in npz_file['col']]
    link = list(zip(row, col))
    cur_graph = {}
    for line in link:
        if line[0] not in cur_graph:
            cur_graph[line[0]] = {'pre': [], 'nxt': []}
        if line[1] not in cur_graph:
            cur_graph[line[1]] = {'pre': [], 'nxt': []}
        cur_graph[line[0]]['pre'].append(line[1])
        cur_graph[line[1]]['nxt'].append(line[0])

    with open(add_link_file, 'r') as f:
        m = json.load(f)
    add_link = []
    if isinstance(m, dict):
        for line in m:
            if len(line) < 2:
                continue
            line = line.split(' ')
            line[0] = line[0][1:-1]
            line[1] = line[1][:-1]
            begin_state = line[2][2:-2]
            end_state = line[3][1:-2]
            add_link.append([str(line[0]), str(line[1]), state[match_state[begin_state][end_state]]])
            if len(add_link) > add_link_num:
                break
    elif isinstance(m, list):
        for line in m:
            add_link.append([str(line[0]), str(line[1]), state[match_state[begin_state][end_state]]])
            if len(add_link) > add_link_num:
                break

    def find_pre_state(node):
        if node not in graph or node not in cur_graph:
            return False
        cur_state = 2
        if len(cur_graph[node]['pre']) == 0:
            return 0
        for _node in cur_graph[node]['pre']:
            min_state = cur_state
            for i in range(cur_state - 1, -1, -1):
                if node in graph[_node][state[i]]:
                    min_state = min(i, cur_state)
                    if min_state == 0:
                        return 0
                    break
            cur_state = min_state
        return cur_state

    add_link_relas = {}
    n = 0
    begin_n = len(add_link)
    while add_link:
        link = add_link.pop(0)
        if link[0] == link[1]:
            continue
        if len(link) == 3:
            s = link[2]
        else:
            s0 = find_pre_state(link[0])
            s1 = find_pre_state(link[1])
            if not s0 or not s1: continue
            if s0 > s1: link[0], link[1] = link[1], link[0]
            s = min(s0, s1)
        add_link_relas[str(link[0]) + ' ' + str(link[1])] = state[s]
        if link[1] not in cur_graph or link[0] not in cur_graph:
            continue
        cur_graph[link[0]]['nxt'].append(link[1])
        try:
            cur_graph[link[1]]['pre'].append(link[0])
        except Exception as e:
            exit()

        n += 1
        for _s in range(s, 3):
            for _node in graph[link[1]][state[_s]]:
                if _node not in cur_graph[link[1]]['pre'] and _node not in cur_graph[link[1]]['nxt'] \
                        and _node in cur_graph:
                    add_link.append([link[1], _node, _s])

    with open(add_link_relas_file, 'w') as f:
        json.dump(add_link_relas, f)


def generate_new_rela(add_link_file: str, relas_file: str, add_link_num: int, cc_as_list_path: str, add_link_path: str,
                      asn: str) -> Dict[int, List[List[int]]]:
    state = {'c2p': 0, 'p2p': 1, 'p2c': 2, 0: 'c2p', 1: 'p2p', 2: 'p2c'}
    match_state = {'1': {'1': 'p2p', '2': 'c2p'}, '2': {'1': 'p2c'}}

    def create_relas(as_rela: str) -> Dict[int, List[List[int]]]:
        relas = {}
        with open(cc_as_list_path, 'r') as f:
            cclist = json.load(f)
        for c in cclist:
            relas[c] = [[], [], []]

        for c in relas:
            if c in as_rela:
                relas[c][2] += [i for i in as_rela[c][0] if i in cclist]
                for i in relas[c][2]:
                    relas[i][2].append(c)
                relas[c][1] += [i for i in as_rela[c][1] if i in cclist]
                for i in relas[c][1]:
                    relas[i][0].append(c)

        for c in list(relas.keys()):
            if relas[c] == [[], [], []]:
                del relas[c]

        return relas

    json_data = as_rela_txt_dont_save(relas_file)
    relas = create_relas(json_data)
    with open(add_link_file, 'r') as f:
        print(add_link_file)
        m = json.load(f)
    add_link = set()
    for line in m:
        if len(line) < 2:
            continue
        line = line.split(' ')
        print(line)
        left_as = line[0][1:-1]
        right_as = line[1][:-1]
        begin_state = line[2][2:-2]
        end_state = line[3][1:-2]

        link_state = state[match_state[begin_state][end_state]]
        if link_state == 1:
            if int(left_as) > int(right_as):
                right_as = line[0][1:-1]
                left_as = line[1][:-1]

        add_link.add('-'.join([str(left_as), str(right_as), str(state[match_state[begin_state][end_state]])]))
        if len(add_link) > add_link_num:
            break

    add_link_list = list(map(lambda x: x.split('-'), add_link))
    with open(os.path.join(add_link_path, f'add_link-{asn}.{add_link_num}.json'), 'w') as f:
        json.dump(add_link_list, f)

    while add_link_list:
        link = add_link_list.pop(0)
        left_as = link[0]
        right_as = link[1]
        print(f'{add_link_num} add Link {link} ')
        if link[0] == link[1]:
            continue
        s = int(link[2])
        if s == 1:
            if right_as not in relas[left_as][2]: relas[left_as][2].append(right_as)
            if left_as not in relas[right_as][2]: relas[right_as][2].append(left_as)
        elif s == 0:
            if right_as not in relas[left_as][0]: relas[left_as][0].append(right_as)
            if left_as not in relas[right_as][1]: relas[right_as][1].append(left_as)
        elif s == 2:
            if right_as not in relas[left_as][1]: relas[left_as][1].append(right_as)
            if left_as not in relas[right_as][0]: relas[right_as][0].append(left_as)
    return relas


class monitor_cut():
    # 3、读取旧的.addDel文件 计算新routingTree下模拟的结果 （37服务器 routingTree.py）

    def __init__(self, file_path, old_del_path, dsn_path, asn):
        self.file_name = file_path
        self.graph = {}
        self.asn = asn
        self.dsn_path = dsn_path
        self.old_del_path = old_del_path
        self.tempgraphname = file_path + '.graph.json'

        # 存储结果：{[]:节点总数量，[queue]:节点数量}
        self.res = {}

        # 创建图
        self.from_npz_create_graph()
        with open(self.tempgraphname, 'w') as f:
            json.dump(self.graph, f)

        with open(self.dsn_path, 'w') as f:
            f.write('#|' + str(self.res['']) + '\n')

        self.monitor_node_addDel()

    def from_npz_create_graph(self):

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
            # self.graph[a][1] 是指向到a的节点
            # self.graph[a][0] 是a指向的节点
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        self.res[''] = len(self.graph)
        for i in self.graph[self.asn][1]:
            self.graph[i][0].remove(self.asn)
        self.graph[self.asn][1].clear()

    def monitor_node_addDel(self):
        with open(self.dsn_path, "a+") as dsn_f:
            with open(self.old_del_path, 'r') as fp:
                line = fp.readline()
                while line:
                    if line[0].isdigit():
                        queue = line.split('|')[0].split(' ')
                        oldbreak = line.split('|')[1].split(' ')
                        linkres = self.monitor_cut_node(queue)

                        dsn_f.write(line.split('|')[0] + '|' + ' '.join(list(map(str, linkres))) + '\n')
                        with open(self.tempgraphname, 'r') as graph_f:
                            self.graph = json.load(graph_f)
                    line = fp.readline()

    def monitor_cut_node(self, queue):
        res = []
        for node in queue:
            for i in self.graph[node][1]:  # 找node连了哪些节点
                self.graph[i][0].remove(node)  # 在node的后向点的前向点中把node删了

            self.graph[node][1] = []  # node设置成不链接任何节点
        while queue:
            n = queue.pop(0)  # 取被影响的节点
            res.append(n)  # 加入返回结果列表
            if n not in self.graph:
                continue

            for i in self.graph[n][0]:
                self.graph[i][1].remove(n)  # 在被影响节点n的前向点中的后向点中把n删了
                if len(self.graph[i][1]) == 0:  # 如果n的前向点没有指向其他的节点，那么这个点也列为被影响的节点
                    queue.append(i)
            del self.graph[n]  # 在图中删掉被影响的节点
        return res


class iSecutiry():

    def __init__(self, path, connect_dsn_path, sort_dsn_path) -> None:

        self.path = path
        self.connect_dsn_path = connect_dsn_path
        self.sort_dsn_path = sort_dsn_path

    def extract_connect_list(self, begin_num=1):
        '''
        对优化后的路由树重新生成count_num
        '''

        def basic_user_domain(line):
            as_list = line.split(' ')
            res1 = 0
            res2 = 0
            for _as in as_list:
                if _as in as_importance:
                    res1 += as_importance[_as][0]
                    res2 += as_importance[_as][1]
            return [line.count(' ') + 1, res1, res2]

        if not os.path.exists(self.path):
            return
        if begin_num < 1: begin_num = 1
        cc_name = os.listdir(self.path)
        for cc in cc_name:
            if cc + '.json' not in os.listdir(as_importance_path):
                continue
            with open(os.path.join(as_importance_path, cc + '.json'), 'r') as f:
                _as_importance = json.load(f)
            as_importance = {}
            for line in _as_importance:
                as_importance[line[0]] = line[1:]

            for num in Num_list:
                file_path = os.path.join(self.path, cc, 'all', str(num))
                if not os.path.exists(file_path):
                    continue
                try:
                    if not os.path.exists(os.path.join(self.connect_dsn_path, cc)):
                        os.makedirs(os.path.join(self.connect_dsn_path, cc))
                    if not os.path.exists(os.path.join(self.connect_dsn_path, cc, str(num))):
                        os.makedirs(os.path.join(self.connect_dsn_path, cc, str(num)))
                except:
                    time.sleep(5)
                file_name = os.listdir(file_path)
                file_name = [i for i in os.listdir(file_path) if i[-4:] == '.txt' and i.find('as-rel') == -1 and i[0] != '.']
                if not len(file_name):
                    break

                for file in file_name:
                    res = {}
                    asname = file.split('.')[0]
                    if os.path.exists(os.path.join(self.connect_dsn_path, cc, str(num), asname + '.json')):
                        continue
                    res[asname] = {}
                    res[asname]['asNum'] = -1
                    res[asname]['connect'] = [[], [], [], []]
                    with open(os.path.join(file_path, file), 'r') as f:
                        for line in f:
                            l = line.split('|')
                            if line[0] == '#':
                                res[asname]['asNum'] = int(l[1])
                            elif len(l) > 1 and line[0][0] != '(' and l[0] != '':
                                l1 = l[0].count(' ')
                                if l1 < len(res[asname]['connect']):
                                    l2 = basic_user_domain(l[1])
                                    res[asname]['connect'][l1].append(l2)
                    with open(os.path.join(self.connect_dsn_path, cc, str(num), asname + '.json'), 'w') as df:
                        json.dump(res, df)


def cal_anova_change_for_single_country(connect_dsn_path, old_connect_path, num, _cc, m, output_path):
    '''
    4、计算某个国家优化后的排名
    '''

    if not os.path.exists(os.path.join(connect_dsn_path, _cc, str(num))):
        return
    if len(os.listdir(os.path.join(connect_dsn_path, _cc, str(num)))) == 0:
        return

    for value in ['basic']:
        new_anova_path = os.path.join(output_path, m, SORT_DSN_PATH_SUFFIX, value + '_' + _cc)
        mkdir(new_anova_path)
        groud_truth_based_anova_for_single_country(os.path.join(connect_dsn_path, _cc, str(num)), _cc, old_connect_path,
                                                   new_anova_path, value, num)


@record_launch_time
def record_result(topo_list, output_path, type_path, _type):
    '''
    topo_list topo类型列表
    output_path output路径
    type_path: anova var
    _type : med var
    5、记录排名的变化
    '''
    global data_dim

    change_res = {}
    for _cc in cc_list:
        if _cc in ['BR', 'US', 'RU']:
            continue
        change_res[_cc] = {}
        for num in Num_list:
            change_res[_cc][str(num)] = {}
            country_internal_rank(topo_list, output_path, RESULT_SUFFIX, type_path, _type, _cc, str(num), data_dim)


def customerToProviderBFS(destinationNode, routingTree, graph):
    '''
        input:
            destinationNode (the root of routing tree)
            empty routing tree which is sparse also
        output:
            routing tree after step 1 of routing tree algorithm
            nodes added this step as a dictionary where key = level and value = list of nodes
        what it does:
            perform a bfs from destinationNode and only add relationship = 3
        '''
    BFS = [(0, [destinationNode])]
    levels = {}  # Dictionary returning the highest level of each key
    allNodes = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
    for node in allNodes:
        levels[node] = -1

    levels[destinationNode] = 0

    for pair in BFS:
        level = pair[0]
        vertices = pair[1]
        for vertex in vertices:
            for node, relationship in zip(graph[vertex].nonzero()[1], graph[vertex].data):
                if (relationship == 3) and (routingTree[node, vertex] == 0 and routingTree[vertex, node] == 0) and (
                        (not levels[node] <= level) or (levels[node] == -1)):
                    routingTree[node, vertex] = 3
                    if BFS[-1][0] == level:
                        BFS.append((level + 1, [node]))
                        levels[node] = level + 1
                    else:
                        BFS[-1][1].append(node)
                        levels[node] = BFS[-1][0]
                elif (relationship == 3) and (routingTree[node, vertex] == 0 and routingTree[vertex, node] == 0):
                    routingTree[node, vertex] = 3
    return routingTree, BFS, levels


def peerToPeer(routingTree, BFS, graph, levels):
    '''
    input:
        routing tree which is sparse also
        nodes from step 1 of RT algorithm in bfs order
    output:
        routing tree after step 2 of routing tree algorithm
        nodes added from this step and previous step as a dictionary where key = level and value = list of nodes
    purpose:
        connect new nodes to nodes added in step 1 with relationship = 1
    '''
    old = {}
    allNodes = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
    for node in allNodes:
        old[node] = 0

    newBFS = copy.deepcopy(BFS)
    newLevels = levels
    for pair in BFS:
        level = pair[0]
        vertices = pair[1]
        for vertex in vertices:
            for node, relationship in zip(graph[vertex].nonzero()[1], graph[vertex].data):
                if (relationship == 1) and (old[node] == 0):
                    routingTree[node, vertex] = 1

                    if newBFS[-1][0] == level:
                        newBFS.append((level + 1, [node]))
                        newLevels[node] = level + 1
                    else:
                        newBFS[-1][1].append(node)
                        newLevels[node] = newBFS[-1][0]
    return routingTree, newBFS, newLevels


def providerToCustomer(routingTree, BFS, graph, levels):
    '''
    input:
        routing tree which is sparse also
        nodes from step 1 and 2 of RT algorithm
    output:
        routing tree after step 3 of routing tree algorithm
        nodes added from this step and previous two steps as a dictionary where key = level and value = list of nodes
    purpose:
        breadth first search of tree, add nodes with relationship 2
    '''
    oldNodes = []
    old = {}
    allNodes = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
    for node in allNodes:
        old[node] = 0
    for pair in BFS:
        oldNodes.extend(pair[1])

    for node in oldNodes:
        old[node] = 1

    for pair in BFS:
        level = pair[0]
        vertices = pair[1]
        for vertex in vertices:
            for node, relationship in zip(graph[vertex].nonzero()[1], graph[vertex].data):
                if (relationship == 2) and (routingTree[vertex, node] == 0 and routingTree[node, vertex] == 0) and \
                        old[node] == 0 and ((not (levels[node] <= level)) or (levels[node] == -1)):
                    routingTree[node, vertex] = 2
                    if BFS[-1][0] == level:
                        BFS.append((level + 1, [node]))
                        levels[node] = level + 1
                    else:
                        BFS[-1][1].append(node)
                        levels[node] = BFS[-1][0]
                elif (relationship == 2) and (routingTree[vertex, node] == 0 and routingTree[node, vertex] == 0):
                    routingTree[node, vertex] = 2
    return routingTree


def saveAsNPZ(fileName, matrix, destinationNode):
    # 生成路由树的时候排除到root的连接
    for i in matrix[destinationNode].nonzero()[1]:
        matrix[destinationNode, i] = 0

    matrixCOO = matrix.tocoo()
    row = matrixCOO.row
    col = matrixCOO.col
    data = matrixCOO.data
    shape = matrixCOO.shape
    np.savez(fileName, row=row, col=col, data=data, shape=shape)


def makeRoutingTree(destinationNode, fullGraph, routingTree, new_npz_path):
    '''
    input:
        destination AS
    output:
        routing tree of destination AS in sparse matrix format
    '''

    print("=================" + str(destinationNode) + "=======================")

    stepOneRT, stepOneNodes, lvls = customerToProviderBFS(destinationNode, routingTree, fullGraph)
    stepTwoRT, stepTwoNodes, lvlsTwo = peerToPeer(stepOneRT, stepOneNodes, fullGraph, lvls)
    stepThreeRT = providerToCustomer(stepTwoRT, stepTwoNodes, fullGraph, lvlsTwo)
    # stepThreeRT
    saveAsNPZ(os.path.join(new_npz_path, "dcomplete" + str(destinationNode)), stepThreeRT, destinationNode)


def dataConverter(relas_list):
    returned_list = []

    for data in relas_list:
        output_data = []
        try:
            if (int(data[0]) not in output_data):
                output_data.append(data[0])
            if (int(data[1]) not in output_data):
                output_data.append(data[1])

        except Exception as e:
            raise e
            exit()
        if (data[2] == 0 or data[2] == "0"):
            output_data.append('p2p')
        else:
            output_data.append('p2c')

        returned_list.append(output_data)

    return returned_list


def graphGenerator(data_list):
    def determineNodeCount(_data_list):
        nodeList = []
        for splitLine in _data_list:
            if (int(splitLine[0]) not in nodeList):
                nodeList.append(int(splitLine[0]))
            if (int(splitLine[1]) not in nodeList):
                nodeList.append(int(splitLine[1]))
        return max(nodeList)

    def fileToSparse(d_list):
        '''
        reads the full AS graph in as a text file of relationships,
        converts it to a sparse matrix (note that row x or column x is for AS x)
        saves the sparse matrix
        loads the sparse matrix and times the loading
        usage: fileToSparse("Cyclops_caida_cons.txt")
        '''

        numNodes = determineNodeCount(d_list)

        empMatrix = sparse.lil_matrix((numNodes + 1, numNodes + 1), dtype=np.int8)
        i = 1
        total = len(d_list)
        for splitLine in d_list:
            if i % 1000 == 0:
                print("completed: " + str((float(i) / float(total)) * 100.0))
            i += 1
            node1 = int(splitLine[0])
            node2 = int(splitLine[1])
            relationship = splitLine[2][:3]
            if relationship == "p2p":
                empMatrix[node1, node2] = 1
                empMatrix[node2, node1] = 1
            if relationship == "p2c":
                empMatrix[node1, node2] = 2
                empMatrix[node2, node1] = 3
        empMatrix = empMatrix.tocsr()
        return numNodes, empMatrix

    return fileToSparse(data_list)


def create_rela_file(relas):
    relas_list = []
    for c in relas:
        for b in relas[c][1]:
            relas_list.append([str(c), str(b), -1])
        for b in relas[c][2]:
            if c <= b:
                relas_list.append([str(c), str(b), 0])
    return relas_list


@record_launch_time_and_param(2, 1)
def add_npz_and_monitor_cut_pool(output_path, m, cname):
    '''
    output_path output路径
    m topo类型
    cname: coutry code
    重新生成 npz文件和破坏结果
    
    '''
    dst_path = os.path.join(output_path, m)
    new_path = os.path.join(dst_path, SUFFIX, 'new_optimize')
    floyed_path = os.path.join(dst_path, SUFFIX, 'floyed')
    rtree_path = os.path.join(dst_path, 'rtree/')
    add_link_path = os.path.join(floyed_path, 'add_link', cname)
    mkdir(new_path)
    mkdir(os.path.join(new_path, cname))
    mkdir(os.path.join(new_path, cname, 'rtree'))
    mkdir(os.path.join(new_path, cname, 'all'))
    mkdir(add_link_path)

    relas_file = os.path.join(rtree_path, cname, 'as-rel.txt')
    add_link_file = os.path.join(floyed_path, cname + '.opt_add_link_rich.json')

    def add_npz_and_monitor_cut_thread(file, add_link_num):
        '''
        file 原来的路由树文件
        add_link_num 要加入的优化连接数量

        重新创建路由树并模拟破坏
        '''
        old_npz_file = os.path.join(rtree_path, cname, file.split('.')[0] + '.npz')
        new_npz_path = os.path.join(new_path, cname, 'rtree', str(add_link_num) + '/')
        temp_path = os.path.join(new_npz_path, 'temp')
        new_npz_file = os.path.join(new_npz_path, file.split('.')[0] + '.npz')
        add_link_relas_file = os.path.join(new_path, cname, 'rtree',
                                           '%s.%s.add_link_relas.json' % (file.split('.')[0], add_link_num))
        old_del_path = os.path.join(rtree_path, cname, file)
        new_del_path = os.path.join(new_path, cname, 'all', str(add_link_num) + '/')
        new_del_file = os.path.join(new_del_path, file)
        cc_as_list_path = os.path.join(output_path, 'cc2as', '%s.json' % cname)

        mkdir(new_npz_path)
        mkdir(new_del_path)
        mkdir(temp_path)

        if os.path.exists(old_npz_file):
            asn = file.split('.')[0][9:]
            rela = generate_new_rela(add_link_file, relas_file, add_link_num, cc_as_list_path, add_link_path,
                                     asn)  # 把优化的节点加入到rtree连接文件中
            relas_list = create_rela_file(rela)
            data_list = dataConverter(relas_list)
            maxNum, fullGraph = graphGenerator(data_list)
            routingTree = sparse.dok_matrix((maxNum + 1, maxNum + 1), dtype=np.int8)

            makeRoutingTree(int(asn), fullGraph, routingTree, new_npz_path)
            monitor_cut(new_npz_file, old_del_path, new_del_file, asn)

    thread_pool_inner = ThreadPool(multiprocessing.cpu_count() * 10)
    for _file in os.listdir(os.path.join(rtree_path, cname)):
        if _file.find('addDel') == -1:
            continue
        for _add_link_num in Num_list:
            thread_pool_inner.apply(add_npz_and_monitor_cut_thread, (
                _file,
                _add_link_num,
            ))
    thread_pool_inner.close()
    thread_pool_inner.join()


@record_launch_time_and_param(1, 0, 2)
def cal_anova_for_single_cc_pool(m, _cc, num, output_path):
    connect_dsn_path = os.path.join(output_path, m, SUFFIX, 'new_optimize_result', 'count_num')
    old_connect_path = os.path.join(output_path, m, 'result/count_num/')
    cal_anova_change_for_single_country(connect_dsn_path, old_connect_path, num, _cc, m, output_path)


@record_launch_time_and_param(1, 0, 2)
def cal_var_for_single_cc_pool(m, _cc, num, output_path, data_dim):
    '''
        output_path output路径
        m topo类型
        _cc : country_code
        num 优化节点数量
        data_dim : basic|user|domain 维度列表
    '''
    old_count_num_path = os.path.join(output_path, m, 'result/count_num/')
    new_count_num_path = os.path.join(output_path, m, SUFFIX, 'new_optimize_result', 'count_num', _cc, str(num))
    for value in data_dim:
        new_var_path = os.path.join(output_path, m, RESULT_SUFFIX, 'var', value + '_' + _cc)
        mkdir(new_var_path)
        cal_var_change_for_single_country(new_count_num_path, old_count_num_path, value, _cc, new_var_path, num)


def judge_var(target_list, result):
    '''
    target_list 待排序的as列表
    result 用来存储最终结果

    用递归的方式，按照方差从小到大为数据排序
    '''
    if len(target_list) == 0:
        return
    source_list = target_list[0]
    target_list.remove(source_list)
    result.append([])
    result[-1].append(source_list['key'])
    if len(target_list) == 0:
        return
    for ii in target_list:
        stat, p = stats.levene(source_list['list'], ii['list'])
        if ii['key'] == source_list['key']:
            continue
        if p > 0.05:
            result[-1].append(ii['key'])
            target_list.remove(ii)
    judge_var(target_list, result)


def cal_var_change_for_single_country(new_count_num_path, old_count_num_path, _type, single_country_name, new_var_path, num):
    '''
    new_count_num_path 新生成的count_num路径
    old_count_num_path 旧的count_num路径
    _type : basic|user|domain 维度类型
    single_country_name : country code
    new_var_path 存储新数据方差排名的路径
    num 破坏节点数量

    把优化后的结果同其他没有优化的结果放在一起排序
    '''

    var_result = {}
    result = []
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)

    def groud_truth_based_var_old_thread(cc):

        def basic_value_map(x):
            if _type == 'basic':
                return x[value_dict[_type]] / N
            else:
                return x[value_dict[_type]]

        if cc == single_country_name: return
        if cc.find('.json') != -1 or cc[0] == '.': return
        if cc[-4:] == 'json':
            return
        as_list = os.listdir(os.path.join(old_count_num_path, cc))
        for as_file_name in as_list:
            if as_file_name.find('.json') == -1 or as_file_name[0] == '.':
                continue
            with open(os.path.join(old_count_num_path, cc, as_file_name), 'r') as as_file:
                as_data = json.load(as_file)
                for _as in as_data:
                    N = as_data[_as]['asNum']
                    if N < 0:
                        continue
                    if N < 20:
                        continue
                    for i in as_data[_as]['connect']:
                        if len(i) == 0:
                            continue
                        var_result['%s-%s' % (cc, as_file_name[:-5])] = {
                            'list': list(map(basic_value_map, i)),
                            'key': '%s-%s' % (cc, as_file_name[:-5])
                        }

    def groud_truth_based_var_new_thread(file):

        def basic_value_map(x):
            if _type == 'basic':
                return x[value_dict[_type]] / N
            else:
                return x[value_dict[_type]]

        with open(os.path.join(new_count_num_path, file), 'r') as as_file:
            as_data = json.load(as_file)
            for _as in as_data:
                N = as_data[_as]['asNum']
                if N < 0:
                    continue
                if N < 20:
                    continue
                for i in as_data[_as]['connect']:
                    if len(i) == 0:
                        continue
                    var_result['%s-%s' % (single_country_name, file[:-5])] = {
                        'list': list(map(basic_value_map, i)),
                        'key': '%s-%s' % (single_country_name, file[:-5])
                    }

    cc_list = os.listdir(old_count_num_path)
    file_list = os.listdir(new_count_num_path)
    for cc in cc_list:
        thread_pool.apply_async(groud_truth_based_var_old_thread, (cc,))
    for file in file_list:
        thread_pool.apply_async(groud_truth_based_var_new_thread, (file,))
    thread_pool.close()
    thread_pool.join()
    var_list = list(var_result.values())
    var_list.sort(key=lambda x: np.var(x['list']))

    judge_var(var_list, result)

    with open(os.path.join(new_var_path, 'sorted_country_%s.%s.json' % (_type, str(num))), 'w') as sorted_var_f:
        json.dump(result, sorted_var_f)


@record_launch_time
def part1(topo_list, output_path):
    '''
    topo_list topo列表
    output_path output路径

    添加优化路径后重新生成npz文件,并生成破坏结果
    '''
    pool = Pool(multiprocessing.cpu_count())
    for m in topo_list:

        for cname in cc_list:
            path = os.path.join(output_path, m, 'rtree/')
            floyed_path = os.path.join(output_path, m, SUFFIX, 'floyed/')
            if not os.path.exists(os.path.join(path, cname, 'as-rel.txt')):
                print(cname + ' 没有as-rel')
                continue
            if not os.path.exists(os.path.join(floyed_path, cname + '.opt_add_link_rich.json')):
                print(cname + ' 没有opt_add_link_rich')
                continue

            add_npz_and_monitor_cut_pool(
                output_path,
                m,
                cname,
            )
            # pool.apply_async(add_npz_and_monitor_cut_pool, (
            #     output_path,
            #     m,
            #     cname,
            # ))
    pool.close()
    pool.join()


@record_launch_time
def part2(output_path, topo_list):
    '''
    output_path output路径
    topo_list topo列表

    重新生成 count_num文件夹
    '''
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    print('Start part2')

    def make_dir_thread(m):
        new_path = os.path.join(output_path, m, SUFFIX, 'new_optimize')
        result_dsn_path = os.path.join(output_path, m, SUFFIX, 'new_optimize_result')
        connect_dsn_path = os.path.join(result_dsn_path, 'count_num')
        sort_dsn_path = os.path.join(result_dsn_path, 'anova')

        mkdir(result_dsn_path)
        mkdir(connect_dsn_path)
        mkdir(sort_dsn_path)

        iS = iSecutiry(new_path, connect_dsn_path, sort_dsn_path)
        iS.extract_connect_list()

    for m in topo_list:
        make_dir_thread(m)
        # thread_pool.apply_async(make_dir_thread, (m, ))
    thread_pool.close()
    thread_pool.join()


@record_launch_time
def part3(topo_list, output_path):
    '''
    output_path output路径
    topo_list topo列表

    对优化的结果进行anova和方差排序
    '''
    global data_dim
    pool = Pool(multiprocessing.cpu_count())
    for _cc in cc_list:

        pool.apply_async(new_cal_anova_for_single_cc_pool, (topo_list, _cc, output_path, data_dim))

        for m in topo_list:
            connect_dsn_path = os.path.join(output_path, m, SUFFIX, 'new_optimize_result', 'count_num')
            for num in Num_list:
                if not os.path.exists(os.path.join(connect_dsn_path, _cc, str(num))):
                    continue
                if len(os.listdir(os.path.join(connect_dsn_path, _cc, str(num)))) == 0:
                    continue

                pool.apply_async(cal_var_for_single_cc_pool, (m, _cc, str(num), output_path, data_dim))
    pool.close()
    pool.join()


def new_cal_anova_for_single_cc_pool(topo_list, _cc, output_path, data_dim):
    '''
    output_path output路径
    topo_list topo列表

    _cc: country code
    data_dim : basic|user|domain维度类型列表

    进行anova排序
    

    把优化后的结果同其他没有优化的结果放在一起排序
    

    '''
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}

    for m in topo_list:
        super_connect_dsn_path = os.path.join(output_path, m, SUFFIX, 'new_optimize_result', 'count_num')
        for _num in Num_list:

            num = str(_num)
            print(num)
            if not os.path.exists(os.path.join(super_connect_dsn_path, _cc, num)):
                continue
            if len(os.listdir(os.path.join(super_connect_dsn_path, _cc, num))) == 0:
                continue
            num = str(_num)
            connect_dsn_path = os.path.join(output_path, m, SUFFIX, 'new_optimize_result', 'count_num', _cc, num)
            old_connect_path = os.path.join(output_path, m, 'result/count_num/')
            old_anova_path = os.path.join(output_path, m, 'result/anova/')
            numIndex = Num_list.index(int(num))
            for value in data_dim:

                new_anova_path = os.path.join(output_path, m, SORT_DSN_PATH_SUFFIX, value + '_' + _cc)

                mkdir(new_anova_path)
                if numIndex == 0:
                    old_anova_file_path = os.path.join(old_anova_path, f'sorted_country_{value}.json')
                else:
                    old_anova_file_path = os.path.join(output_path, m, SORT_DSN_PATH_SUFFIX, f'{value}_{_cc}',
                                                       f'sorted_country_{value}.{Num_list[numIndex - 1]}.json')
                print('%s compare with %s' % (str(num), old_anova_file_path))
                # 读取之前的anova数据
                with open(old_anova_file_path, 'r') as f:
                    old_anova_data = json.load(f)
                new_anova_data = []
                new_as_list = list(map(lambda x: f'{_cc}-{x[:-5]}', os.listdir(os.path.join(connect_dsn_path))))
                old_rank_map = {}
                for old_index, old_rank in enumerate(old_anova_data):

                    new_anova_data.append([])  # 复制除了当前国家的所有排名
                    for old_as in old_rank:
                        if old_as in new_as_list:
                            old_rank_map[old_as] = old_index

                            continue

                        new_anova_data[-1].append(old_as)

                for new_as in new_as_list:  # 循环优化后的节点
                    if new_as not in old_rank_map:
                        continue
                    if old_rank_map[new_as] == 0:
                        new_anova_data[old_rank_map[new_as]].append(new_as)
                        continue
                    new_rank = old_rank_map[new_as]
                    for i in range(new_rank - 1, -1, -1):  # 从优化后的节点排名往0循环
                        if len(old_anova_data[i]) == 0:
                            new_rank -= 1
                            continue
                        compare_as = old_anova_data[i][-1]
                        compared_cc, conmpared_prefix = compare_as.split('-')
                        if numIndex == 0:
                            compare_file_path = os.path.join(old_connect_path, compared_cc, f"{conmpared_prefix}.json")
                        else:
                            compare_file_path = os.path.join(output_path, m, RESULT_SUFFIX, 'count_num', _cc, str(Num_list[numIndex - 1]),
                                                             f"{new_as.split('-')[1]}.json")
                        with open(compare_file_path, 'r') as compared_f:
                            compared_data = json.load(compared_f)
                            old_l = []
                            l = {}
                        for _as in compared_data:
                            N = compared_data[_as]['asNum']
                            for i in compared_data[_as]['connect']:
                                if value == 'basic':
                                    old_l += [_i[value_dict[value]] / N for _i in i]
                                else:
                                    old_l += [_i[value_dict[value]] for _i in i]
                        l[compare_as] = old_l

                        opted_as_count_num_file = os.path.join(connect_dsn_path, f"{new_as.split('-')[1]}.json")
                        with open(opted_as_count_num_file, 'r') as opted_f:
                            new_l = []
                            opted_data = json.load(opted_f)
                        for _as in opted_data:
                            N = opted_data[_as]['asNum']
                            for i in opted_data[_as]['connect']:
                                if value == 'basic':
                                    new_l += [_i[value_dict[value]] / N for _i in i]
                                else:
                                    new_l += [_i[value_dict[value]] for _i in i]
                        l[new_as] = new_l

                        nums, groups = [], []
                        for k, v in l.items():
                            nums += v
                            groups += len(v) * [k]
                        mc = MultiComparison(nums, groups)  # 比较新旧数据
                        result = mc.tukeyhsd()

                        line = result._results_table.data[1]
                        print(f'newAs -> {new_as} compare_as -> {compare_as} line => {line} ')
                        if new_as == line[0]:
                            if not line[-1] or line[2] < 0:
                                # 0 new
                                # 1 old
                                break

                            new_rank -= 1

                        elif new_as == line[1]:
                            if not line[-1] or line[2] > 0:
                                # 0 new
                                # 1 old
                                break

                            new_rank -= 1

                    new_anova_data[new_rank].append(new_as)

                new_anova_data = list(filter(lambda x: len(x) > 0, new_anova_data))
                print('save %s' % os.path.join(new_anova_path, f'sorted_country_{value}.{num}.json'))
                with open(os.path.join(new_anova_path, f'sorted_country_{value}.{num}.json'), 'w') as result_f:
                    json.dump(new_anova_data, result_f)


cc_list = []
as_importance_path = None


@record_launch_time
def train_routing_tree(topo_list, _cc_list, output_path, _as_importance_path, optimize_link_num_list, _data_dim):
    '''
    topo_list topo类型
    _cc_list 国家列表
    output_path output路径
    _as_importance_path 权重路径
    optimize_link_num_list 优化连接数量列表
    _data_dim basic|user|domain 维度类型列表

    根据优化结果重新创建路由树，破坏，排序，生成排名
    '''
    global as_importance_path
    global cc_list
    global data_dim
    global Num
    global Num_list
    Num_list = optimize_link_num_list
    cc_list = _cc_list
    as_importance_path = _as_importance_path
    data_dim = _data_dim
    part1(topo_list, output_path)
    part2(output_path, topo_list)
    part3(topo_list, output_path)
    record_result(topo_list, output_path, 'anova', 'med')
    record_result(topo_list, output_path, 'var', 'var')
