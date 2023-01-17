import copy
import json
import os
from typing import Dict, List
from do_Internal.data_analysis import as_rela_txt_dont_save
from other_script.my_types import *
from other_script.util import mkdir, record_launch_time_and_param
import numpy as np
from scipy import stats, sparse
import multiprocessing
from multiprocessing.pool import ThreadPool

SUFFIX = 'optimize_link'
RESULT_SUFFIX = SUFFIX + '/new_optimize_result'
SORT_DSN_PATH_SUFFIX = RESULT_SUFFIX + '/anova'


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
        # print "---level---: ",level
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


def create_rela_file(relas):
    relas_list = []
    for c in relas:
        for b in relas[c][1]:
            relas_list.append([str(c), str(b), -1])
        for b in relas[c][2]:
            if c <= b:
                relas_list.append([str(c), str(b), 0])
    return relas_list


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


def generate_new_rela(add_link_file: str, relas_file: str, add_link_num: int, cc_as_list_path: str, add_link_path: str,
                      asn: str, old_as_data: Set[AS_CODE], numberAsns: Dict[AS_CODE, int]) -> Dict[int, List[List[int]]]:
    state = {'c2p': 0, 'p2p': 1, 'p2c': 2, 0: 'c2p', 1: 'p2p', 2: 'p2c'}
    match_state = {'1': {'1': 'p2p', '2': 'c2p'}, '2': {'1': 'p2c'}}

    def filter_opt_link(as_list, state_list):

        left_as, right_as = as_list
        begin_state, end_state = state_list
        s = state[match_state[begin_state][end_state]]

        # 如果set里面已经有了两个链接，只是链接类型不一样，则跳过
        if '-'.join([str(left_as), str(right_as)]) in list(map(lambda x: '-'.join(x.split('-')[:-1]), add_link)):
            return False
        if '-'.join([str(right_as), str(left_as)]) in list(map(lambda x: '-'.join(x.split('-')[:-1]), add_link)):
            return False

        if right_as in numberAsns and left_as in numberAsns:

            # 如果作为provider的cone比customer的要小3倍，跳过

            # c2p
            if numberAsns[right_as] * 3 < numberAsns[left_as] and s == 0:
                return False
            # p2c
            if numberAsns[left_as] * 3 < numberAsns[right_as] and s == 2:
                return False

        else:
            return False

        return True

    def add_opt_link():
        old_set_len = len(add_link)
        for line in m:
            as_list, state_list = line[:-1]
            if not filter_opt_link(as_list, state_list):
                continue

            left_as, right_as = as_list

            begin_state, end_state = state_list
            link_state = state[match_state[begin_state][end_state]]

            # 统一p2p链接格式
            if link_state == 1:
                if int(left_as) > int(right_as):
                    right_as, left_as = as_list

            add_link.add('-'.join([str(left_as), str(right_as), str(state[match_state[begin_state][end_state]])]))
            # 如果set变化了
            if old_set_len != len(add_link):
                find_rtree_list(right_as)
                break

    def create_relas(as_rela: str) -> Dict[int, List[List[int]]]:
        relas = {}
        with open(cc_as_list_path, 'r') as f:
            cclist = json.load(f)
        for c in cclist:
            # [provider、customer、peer]
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

    # 递归寻找新加入的节点和其子节点

    def find_rtree_list(asn):
        as_list = []
        old_as_data.add(asn)
        if asn not in relas:
            return
        for i in relas[asn]:
            as_list += i

        for ii in set(as_list):
            if ii in old_as_data:
                continue
            find_rtree_list(ii)

    json_data = as_rela_txt_dont_save(relas_file)
    relas = create_relas(json_data)
    with open(add_link_file, 'r') as f:
        m = json.load(f)

    add_link = set()
    for i in range(add_link_num):
        add_opt_link()

    add_link_list = list(map(lambda x: x.split('-'), add_link))
    with open(os.path.join(add_link_path, f'add_link-{asn}.{add_link_num}.json'), 'w') as f:
        json.dump(add_link_list, f)
    while add_link_list:
        link = add_link_list.pop(0)
        left_as = link[0]
        right_as = link[1]
        print(f'{asn} {add_link_num} add Link {link} ')
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

    def __init__(self, file_path, old_del_path, dsn_path, asn, all_as_list):
        self.file_name = file_path
        self.graph = {}
        self.asn = asn
        self.dsn_path = dsn_path
        self.old_del_path = old_del_path
        self.tempgraphname = file_path + '.graph.json'
        self.all_as_list = all_as_list

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
        for node in queue:
            for i in self.graph[node][1]:  # 找node连了哪些节点
                self.graph[i][0].remove(node)  # 在node的后向点的前向点中把node删了

            self.graph[node][1] = []  # node设置成不链接任何节点
        while queue:
            n = queue.pop(0)  # 取被影响的节点
            if n not in self.graph:
                continue

            for i in self.graph[n][0]:
                self.graph[i][1].remove(n)  # 在被影响节点n的前向点中的后向点中把n删了
                if len(self.graph[i][1]) == 0:  # 如果n的前向点没有指向其他的节点，那么这个点也列为被影响的节点
                    queue.append(i)
            del self.graph[n]  # 在图中删掉被影响的节点
        return [ii for ii in self.all_as_list if ii not in self.graph.keys()]


@record_launch_time_and_param(2, 1)
def add_npz_and_monitor_cut(output_path, m, cname, num_list, cc2as_list_path, numberAsns):
    dst_path = os.path.join(output_path, m)
    new_path = os.path.join(dst_path, SUFFIX, 'new_optimize')
    floyed_path = os.path.join(dst_path, SUFFIX, 'floyed')
    rtree_path = os.path.join(dst_path, 'rtree/')
    add_link_path = os.path.join(floyed_path, 'add_link', cname)

    with open(cc2as_list_path, 'r') as f:
        all_as_list = json.load(f)

    mkdir(new_path)
    mkdir(os.path.join(new_path, cname))
    mkdir(os.path.join(new_path, cname, 'rtree'))
    mkdir(os.path.join(new_path, cname, 'all'))
    mkdir(add_link_path)

    relas_file = os.path.join(rtree_path, cname, 'as-rel.txt')
    add_link_file = os.path.join(floyed_path, cname + '.opt_add_link_rich.json')

    def add_npz_and_monitor_cut_thread(file, add_link_num):

        old_npz_file = os.path.join(rtree_path, cname, file.split('.')[0] + '.npz')
        new_npz_path = os.path.join(new_path, cname, 'rtree', str(add_link_num) + '/')
        temp_path = os.path.join(new_npz_path, 'temp')
        new_npz_file = os.path.join(new_npz_path, file.split('.')[0] + '.npz')
        old_del_path = os.path.join(rtree_path, cname, file)
        new_del_path = os.path.join(new_path, cname, 'all', str(add_link_num) + '/')
        new_del_file = os.path.join(new_del_path, file)
        cc_as_list_path = os.path.join(output_path, 'cc2as', '%s.json' % cname)

        mkdir(new_npz_path)
        mkdir(new_del_path)
        mkdir(temp_path)

        if os.path.exists(old_npz_file):
            asn = file.split('.')[0][9:]
            old_npz_data = np.load(old_npz_file)

            old_as_data = [asn]
            old_as_data += list(old_npz_data['row'])
            old_as_data += list(old_npz_data['col'])
            rela = generate_new_rela(add_link_file, relas_file, add_link_num, cc_as_list_path, add_link_path, asn,
                                     set(old_as_data), numberAsns)  # 把优化的节点加入到rtree连接文件中

            maxNum, fullGraph = graphGenerator(dataConverter(create_rela_file(rela)))
            routingTree = sparse.dok_matrix((maxNum + 1, maxNum + 1), dtype=np.int8)

            makeRoutingTree(int(asn), fullGraph, routingTree, new_npz_path)
            monitor_cut(new_npz_file, old_del_path, new_del_file, asn, all_as_list)

    thread_pool_inner = ThreadPool(multiprocessing.cpu_count() * 10)
    for _file in os.listdir(os.path.join(rtree_path, cname)):
        if _file.find('addDel') == -1:
            continue
        for _add_link_num in num_list:
            add_npz_and_monitor_cut_thread(
                _file,
                _add_link_num,
            )
            # thread_pool_inner.apply_async(add_npz_and_monitor_cut_thread, (
            #     _file,
            #     _add_link_num,
            # ))
    thread_pool_inner.close()
    thread_pool_inner.join()
