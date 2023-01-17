#!usr/bin/env python
# _*_ coding:utf8 _*_
'''

生成国家内部routingTree

'''
from glob import glob
import os
import json
import copy
from other_script.my_types import *
import numpy as np
from scipy import sparse
import scipy.io
import multiprocessing
from multiprocessing.pool import ThreadPool
from importlib import import_module
from other_script.util import mkdir, record_launch_time, record_launch_time_and_param

# as-cone关系数据
gl_asn_data: Dict[AS_CODE, int]
# 创建rtree时过滤器
gl_filter_rtree: Callable
# 输出output路径
gl_dst_dir_path: OUTPUT_PATH
# 输入json文件路径
gl_as_rela_file_path: str

'''
把inFile里面的所有AS号写入到outFile中

infile数据格式 as1|as2|0
outFile里面的格式 as1\tas2\tp2p

生成.nodeList文件
格式as1\nas2\as3...
'''


# 将AS间的关系由数字表示转为文本表示
def dataConverter(in_relaFile_path, out_temp_text_file):
    '''
    in_relaFile_path :as-rel.txt存储路径
    out_temp_text_file 临时输出文件-1路径
    将AS间的关系由数字表示转为文本表示,存储到out_temp_text_file
    '''
    of = open(out_temp_text_file, 'w')

    with open(in_relaFile_path) as fp:
        line = fp.readline()
        cnt = 1
        nodeList = []

        while line:
            # 现在逻辑不会有‘#’开头
            if (line[0] != '#'):
                data: List[AS_CODE or str] = line.split('|')
                outString: str = str(data[0]) + "\t" + str(data[1])
                try:
                    if (int(data[0]) not in nodeList):
                        nodeList.append(int(data[0]))
                    if (int(data[1]) not in nodeList):
                        nodeList.append(int(data[1]))
                except Exception as e:
                    print(e)
                    raise e
                    exit()
                if (data[2] == "0\n" or data[2] == "0"):
                    outString += "\t" + "p2p\n"  ## add endline here for saving
                else:
                    outString += "\t" + "p2c\n"
                of.write(outString)
                line = fp.readline()
                cnt += 1
            else:
                line = fp.readline()

    of.close()

    ### Saving the node list ###
    outFileList = out_temp_text_file + ".nodeList"
    ouf = open(outFileList, 'w')
    for node in nodeList:
        ouf.write(str(node) + "\n")
    ouf.close()


def graphGenerator(in_temp_text_file: str, out_mtx_file: str):
    '''
    in_temp_text_file 临时输出文件-1路径
    out_mtx_file : 二维矩阵路径 大小为as最大值
    
    将临时输出文件-1转为矩阵存到out_mtx_file
    同时存下矩阵所有节点,路径为out_mtx_file.nodeList
    '''

    def determineNodeCount(in_temp_text_file: str, out_mtx_file: str) -> int:
        # 所有AS列表
        nodeList: List[AS_CODE] = []
        with open(in_temp_text_file, 'r') as f:
            content = f.readlines()
        for line in content:
            if (line[0] != '#'):
                splitLine: List[Union[AS_CODE, str]] = line.split("\t", 2)
                if (int(splitLine[0]) not in nodeList):
                    nodeList.append(int(splitLine[0]))
                if (int(splitLine[1]) not in nodeList):
                    nodeList.append(int(splitLine[1]))

        ### Saving the node list ###
        # 这里存就是看看数据，后面没有用到
        node_list_path = out_mtx_file + ".nodeList"
        ouf = open(node_list_path, 'w')
        for node in nodeList:
            ouf.write(str(node) + "\n")
        ouf.close()
        return max(nodeList)

    def fileToSparse(in_temp_text_file: str, out_mtx_file: str):
        '''
        reads the full AS graph in as a text file of relationships,
        converts it to a sparse matrix (note that row x or column x is for AS x)
        saves the sparse matrix
        loads the sparse matrix and times the loading
        usage: fileToSparse("Cyclops_caida_cons.txt")
        '''

        numNodes: int = determineNodeCount(in_temp_text_file, out_mtx_file)

        with open(in_temp_text_file, 'r') as f:
            content = f.readlines()

        # 用最大的AS号创建 numNodes行 numNodes列矩阵
        empMatrix: MATRIX = sparse.lil_matrix((numNodes + 1, numNodes + 1), dtype=np.int8)
        i = 1
        total = len(content)

        for line in content:
            if i % 1000 == 0:
                # 记录进度
                print("completed: " + str((float(i) / float(total)) * 100.0))
            i += 1

            splitLine: List[AS_CODE] = line.split("\t", 2)
            node1 = int(splitLine[0])
            node2 = int(splitLine[1])
            relationship: str = splitLine[2][:3]
            if relationship == "p2p":
                empMatrix[node1, node2] = 1
                empMatrix[node2, node1] = 1
            if relationship == "p2c":
                empMatrix[node1, node2] = 2
                empMatrix[node2, node1] = 3

        # 转换格式存入文件
        scipy.io.mmwrite(out_mtx_file, empMatrix.tocsr())
        return numNodes

    return fileToSparse(in_temp_text_file, out_mtx_file)


# 准备工作及创建路由树
def speedyGET(mtx_path: str, dsn_file: RTREE_CC_PATH, mtx_nodeList_file: str, numNodes: int) -> None:
    '''
    mtx_path 矩阵路径
    dsn_file rtree/cc路径
    mtx_nodeList_file:矩阵节点列表文件路径
    numNodes:最大AS号
    '''

    def checkPreviousLevelsAlt(BFS, node, level):
        '''
        check if node is in BFS at given level or any previous level
        '''
        while level >= 0:
            if node in BFS[level][1]:
                return True
            level -= 1
        return False

    def customerToProviderBFS(destinationNode: AS_CODE, routingTree: MATRIX, graph: MATRIX) -> Tuple[
        MATRIX, List[Tuple[int, List[AS_CODE]]], Dict[AS_CODE, int]]:
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
        BFS: List[Tuple[int, List[AS_CODE]]] = [(0, [destinationNode])]
        levels: Dict[AS_CODE, int] = {}  # Dictionary returning the highest level of each key

        # get all Node
        allNodes: List[AS_CODE] = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
        for node in allNodes:
            levels[node] = -1

        levels[destinationNode] = 0

        for pair in BFS:
            level: int = pair[0]
            vertices: List[AS_CODE] = pair[1]
            for vertex in vertices:
                for node, relationship in zip(graph[vertex].nonzero()[1], graph[vertex].data):
                    # 找当前节点的provider，并且不在矩阵中，并且层数不大于当前节点
                    if (relationship == 3) and (routingTree[node, vertex] == 0 and routingTree[vertex, node] == 0) and (
                            (not levels[node] <= level) or (levels[node] == -1)):
                        # 设置矩阵节点
                        routingTree[node, vertex] = 3
                        # 如果当前是BFS的最后一层
                        if BFS[-1][0] == level:
                            # 把这个点加入到BFS的下一层
                            BFS.append((level + 1, [node]))
                            levels[node] = level + 1
                        else:
                            # 否则记录到同一层中
                            BFS[-1][1].append(node)
                            levels[node] = BFS[-1][0]
                    elif (relationship == 3) and (routingTree[node, vertex] == 0 and routingTree[vertex, node] == 0):
                        # 在其他层出现过的节点也记录到矩阵
                        routingTree[node, vertex] = 3
        return routingTree, BFS, levels

    def peerToPeer(routingTree: MATRIX, BFS: List[Tuple[int, List[AS_CODE]]], graph: MATRIX, levels: Dict[AS_CODE, int]) -> Tuple[
        MATRIX, List[Tuple[int, List[AS_CODE]]], Dict[AS_CODE, int]]:
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
        oldNodes: List[AS_CODE] = []
        old: Dict[AS_CODE, int] = {}
        allNodes: List[AS_CODE] = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
        for node in allNodes:
            old[node] = 0

        for pair in BFS:
            oldNodes.extend(pair[1])
            for node in pair[1]:
                old[node] = 1
        newBFS: List[Tuple[int, List[AS_CODE]]] = copy.deepcopy(BFS)
        newLevels: Dict[AS_CODE, int] = levels
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

    def providerToCustomer(routingTree: MATRIX, BFS: List[Tuple[int, List[AS_CODE]]], graph: MATRIX, levels: Dict[AS_CODE, int]) -> MATRIX:
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
        oldNodes: List[AS_CODE] = []
        old: Dict[AS_CODE, int] = {}
        allNodes: List[AS_CODE] = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
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

    def saveAsNPZ(fileName: str, matrix: MATRIX):
        matrixCOO = matrix.tocoo()
        row = matrixCOO.row
        col = matrixCOO.col
        data = matrixCOO.data
        shape = matrixCOO.shape
        np.savez(fileName, row=row, col=col, data=data, shape=shape)

    def makeRoutingTree(destinationNode: AS_CODE, fullGraph: MATRIX):
        '''
        input:
            destination AS
        output:
            routing tree of destination AS in sparse matrix format
        '''

        print("=================" + str(destinationNode) + "=======================")
        if "dcomplete" + str(destinationNode) + '.npz' in os.listdir(dsn_file):
            print('exist')
            return
        routingTree: MATRIX = sparse.dok_matrix((numNodes + 1, numNodes + 1), dtype=np.int8)

        # 找是 destinationNode 的 provider 的节点
        stepOneRT, stepOneNodes, lvls = customerToProviderBFS(destinationNode, routingTree, fullGraph)
        # 找是 destinationNode 的 peer 的节点
        stepTwoRT, stepTwoNodes, lvlsTwo = peerToPeer(stepOneRT, stepOneNodes, fullGraph, lvls)
        # 找是 destinationNode 的 customer 的节点
        stepThreeRT: MATRIX = providerToCustomer(stepTwoRT, stepTwoNodes, fullGraph, lvlsTwo)
        saveAsNPZ(os.path.join(dsn_file, "dcomplete" + str(destinationNode)), stepThreeRT)

    fullGraph: MATRIX = scipy.io.mmread(str(mtx_path)).tocsr()  # read the graph on all ranks

    nodeList: List[AS_CODE] = []
    print('nodeListFile', mtx_nodeList_file)
    with open(mtx_nodeList_file) as fp:
        line: AS_CODE = fp.readline()
        while line:
            if (line[-1] == '\n'):
                line = line[:-1]
            nodeList.append(int(line))
            line = fp.readline()
    print("Max ASNode ID: " + str(max(nodeList)))

    # 按照需求过滤节点
    nodeList = gl_filter_rtree(nodeList, gl_asn_data)
    print('len(nodeList)', len(gl_filter_rtree(nodeList, gl_asn_data)))
    thread_pool = ThreadPool(processes=multiprocessing.cpu_count() * 10)
    for destinationNode in nodeList:
        try:

            # 创建路由树
            thread_pool.apply_async(makeRoutingTree, (
                destinationNode,
                fullGraph,
            ))
            # makeRoutingTree (
            #     destinationNode,
            #     fullGraph,
            # )
        except Exception as e:
            print('Exception', e)
            raise e

    thread_pool.close()
    thread_pool.join()


'''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''


def create_rela_file(relas: Dict[AS_CODE, List[List[AS_CODE]]], relaFile_path: str):
    '''
    relas:as关系字典
    relaFile_path:as-rel.txt存储路径

    生产as-rel.txt,用来存储当前国家的topo数据

    '''
    with open(relaFile_path, 'w') as f:
        for c in relas:
            for b in relas[c][1]:
                # p2c关系存-1
                f.write(str(c) + '|' + str(b) + '|-1\n')
            for b in relas[c][2]:
                if c <= b:
                    # peer关系存0
                    f.write(str(c) + '|' + str(b) + '|0\n')


def start_create_routingTree(dsn_file: RTREE_CC_PATH, relaFile_path: str, cc: COUNTRY_CODE) -> None:
    '''
    dsn_file:rtree/cc路径
    relaFile_path:as-rel.txt存储路径
    cc:country code
    开始创建路由树
    '''
    # 将 cc 下所有AS全路径图 存为文本文件
    dataConverter(relaFile_path, os.path.join(gl_dst_dir_path, 'temp', '%s_bgp-sas.npz' % cc))

    # 将 cc 下所有AS全路径图的文本文件转为矩阵文件
    maxNum: int = graphGenerator(os.path.join(gl_dst_dir_path, 'temp', '%s_bgp-sas.npz' % cc),
                                 os.path.join(gl_dst_dir_path, 'temp', '%s_routingTree.mtx' % cc))

    speedyGET(
        os.path.join(gl_dst_dir_path, 'temp', '%s_routingTree.mtx' % cc), dsn_file,
        os.path.join(gl_dst_dir_path, 'temp', '%s_routingTree.mtx.nodeList' % cc),
        maxNum
    )


'''生成一个国家内的排名前10的AS的路由树'''


def rTree(relas: Dict[AS_CODE, List[List[AS_CODE]]], dsn_file: RTREE_CC_PATH, cc: COUNTRY_CODE) -> None:
    '''
    relas:as关系字典
    dsn_file :存储rtree路径
    cc:Country code
    创建路由树
    '''
    relaFile_path: str = os.path.join(dsn_file, 'as-rel.txt')
    # 创建 cc 下所有AS全路径图
    create_rela_file(relas, relaFile_path)
    # 开始准备创建路由树
    start_create_routingTree(dsn_file, relaFile_path, cc)


'''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''


def create_relas(file: CC_PATH) -> Dict[AS_CODE, List[List[AS_CODE]]]:
    '''
    file:cc2as单一json路径
    生成as关系字典
    {as:[provider、customer、peer]}
    return as关系字典
    '''
    # global relas
    relas: Dict[AS_CODE, List[List[AS_CODE]]] = {}
    with open(gl_as_rela_file_path, 'r') as f:
        as_rela: Dict[AS_CODE, List[AS_CODE]] = json.load(f)
    with open(file, 'r') as f:
        cclist: List[AS_CODE] = json.load(f)
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

    return relas


'''
file cc2as里面的数据
dsnpath 输出路径
country 国家代码
gl_as_rela_file_path 国家对应的topo关系文件

目的 生成对应国家排名前10的路由树文件
'''


@record_launch_time_and_param(2)
def create_route_tree(cc_path: CC_PATH, _dsn_path: RTREE_PATH, country: COUNTRY_CODE) -> None:
    '''
    cc_path:cc2as文件夹下单独json文件路径
    _dst_dir_path:output路径
    country:国家代码

    创建路由树,存在{_dst_dir_path}/rtree
    '''
    country_name: List[COUNTRY_CODE] = os.listdir(_dsn_path)
    dsn_path: RTREE_CC_PATH = os.path.join(_dsn_path, country)
    if country not in country_name:
        mkdir(dsn_path)
    print(country + ' begin')
    relas: Dict[AS_CODE, List[List[AS_CODE]]] = create_relas(cc_path)

    # 判断rela是否为空
    relas_is_empty = True
    for i in relas:
        for ii in relas[i]:
            if len(ii) != 0:
                relas_is_empty = False
                break
        if not relas_is_empty:
            break

    if relas_is_empty:
        print('%s end ,relas is empty' % country)
        return

    rTree(relas, dsn_path, country)
    print(country + ' end')


def set_glabal_variable(as_rela_file: str, asn_data: Dict[AS_CODE, int], model_path: str, _dst_dir_path: OUTPUT_PATH) -> None:
    global gl_asn_data
    global gl_filter_rtree
    global gl_as_rela_file_path
    global gl_dst_dir_path

    gl_asn_data = asn_data
    dynamic_module = import_module(model_path)
    gl_filter_rtree = dynamic_module.filter_rtree
    gl_dst_dir_path = _dst_dir_path
    gl_as_rela_file_path = as_rela_file


@record_launch_time
def create_rtree(as_rela_file: str, _dst_dir_path: OUTPUT_PATH, _type: TOPO_TPYE, cc_list: List[COUNTRY_CODE], asn_data: Dict[AS_CODE, int],
                 cc2as_path: CC2AS_PATH, model_path: str):
    '''
    as_rela_file:原始txt转json之后的文件路径
    _dst_dir_path:output路径
    _type:topo类型
    cc_list:国家列表
    asn_data:cone字典
    cc2as_path:as和国家关系文件夹路径
    model_path:自定义模块路径

    创建路由树,存在{_dst_dir_path}/rtree
    '''

    set_glabal_variable(as_rela_file, asn_data, model_path, _dst_dir_path)

    rtree_path: RTREE_PATH = os.path.join(_dst_dir_path, _type, 'rtree')
    mkdir(rtree_path)

    process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_list:
        try:
            cc_path: CC_PATH = os.path.join(cc2as_path, cc + '.json')

            # 主体入口
            # 异步路由树可能会出问题
            create_route_tree(cc_path, rtree_path, cc)
            # process_pool.apply(create_route_tree, (cc_path, rtree_path, cc))
        except Exception as e:
            print(e)
            raise e
    process_pool.close()
    process_pool.join()
