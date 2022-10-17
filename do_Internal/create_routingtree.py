#!usr/bin/env python
# _*_ coding:utf8 _*_
'''

生成国家内部routingTree

'''
from glob import glob
import os
import json
import copy
from types import ModuleType
from typing import Callable, Dict, List
import numpy as np
import time
from mpi4py import MPI
from scipy import sparse
import scipy.io
import itertools
import multiprocessing
from multiprocessing.pool import ThreadPool
from importlib import import_module
from util import mkdir, record_launch_time, record_launch_time_and_param

gl_asn_data:Dict[str, int]
gl_filter_rtree:Callable
gl_dst_dir_path:str
gl_as_rela_file_path:str
'''
把inFile里面的所有AS号写入到outFile中

infile数据格式 as1|as2|0
outFile里面的格式 as1\tas2\tp2p

生成.nodeList文件
格式as1\nas2\as3...
'''


def dataConverter(inFile, outFile):
    of = open(outFile, 'w')

    with open(inFile) as fp:
        line = fp.readline()
        cnt = 1
        nodeList = []
        #
        while line:
            if (line[0] != '#'):
                data = line.split('|')
                outString = str(data[0]) + "\t" + str(data[1])
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

        # print("number of unique nodes: " + str(len(nodeList)))

    of.close()

    ### Saving the node list ###
    outFileList = outFile + ".nodeList"
    ouf = open(outFileList, 'w')
    for node in nodeList:
        ouf.write(str(node) + "\n")
    ouf.close()


def graphGenerator(inFile, outFile):

    def determineNodeCount(fileName, outName):
        nodeList = []
        with open(fileName, 'r') as f:
            content = f.readlines()
            # print(fileName)
        for line in content:
            if (line[0] != '#'):
                splitLine = line.split("\t", 2)
                if (int(splitLine[0]) not in nodeList):
                    nodeList.append(int(splitLine[0]))
                if (int(splitLine[1]) not in nodeList):
                    nodeList.append(int(splitLine[1]))
        # print("Node Count: " + str(len(nodeList)))
        # print("Max Node ID: " + str(max(nodeList)))

        ### Saving the node list ###
        outFileList = outName + ".nodeList"
        ouf = open(outFileList, 'w')
        for node in nodeList:
            ouf.write(str(node) + "\n")
        ouf.close()
        return max(nodeList)

    def fileToSparse(fileName, outName):
        '''
        reads the full AS graph in as a text file of relationships,
        converts it to a sparse matrix (note that row x or column x is for AS x)
        saves the sparse matrix
        loads the sparse matrix and times the loading
        usage: fileToSparse("Cyclops_caida_cons.txt")
        '''

        numNodes = determineNodeCount(fileName, outName)

        with open(fileName, 'r') as f:
            content = f.readlines()
        empMatrix = sparse.lil_matrix((numNodes + 1, numNodes + 1), dtype=np.int8)
        i = 1
        total = len(content)
        for line in content:
            if i % 1000 == 0:
                print("completed: " + str((float(i) / float(total)) * 100.0))
            i += 1
            splitLine = line.split("\t", 2)
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
        scipy.io.mmwrite(outName, empMatrix)
        start = time.time()
        test = scipy.io.mmread(outName).tolil()  # 5.4MB to save sparse matrix
        end = time.time()
        # print(end - start, " seconds to load")  # 2.3 seconds
        return numNodes

    return fileToSparse(inFile, outFile)


def speedyGET(args):
    # thread_pool = ThreadPool(processes=multiprocessing.cpu_count() * 5)
    def checkPreviousLevelsAlt(BFS, node, level):
        '''
        check if node is in BFS at given level or any previous level
        '''
        while level >= 0:
            if node in BFS[level][1]:
                return True
            level -= 1
        return False

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
                # print(vertex)
                # print(fullGraph)
                for node, relationship in zip(graph[vertex].nonzero()[1], graph[vertex].data):
                    if (relationship == 3) and (routingTree[node, vertex] == 0 and routingTree[vertex, node] == 0) and (
                        (not levels[node] <= level) or (levels[node] == -1)):
                        routingTree[node, vertex] = 1
                        if BFS[-1][0] == level:
                            BFS.append((level + 1, [node]))
                            levels[node] = level + 1
                        else:
                            BFS[-1][1].append(node)
                            levels[node] = BFS[-1][0]
                    elif (relationship == 3) and (routingTree[node, vertex] == 0 and routingTree[vertex, node] == 0):
                        routingTree[node, vertex] = 1
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
        oldNodes = []
        old = {}
        allNodes = set(np.append(graph.nonzero()[1], graph.nonzero()[0]))
        for node in allNodes:
            old[node] = 0

        for pair in BFS:
            oldNodes.extend(pair[1])
            for node in pair[1]:
                old[node] = 1
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
        edgesCount = 0
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
                        routingTree[node, vertex] = 1
                        if BFS[-1][0] == level:
                            BFS.append((level + 1, [node]))
                            levels[node] = level + 1
                        else:
                            BFS[-1][1].append(node)
                            levels[node] = BFS[-1][0]
                    elif (relationship == 2) and (routingTree[vertex, node] == 0 and routingTree[node, vertex] == 0):
                        routingTree[node, vertex] = 1
        return routingTree

    def saveAsNPZ(fileName, matrix):
        matrixCOO = matrix.tocoo()
        row = matrixCOO.row
        col = matrixCOO.col
        data = matrixCOO.data
        shape = matrixCOO.shape
        np.savez(fileName, row=row, col=col, data=data, shape=shape)

    def makeRoutingTree(destinationNode, fullGraph):
        '''
        input:
            destination AS
        output:
            routing tree of destination AS in sparse matrix format
        '''

        print("=================" + str(destinationNode) + "=======================")
        if "dcomplete" + str(destinationNode) + '.npz' in file_name:
            print('exist')
            return
        routingTree = sparse.dok_matrix((numNodes + 1, numNodes + 1), dtype=np.int8)
        # print('numNodes',numNodes)
        # print('routingTree',routingTree)
        stepOneRT, stepOneNodes, lvls = customerToProviderBFS(destinationNode, routingTree, fullGraph)
        stepTwoRT, stepTwoNodes, lvlsTwo = peerToPeer(stepOneRT, stepOneNodes, fullGraph, lvls)
        stepThreeRT = providerToCustomer(stepTwoRT, stepTwoNodes, fullGraph, lvlsTwo)
        saveAsNPZ(os.path.join(str(args[3]), "dcomplete" + str(destinationNode)), stepThreeRT)

    ### Helper Functions ###

    # Interpret User Input
    verbose = False
    if (args[2] == 'v'):
        verbose = True
    # interpretArgs(args) #TODO

    ### initialization phase ###
    fullGraph = scipy.io.mmread(str(args[1])).tocsr()  # read the graph on all ranks
    # print(args)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ### Get the node count and node list
    if (rank == 0):
        nodeListFile = str(args[4])
        nodeList = []
        print('nodeListFile', nodeListFile)
        with open(nodeListFile) as fp:
            line = fp.readline()
            while line:
                if (line[-1] == '\n'):
                    line = line[:-1]
                nodeList.append(int(line))
                line = fp.readline()
        if (verbose):
            print("Max ASNode ID: " + str(max(nodeList)))
    else:
        nodeList = None
    nodeList = comm.bcast(nodeList, root=0)
    numNodes = int(args[5])  # max(nodeList)
    
    # firstIndex, lastIndex = getRankBounds(nodeList)
    file_name = os.listdir(str(args[3]))
    # if (verbose):  ### Printing MPI Status for debugging purposes
    #     print("MPI STATUS... rank " + str(rank) + " reporting... working on nodes " + str(firstIndex) + " to " + str(
    #         lastIndex))

    comm.Barrier()  ## Synchronize here, then continue ##

    timer = {'start': 0.0, 'end': 0.0}
    timer['start'] = time.time()

    ### Primary Loop, executed distrubitively in parallel
    nodeList = gl_filter_rtree(nodeList, gl_asn_data)
    print('len(nodeList)',len(gl_filter_rtree(nodeList, gl_asn_data)))
    thread_pool = ThreadPool(processes=multiprocessing.cpu_count() * 10)
    for destinationNode in nodeList:
        #    random_index = get_random_index()
        #    print('random_index,len(nodeList) %s %s' % (random_index,len(nodeList)))
        file_name = os.listdir(str(args[3]))
        #    destinationNode = nodeList[index]
        #    destinationNode = 39642
        try:
            # t = threading.Thread(target=makeRoutingTree,args=(destinationNode,routingTree,fullGraph,))
            # t.start()
            # threads.append(t)
            thread_pool.apply_async(makeRoutingTree, (
                destinationNode,
                fullGraph,
            ))
        except Exception as e:
            print('Exception', e)
    # for tt in threads:
    #     tt.join()
    thread_pool.close()
    thread_pool.join()
    #    routingTree = makeRoutingTree(destinationNode)  ### Calculate the routing tree for this node

    # for index in [36344,21326,19368,6621,33004]:
    #     # file_name = os.listdir(str(args[3]))
    #     destinationNode = index
    #     routingTree = makeRoutingTree(destinationNode)  ### Calculate the routing tree for this node

    # with open('/data/lyj/shiyan_database/ccExternal/globalCountryLabel/add_hidden_link/cal_rtree_code_v2.json', 'r') as f:
    #     cal_node = json.load(f)
    # cal_node.reverse()
    # cal_node = cal_node[int(len(cal_node)/2):]
    # with open('/data/lyj/shiyan_database/ccExternal/globalCountryLabel/add_hidden_link/rtree/errorfile.txt', 'a') as err_f:
    #     for nodename in cal_node:
    #         file_name = os.listdir(str(args[3]))
    #         try:
    #             routingTree = makeRoutingTree(int(nodename))
    #         except:
    #             err_f.write(nodename+'\n')

    ### wait for all ranks to check time
    comm.Barrier()
    timer['end'] = time.time()
    if (rank == 0):
        print("All Routing Trees Completed. Elapsed Time: " + str((timer['end'] - timer['start'])))


'''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''


def create_rela_file(relas, relaFile):
    sum = 0
    # print('relaFile',relas)
    with open(relaFile, 'w') as f:
        for c in relas:
            sum += 1
            for b in relas[c][1]:
                f.write(str(c) + '|' + str(b) + '|-1\n')
            for b in relas[c][2]:
                if c <= b:
                    f.write(str(c) + '|' + str(b) + '|0\n')
    # print(sum)


def run_routingTree(dsn_file, relaFile, cc):
    dataConverter(relaFile, os.path.join(gl_dst_dir_path, 'temp', '%s_bgp-sas.npz' % cc))
    maxNum = graphGenerator(os.path.join(gl_dst_dir_path, 'temp', '%s_bgp-sas.npz' % cc),
                            os.path.join(gl_dst_dir_path, 'temp', '%s_routingTree.mtx' % cc))
    speedyGET([
        '',
        os.path.join(gl_dst_dir_path, 'temp', '%s_routingTree.mtx' % cc), 'v', dsn_file,
        os.path.join(gl_dst_dir_path, 'temp', '%s_routingTree.mtx.nodeList' % cc),
        str(maxNum)
    ])


'''生成一个国家内的排名前10的AS的路由树'''


def rTree(relas, dsn_file, cc):

    relaFile = os.path.join(dsn_file, 'as-rel.txt')
    create_rela_file(relas, relaFile)
    run_routingTree(dsn_file, relaFile, cc)


'''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''


def create_relas(
    file,
    if_del_stub_as=False,
):
    # global relas
    relas = {}
    with open(gl_as_rela_file_path, 'r') as f:
        as_rela = json.load(f)
        # print('len(as_rela.keys())',len(as_rela.keys()))
        # print(as_rela.keys())
    with open(file, 'r') as f:
        cclist = json.load(f)
    for c in cclist:

        # if c not in gl_asn_data or gl_asn_data[c] <= 1:
        #     continue

        
        

        relas[c] = [[], [], []]

    for c in relas:
        if c in as_rela:
            relas[c][2] += [i for i in as_rela[c][0] if i in cclist]
            for i in relas[c][2]:
                relas[i][2].append(c)
            relas[c][1] += [i for i in as_rela[c][1] if i in cclist]
            for i in relas[c][1]:
                relas[i][0].append(c)

    if if_del_stub_as:
        l = list(relas.keys())
        for c in l:
            relas[c][0] = list(set(relas[c][0]))
            relas[c][1] = list(set(relas[c][1]))
            relas[c][2] = list(set(relas[c][2]))
            if c in relas[c][0]: relas[c][0].remove(c)
            if c in relas[c][1]: relas[c][1].remove(c)
            if c in relas[c][2]: relas[c][2].remove(c)
            if not (len(relas[c][1]) or len(relas[c][2])):
                for i in relas[c][0]:
                    relas[i][1].remove(c)
                for i in relas[c][2]:
                    relas[i][2].remove(c)
                del relas[c]
    return relas


'''
file cc2as里面的数据
dsnpath 输出路径
country 国家代码
gl_as_rela_file_path 国家对应的topo关系文件

目的 生成对应国家排名前10的路由树文件
'''


@record_launch_time_and_param(2)
def monitor_routingTree(file:str, dsn_path:str, country:str):
    country_name:List[str] = os.listdir(dsn_path)
    dsn_path:str = os.path.join(dsn_path, country)
    if country not in country_name:
        os.makedirs(dsn_path, exist_ok=True)
    print(country + ' begin')
    relas = create_relas(file)
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
    if len(relas) < 10: return
    rTree(relas, dsn_path, country)
    print(country + ' end')


def set_glabal_variable(as_rela_file:str, asn_data:Dict[str, int], model_path:str, _dst_dir_path:str) -> None:
    global gl_asn_data
    global gl_filter_rtree
    global gl_as_rela_file_path
    global gl_dst_dir_path

    gl_asn_data = asn_data
    dynamic_module = import_module(model_path)
    gl_filter_rtree= dynamic_module.filter_rtree
    gl_dst_dir_path = _dst_dir_path
    gl_as_rela_file_path = as_rela_file


@record_launch_time
def create_rtree(as_rela_file: str, _dst_dir_path: str, _type: str, cc_list: List[str], asn_data: Dict[str, int],
                 cc2as_path: str, model_path: str):

    set_glabal_variable(as_rela_file, asn_data, model_path, _dst_dir_path)

    rtree_path:str = os.path.join(_dst_dir_path, _type, 'rtree')
    mkdir(rtree_path)

    process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_list:
        try:
            cc_path: str = os.path.join(cc2as_path, cc + '.json')
            monitor_routingTree(cc_path, rtree_path, cc)
            # process_pool.apply_async(monitor_routingTree, (cc_path, rtree_path, cc))
        except Exception as e:
            print(e)
            raise e
    process_pool.close()
    process_pool.join()