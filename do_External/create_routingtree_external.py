#!usr/bin/env python
# _*_ coding:utf8 _*_
'''

生成国家内部routingTree

'''
from ast import Lambda
import os
import json
import copy
from typing import Callable, Dict
import numpy as np
import time
from mpi4py import MPI
from scipy import sparse
import scipy.io
import itertools
# from itertools import izip
from multiprocessing.pool import ThreadPool
from importlib import import_module
import multiprocessing
izip=zip
relas = {}

gl_filter_rtree:Callable
source_path:str
gl_incoder:Dict[str,str]

def dataConverter(inFile, outFile):
    of = open(outFile, 'w')

    with open(inFile) as fp:
        line = fp.readline()
        cnt = 1
        nodeList = []
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
                    print(data,e)
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

        print("number of unique nodes: " + str(len(nodeList)))

    of.close()

    ### Saving the node list ###
    outFileList = outFile + ".nodeList"
    ouf = open(outFileList, 'w')
    for node in nodeList:
        ouf.write(str(node) + "\n")
    ouf.close()

def graphGenerator(inFile, outFile,v2_path):
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
        print("Node Count: " + str(len(nodeList)))
        print("Max Node ID: " + str(max(nodeList)))

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

        # numNodes = determineNodeCount(fileName, outName)
        with open(fileName,'r') as ff:
            numNodes = -1
            for i in ff.readlines():
                if len(i.split('\t')) < 3:
                    continue
                as1,as2,_ = i.split('\t')
                numNodes = max(int(as1),numNodes)
                numNodes = max(int(as2),numNodes)
            # asns = json.load(ff)
            # asns_list = list(map(lambda x : int(x),asns))
            # numNodes = max(asns_list)
            print('max(asns_list)',numNodes)
        # numNodes = determineNodeCount(fileName, outName)

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
            # if node1 not in asns or node2 not in asns:
            #     continue
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
        print(end - start, " seconds to load")  # 2.3 seconds
        return numNodes

    return fileToSparse(inFile, outFile)

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
            
            for node, relationship in izip(graph[vertex].nonzero()[1], graph[vertex].data):
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
            for node, relationship in izip(graph[vertex].nonzero()[1], graph[vertex].data):
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
            for node, relationship in izip(graph[vertex].nonzero()[1], graph[vertex].data):
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

def saveAsNPZ(del_path,destinationNode, matrix):
    matrixCOO = matrix.tocoo()
    row = matrixCOO.row
    col = matrixCOO.col
    data = matrixCOO.data
    
    shape = matrixCOO.shape
    # if len(data) == 0:
    #     return
    np.savez(os.path.join(del_path, "dcomplete" + str(destinationNode)), row=row, col=col, data=data, shape=shape)
    remove_internal_link(destinationNode,matrixCOO)

def remove_internal_link(asn,m):
    
    def resolve(s):
        return [s.split('-')[0], s.split('-')[1]]
    
    json_path = os.path.join(source_path,'json')
    
    broad = []
    link = list(zip(m.row, m.col))

        # 第一次遍历，存储边界AS代码
    for a, b in link:
        as1, cy1 = resolve(gl_incoder[str(a)])
        as2, cy2 = resolve(gl_incoder[str(b)])
        if cy1 != cy2 and 'None' not in [as1, as2, cy1, cy2]:
                # if cy1!=cy2:
            broad.append(str(a))
            broad.append(str(b))

    for index in range(len(link) - 1, -1, -1):
        as1, cy1 = resolve(gl_incoder[str(link[index][0])])
        as2, cy2 = resolve(gl_incoder[str(link[index][1])])
        if str(link[index][0]) in broad and str(link[index][1]) in broad:
            link[index] = [gl_incoder[str(link[index][0])], gl_incoder[str(link[index][1])]]
        else:
            link.pop(index)

        # 记录link列表（国家A-》国家B）：[link]
    cc_pair_link = {}
        # 以国家为单位，找到前向国家，后向国家
    cc_rela = {}
    for a, b in link:
        as1, cy1 = resolve(a)
        as2, cy2 = resolve(b)
        if cy1 not in cc_rela:
            cc_rela[cy1] = [set(), set()]
        if cy2 not in cc_rela:
            cc_rela[cy2] = [set(), set()]
        cc_rela[cy1][1].add(cy2)
        cc_rela[cy2][0].add(cy1)

        if cy1 + ' ' + cy2 not in cc_pair_link:
            cc_pair_link[cy1 + ' ' + cy2] = set()
        cc_pair_link[cy1 + ' ' + cy2].add(as1 + ' ' + as2)

    for key in cc_pair_link.keys():
        cc_pair_link[key] = list(cc_pair_link[key])
    for key in cc_rela:
        cc_rela[key][0] = list(cc_rela[key][0])
        cc_rela[key][1] = list(cc_rela[key][1])
    # print(json_path)
    with open(os.path.join(json_path, f'dcomplete{asn}.cc_pair_link.json'), 'w') as f:
        json.dump(cc_pair_link, f)
    with open(os.path.join(json_path, f'dcomplete{asn}.cc_rela.json'), 'w') as f:
        json.dump(cc_rela, f)


def makeRoutingTree(destinationNode,fullGraph,file_name,numNodes,del_path):
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
    
    routingTree = sparse.dok_matrix((int(numNodes) + 1, int(numNodes) + 1), dtype=np.int8)
    
    stepOneRT, stepOneNodes, lvls = customerToProviderBFS(destinationNode, routingTree, fullGraph)
    stepTwoRT, stepTwoNodes, lvlsTwo = peerToPeer(stepOneRT, stepOneNodes, fullGraph, lvls)
    stepThreeRT = providerToCustomer(stepTwoRT, stepTwoNodes, fullGraph, lvlsTwo)
    saveAsNPZ(del_path,destinationNode, stepThreeRT)
    return stepThreeRT


def speedyGET(args):
    # def checkPreviousLevelsAlt(BFS, node, level):
    #     '''
    #     check if node is in BFS at given level or any previous level
    #     '''
    #     while level >= 0:
    #         if node in BFS[level][1]:
    #             return True
    #         level -= 1
    #     return False


    ### Helper Functions ###

    ### getRankBounds
    ### Input:
    ###   nodeListFile, a file holding list of ASNode IDs
    ### Output:
    ###   bounds, a dictionary describing the bounds of the rank
    def getRankBounds(nodeList):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        numRanks = comm.Get_size()

        dataSize = len(nodeList)

        if (verbose):
            print("Node Count: " + str(dataSize))

        dataSizePerRank = dataSize / numRanks
        leftOver = dataSize % numRanks
        startIndex = dataSizePerRank * rank
        lastIndex = (dataSizePerRank * (rank + 1)) - 1
        if (rank < leftOver):
            dataSizePerRank += 1
            if (rank != 0):
                startIndex += 1
                lastIndex += 2
            else:
                lastIndex += 1
        else:
            startIndex += leftOver
            lastIndex += leftOver

        return startIndex, lastIndex

    # Interpret User Input
    verbose = False
    if (args[2] == 'v'):
        verbose = True
    # interpretArgs(args) #TODO

    ### initialization phase ###
    fullGraph = scipy.io.mmread(str(args[1])).tocsr()  # read the graph on all ranks
    
    file_name = os.listdir(str(args[3]))
    # if (verbose):  ### Printing MPI Status for debugging purposes
    #     print("MPI STATUS... rank " + str(rank) + " reporting... working on nodes " + str(firstIndex) + " to " + str(
    #         lastIndex))

    # comm.Barrier()  ## Synchronize here, then continue ##

    # timer = {'start': 0.0, 'end': 0.0}
    # timer['start'] = time.time()

    ### Primary Loop, executed distrubitively in parallel
    
    # for index in range(int(firstIndex), int(lastIndex) + 1):
    #    file_name = os.listdir(str(args[3]))
    #    destinationNode = nodeList[index]
    #    routingTree = makeRoutingTree(destinationNode)  ### Calculate the routing tree for this node

    # nodeList = gl_filter_rtree(nodeList, gl_asn_data)
    pool = multiprocessing.Pool(20)
    
    with open(args[6]) as ff:
        asns = json.load(ff)
        asns_list = list(map(lambda x : int(x),asns))
    numNodes = args[5]
    print('max(asns_list)',numNodes)
    for index in asns_list:
        # file_name = os.listdir(str(args[3]))
        destinationNode = index
        # routingTree = makeRoutingTree(destinationNode,fullGraph,file_name,int(numNodes),str(args[3]))  ### Calculate the routing tree for this node
        pool.apply_async(makeRoutingTree,(destinationNode,fullGraph,file_name,numNodes,str(args[3],)))
    pool.close()
    pool.join()
    # try:
    #     for index in asns_list:
    #         # file_name = os.listdir(str(args[3]))
    #         destinationNode = index
    #         # routingTree = makeRoutingTree(destinationNode,fullGraph,file_name,int(numNodes),str(args[3]))  ### Calculate the routing tree for this node
    #         pool.apply_async(makeRoutingTree,(destinationNode,fullGraph,file_name,numNodes,str(args[3],)))
    #     pool.close()
    #     pool.join()
    # except Exception as e:
    #     print('Exception',e)
    
        

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
    # comm.Barrier()
    # timer['end'] = time.time()
    # if (rank == 0):
    #     print("All Routing Trees Completed. Elapsed Time: " + str((timer['end'] - timer['start'])))


def create_rela_file(relas, relaFile):
    '''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''
    sum = 0
    with open(relaFile, 'w') as f:
        for c in relas:
            sum += 1
            for b in relas[c][1]:
                f.write(str(c)+'|'+str(b)+'|-1\n')
            for b in relas[c][2]:
                if c<=b:
                    f.write(str(c)+'|'+str(b)+'|0\n')
    print(sum)

def run_routingTree(dsn_file,v2_path, relaFile):
    dataConverter(relaFile, 'bgp-sas.npz')
    # print(relaFile)
    maxNum = graphGenerator('bgp-sas.npz', 'routingTree.mtx',v2_path)
    speedyGET(['','routingTree.mtx', 'v', dsn_file, 'routingTree.mtx.nodeList', str(maxNum),v2_path])




def monitor_routingTree(as_rela_file,v2_path,dsn_path):
    global relas
    
    run_routingTree(dsn_path, v2_path,as_rela_file)



def main(prefix,v2_path,as_rela_file,build_rtree_model_path):
    global relas
    global gl_filter_rtree
    global source_path
    global gl_incoder
    
    source_path = prefix
    json_path = os.path.join(source_path,'json')
    os.makedirs(json_path, exist_ok=True)

    encoder_path = os.path.join(source_path,'as-country-code.json')
    with open(encoder_path, 'r') as f:
        encoder = json.load(f)
        incoder = {encoder[i]: i for i in encoder}
    gl_incoder = incoder
    dynamic_module = import_module(build_rtree_model_path)
    gl_filter_rtree= dynamic_module.filter_rtree
    p2 = os.path.join(prefix,'rtree/')
    old_rank_file = '/home/peizd01/for_dragon/pzd_External/public/rank_2.json'
    with open(old_rank_file, 'r') as f: temp = json.load(f)
    if not os.path.exists(p2): os.makedirs(p2)
    cces = list(temp.keys())
    cces.reverse()
    # for cc in cces:
    #     # if cc == 'US':
    #     #     continue
    #     # if cc in ['BR', 'US', 'RU']: continue
    #     relas = {}
    monitor_routingTree(as_rela_file,v2_path, p2)




# 需要定义old_rank_file、p1、p2、as_rela_file
# 从old_rank_file拿到需要计算的区域，可以自己定义，具体在main()使用
# p1是存储每个国家的AS
# p2是结果输出路径，生成每个AS的routingTree 
# as_rela_file存储全球的as关系

# p1 = os.path.join(prefix,'public/cc2as/')
# as_rela_file = '/home/peizd01/for_dragon/new_data_pzd/as_rela_code-as_rela_from_toposcope_hidden.json'


# main(prefix,as_rela_file)
# [36344,21326,19368,6621,33004]