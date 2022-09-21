#!usr/bin/env python
# _*_ coding:utf8 _*_
'''

生成国家内部routingTree

'''
from ast import Lambda
import os
import json
import copy
import numpy as np
import time
from mpi4py import MPI
from scipy import sparse
import scipy.io
import itertools
# from itertools import izip
izip=zip


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
                except:
                    print(data)
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
        print(end - start, " seconds to load")  # 2.3 seconds
        return numNodes

    return fileToSparse(inFile, outFile)


def speedyGET(args):
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
                for node, relationship in izip(fullGraph[vertex].nonzero()[1], fullGraph[vertex].data):
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
                for node, relationship in izip(fullGraph[vertex].nonzero()[1], fullGraph[vertex].data):
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
                for node, relationship in izip(fullGraph[vertex].nonzero()[1], fullGraph[vertex].data):
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

    def makeRoutingTree(destinationNode):
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
        stepOneRT, stepOneNodes, lvls = customerToProviderBFS(destinationNode, routingTree, fullGraph)
        stepTwoRT, stepTwoNodes, lvlsTwo = peerToPeer(stepOneRT, stepOneNodes, fullGraph, lvls)
        stepThreeRT = providerToCustomer(stepTwoRT, stepTwoNodes, fullGraph, lvlsTwo)
        saveAsNPZ(os.path.join(str(args[3]), "dcomplete" + str(destinationNode)), stepThreeRT)
        return stepThreeRT

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
    print(args)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numRanks = comm.Get_size()

    ### Get the node count and node list
    if (rank == 0):
        nodeListFile = str(args[4])
        nodeList = []
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
    firstIndex, lastIndex = getRankBounds(nodeList)
    file_name = os.listdir(str(args[3]))
    if (verbose):  ### Printing MPI Status for debugging purposes
        print("MPI STATUS... rank " + str(rank) + " reporting... working on nodes " + str(firstIndex) + " to " + str(
            lastIndex))

    comm.Barrier()  ## Synchronize here, then continue ##

    timer = {'start': 0.0, 'end': 0.0}
    timer['start'] = time.time()

    ### Primary Loop, executed distrubitively in parallel
    
    # for index in range(int(firstIndex), int(lastIndex) + 1):
    #    file_name = os.listdir(str(args[3]))
    #    destinationNode = nodeList[index]
    #    routingTree = makeRoutingTree(destinationNode)  ### Calculate the routing tree for this node
    

    with open('/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/cal_rtree_code_v2.json') as ff:
        asns = json.load(ff)
        asns_list = list(map(lambda x : int(x),asns))

        for index in asns_list:
            # file_name = os.listdir(str(args[3]))
            destinationNode = index
            routingTree = makeRoutingTree(destinationNode)  ### Calculate the routing tree for this node
        

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

def run_routingTree(dsn_file, relaFile):
    print('!!!')
    dataConverter(relaFile, 'bgp-sas.npz')
    print('!!!??')
    maxNum = graphGenerator('bgp-sas.npz', 'routingTree.mtx')
    speedyGET(['','routingTree.mtx', 'v', dsn_file, 'routingTree.mtx.nodeList', str(maxNum)])


def rTree(relas, dsn_file):
    
    relaFile = os.path.join(dsn_file, 'as-rel.txt')
    #if len(relas)==0:
    #    return
    # create_rela_file(relas, relaFile)
    run_routingTree(dsn_file, as_rela_file)
    
def create_relas(file, if_del_stub_as=False):
    '''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''
    global relas
    with open(as_rela_file, 'r') as f:
        as_rela = json.load(f)
        # print(as_rela.keys())
    with open(file, 'r') as f:
        cclist = json.load(f)
    for c in cclist:
        relas[c] =[[],[],[]]
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




def monitor_routingTree(file,dsn_path):
    global relas
    # country_name = os.listdir(dsn_path)
    dsn_path = os.path.join(dsn_path, 'test_for_external')
    # if country not in country_name:
    #     os.popen('mkdir '+dsn_path)
    # print(country+' begin')
    # create_relas(file)
    # if country=='PT':
    #     print(relas)
    # if len(relas)<10: return
    rTree(relas, dsn_path)
    # print(country+' end')



def main():
    global relas
    with open(old_rank_file, 'r') as f: temp = json.load(f)
    if not os.path.exists(p2): os.makedirs(p2)
    cces = list(temp.keys())
    cces.reverse()
    # for cc in cces:
    #     # if cc == 'US':
    #     #     continue
    #     # if cc in ['BR', 'US', 'RU']: continue
    #     relas = {}
    monitor_routingTree('/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/as_rela_code.txt', p2)




# 需要定义old_rank_file、p1、p2、as_rela_file
# 从old_rank_file拿到需要计算的区域，可以自己定义，具体在main()使用
# p1是存储每个国家的AS
# p2是结果输出路径，生成每个AS的routingTree 
# as_rela_file存储全球的as关系
old_prefix = '/home/peizd01/for_dragon/pzd_External/'
prefix = '/home/peizd01/for_dragon/pzd_External/'
old_rank_file = os.path.join(old_prefix,'public/rank_2.json')
p1 = os.path.join(prefix,'public/cc2as/')
p2 = os.path.join(prefix,'globalCountryLabel/add_hidden_link/rtree/')
# as_rela_file = '/home/peizd01/for_dragon/new_data_pzd/as_rela_code-as_rela_from_toposcope_hidden.json'
as_rela_file = '/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/as_rela_code.txt'
relas = {}
main()
# [36344,21326,19368,6621,33004]