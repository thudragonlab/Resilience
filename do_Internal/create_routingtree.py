#!usr/bin/env python
# _*_ coding:utf8 _*_
'''

生成国家内部routingTree

'''
from glob import glob
import os
import json
import copy
import numpy as np
import time
from mpi4py import MPI
from scipy import sparse
import scipy.io
import itertools
import multiprocessing
from multiprocessing.pool import ThreadPool
# from itertools import izip
from random import randint
# from do_Internal.create_rtree_model.sort_by_cone_top_100 import filter_rtree
from do_Internal.create_rtree_model.sort_by_cone_top_50 import filter_rtree
from util import record_launch_time
# from do_Internal.create_rtree_model.sort_by_cone_top_10 import filter_rtree
# from do_Internal.create_rtree_model.random_100 import filter_rtree
# from do_Internal.create_rtree_model.all_tree import filter_rtree
# from do_Internal.create_rtree_model.default import filter_rtree

asn_data_global = {}


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
                for node, relationship in zip(fullGraph[vertex].nonzero()[1], fullGraph[vertex].data):
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
                for node, relationship in zip(fullGraph[vertex].nonzero()[1], fullGraph[vertex].data):
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
                for node, relationship in zip(fullGraph[vertex].nonzero()[1], fullGraph[vertex].data):
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

    def makeRoutingTree(destinationNode,routingTree,fullGraph):
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
        # print('numNodes',numNodes)
        # print('routingTree',routingTree)
        stepOneRT, stepOneNodes, lvls = customerToProviderBFS(destinationNode, routingTree, fullGraph)
        # print('stepOneNodes',stepOneNodes)
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
    # def getRankBounds(nodeList):
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
    #     numRanks = comm.Get_size()

    #     dataSize = len(nodeList)

    #     if (verbose):
    #         print("Node Count: " + str(dataSize))

    #     dataSizePerRank = dataSize / numRanks
    #     leftOver = dataSize % numRanks
    #     startIndex = dataSizePerRank * rank
    #     lastIndex = (dataSizePerRank * (rank + 1)) - 1
    #     if (rank < leftOver):
    #         dataSizePerRank += 1
    #         if (rank != 0):
    #             startIndex += 1
    #             lastIndex += 2
    #         else:
    #             lastIndex += 1
    #     else:
    #         startIndex += leftOver
    #         lastIndex += leftOver

    #     return startIndex, lastIndex

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
    numRanks = comm.Get_size()

    ### Get the node count and node list
    if (rank == 0):
        nodeListFile = str(args[4])
        nodeList = []
        print('nodeListFile',nodeListFile)
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
    routingTree = sparse.dok_matrix((numNodes + 1, numNodes + 1), dtype=np.int8)
    # firstIndex, lastIndex = getRankBounds(nodeList)
    file_name = os.listdir(str(args[3]))
    # if (verbose):  ### Printing MPI Status for debugging purposes
    #     print("MPI STATUS... rank " + str(rank) + " reporting... working on nodes " + str(firstIndex) + " to " + str(
    #         lastIndex))

    comm.Barrier()  ## Synchronize here, then continue ##

    timer = {'start': 0.0, 'end': 0.0}
    timer['start'] = time.time()

    ### Primary Loop, executed distrubitively in parallel
    
    nodeList = filter_rtree(nodeList,asn_data_global)
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
            thread_pool.apply_async(makeRoutingTree,(destinationNode,routingTree,fullGraph,))
       except Exception as e:
            print('Exception',e)
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
                f.write(str(c)+'|'+str(b)+'|-1\n')
            for b in relas[c][2]:
                if c<=b:
                    f.write(str(c)+'|'+str(b)+'|0\n')
    print(sum)

def run_routingTree(dsn_file, relaFile,_dst_dir_path,cc):
    dataConverter(relaFile, os.path.join(_dst_dir_path,'temp','%s_bgp-sas.npz' % cc))
    maxNum = graphGenerator(os.path.join(_dst_dir_path,'temp','%s_bgp-sas.npz' % cc), os.path.join(_dst_dir_path,'temp','%s_routingTree.mtx' % cc))
    speedyGET(['',os.path.join(_dst_dir_path,'temp','%s_routingTree.mtx' % cc), 'v', dsn_file, os.path.join(_dst_dir_path,'temp','%s_routingTree.mtx.nodeList' % cc), str(maxNum)])


'''生成一个国家内的排名前10的AS的路由树'''
def rTree(relas, dsn_file,_dst_dir_path,cc):
    
    relaFile = os.path.join(dsn_file, 'as-rel.txt')
    #if len(relas)==0:
    #    return
    create_rela_file(relas, relaFile)
    run_routingTree(dsn_file, relaFile,_dst_dir_path,cc)
    


'''
    从国家的as，以及as关系文件，提取出国家内部拓扑的relas [provider、customer、peer]
    '''
def create_relas(file, as_rela_file,if_del_stub_as=False,):
    # global relas
    relas = {}
    with open(as_rela_file, 'r') as f:
        as_rela = json.load(f)
        # print('len(as_rela.keys())',len(as_rela.keys()))
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
    return relas



'''
file cc2as里面的数据
dsnpath 输出路径
country 国家代码
as_rela_file 国家对应的topo关系文件

目的 生成对应国家排名前10的路由树文件
'''
def monitor_routingTree(file,dsn_path,country,as_rela_file,_dst_dir_path):
    # global relas
    # print(file,dsn_path,country,as_rela_file,_dst_dir_path)
    country_name = os.listdir(dsn_path)
    dsn_path = os.path.join(dsn_path, country)
    if country not in country_name:
        os.makedirs(dsn_path,exist_ok=True)
        # os.popen('mkdir '+dsn_path)
    print(country+' begin')
    relas = create_relas(file,as_rela_file)
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
    # print('finish',country)
    if len(relas)<10: return
    rTree(relas, dsn_path,_dst_dir_path,country)
    print(country+' end')


@record_launch_time
def create_rtree(_as_rela_file,_dst_dir_path,_type,cc_list,asn_data,cc2as_path):
    
    # global relas
    # global as_rela_file
    global asn_data_global
    asn_data_global = asn_data
    process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # old_rank_file = '/home/peizd01/for_dragon/public/rank_2.json'
    p1 = cc2as_path
    p2 = os.path.join(_dst_dir_path,_type,'rtree')
    as_rela_file = _as_rela_file
    # print(_as_rela_file,_dst_dir_path,_type,cc_list)
    # with open(old_rank_file, 'r') as f: temp = json.load(f)
    if not os.path.exists(p2): os.makedirs(p2)
    # cces = list(temp.keys())
    # cces.reverse()
    # print(cces)
    cces = cc_list
    for cc in cces:
        # if cc != 'US':
        #     continue
        # if cc not in ['BR', 'US', 'RU']: continue
        # print(cc)
        # relas = {}
        try:
            process_pool.apply_async(monitor_routingTree,(os.path.join(p1,cc+'.json'), p2, cc,as_rela_file,_dst_dir_path))
        except Exception as e:
            print(e)
            raise e
            
        # monitor_routingTree(os.path.join(p1,cc+'.json'), p2, cc,as_rela_file,_dst_dir_path)
    process_pool.close()
    process_pool.join()




# 需要定义old_rank_file、p1、p2、as_rela_file
# 从old_rank_file拿到需要计算的区域，可以自己定义，具体在main()使用
# p1是存储每个国家的AS
# p2是结果输出路径，生成每个AS的routingTree 
# as_rela_file存储全球的as关系
# dst_dir_path = '/home/peizd01/for_dragon/pzd_python/do_Internal/output'
# # old_prefix = dst_dir_path
# prefix = dst_dir_path
# p2 = os.path.join(prefix,'asRank/rtree/')

# as_rela_file = '/home/peizd01/for_dragon/pzd_python/do_Internal/source/20220801.as-rel-as_rela_asRank.json'

# main()
# [36344,21326,19368,6621,33004]