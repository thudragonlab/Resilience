#!usr/bin/env python
# _*_ coding:utf8 _*_
import multiprocessing
import os
import itertools
import copy
import numpy as np
import json
import itertools
from multiprocessing.pool import ThreadPool
import random
from util import record_launch_time
# from itertools import izip
izip=zip


class monitor_cut():
    def __init__(self, n_node, n_link, file_path, dsn_path, del_n=False):
        self.file_name = file_path
        self.n_node = n_node
        self.n_link = n_link
        self.graph = {}
        self.dsn_path = dsn_path
        self.tempgraphname = file_path+'.graph.json'

        #存储结果：{[]:节点总数量，[queue]:节点数量}
        self.res = {}

        #创建图
        print(file_path+' graph create')
        self.from_npz_create_graph()
        print(self.res[''])
        with open(self.tempgraphname, 'w') as f:
            json.dump(self.graph, f)

        with open(self.dsn_path, 'w') as f:
            f.write('#|'+str(self.res[''])+'\n')

        if del_n:
            if len(self.graph)<self.n_node:
                self.n_node = len(self.graph)//2
                self.n_link = len(self.graph)//2
        
        print(file_path+' monitor node')
        self.monitor_random_node_addDel()
        self.monitor_random_link_addDel()

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
            if a not in self.graph: self.graph[a] = [[], []]
            if b not in self.graph: self.graph[b] = [[], []]
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        #monitor_cut_node(self.graph, list(self.graph.keys())[2])
        self.res[''] = len(self.graph)


    def monitor_random_node_addDel(self):
        nodelist = set(self.row)
        tempG = len(self.graph)
        print(self.dsn_path)
        # if len(nodelist)<1000:
        #     epoch = len(nodelist)
        # else:
        #     epoch = 1000
        nodelist_len = len(nodelist)
        with open(self.dsn_path, 'a') as f:
            for num in range(1,self.n_node):
                flag = 0
                while flag<nodelist_len:
                # while flag<epoch:
                    flag+=1
                    node = random.sample(nodelist, num)
                    node = list(set(list(node)))
                    node.sort()
                    temp = ' '.join(list(map(str, node)))
                    linkres = self.monitor_cut_node(node)
                    # f.write(temp + '|' + str(linkres) + '\n')
                    f.write(temp+'|'+' '.join(list(map(str,linkres)))+'\n')
                    if len(self.graph) != tempG:
                        with open(self.tempgraphname, 'r') as ff:
                            self.graph = json.load(ff)


    def monitor_random_link_addDel(self):
        linklist = list(zip(self.row, self.col))
        tempG = len(self.graph)
# # +
        # if len(linklist)<1000:
        #     epoch = len(linklist)
        # else:
        #     epoch = 1000
# +  
        linklist_len = len(linklist)
        with open(self.dsn_path, 'a') as f:
            for num in range(self.n_link):
                # size = 500
                # flag = size-1
                flag = 0
                while flag<linklist_len:
                # while flag<epoch:
                    flag+=1
                    link = random.sample(linklist, num)
                    link = list(set(list(link)))
                    link.sort()
                    temp = ' '.join(list(map(str, link)))
                    linkres = self.monitor_cut_link(link)
                    # f.write(temp + '|' + str(linkres) + '\n')
                    f.write(temp+'|'+' '.join(list(map(str,linkres)))+'\n')
                    if len(self.graph) != tempG:
                        with open(self.tempgraphname, 'r') as ff:
                            self.graph = json.load(ff)

                # for i in itertools.permutations(linklist, num):
                #     if flag>=size-1:
                #         choise = np.random.uniform(0, len(linklist)**num+1, size)
                #         flag = -1
                #     flag += 1
                #     if choise[flag]<=len(linklist):
                #         link = list(set(list(linklist)))
                #         link.sort()
                #         temp = ' '.join(list(map(str, link)))
                #         linkres = self.monitor_cut_link(link)
                #         f.write(temp + '|' + ' '.join(linkres) + '\n')
                #         if len(self.graph) != tempG:
                #             with open(self.tempgraphname, 'r') as ff:
                #                 self.graph = json.load(ff)

    def monitor_random_node(self):
        nodelist = set(self.row)
        for i in itertools.combinations_with_replacement(nodelist, self.n_node):
            node = list(set(list(i)))
            node.sort()
            # print(node)
            temp = ' '.join(list(map(str, node)))
            self.res[temp] = self.monitor_cut_node(node)
            if len(self.graph) != len(tempG):
                #self.graph = copy.deepcopy(tempG)
                with open(self.tempgraphname, 'r') as f:
                    self.graph = json.load(f)
    
    
    def monitor_random_link(self):
        linklist = list(zip(self.row, self.col))
        tempG = copy.deepcopy(self.graph)
        for i in itertools.combinations_with_replacement(linklist, self.n_link):
            link = list(set(list(i)))
            link.sort()
            # print(link)
            temp = ' '.join(list(map(str, link)))
            self.res[temp] = self.monitor_cut_link(link)
            if len(self.graph)!=len(tempG):
                # self.graph = copy.deepcopy(tempG)
                with open(self.tempgraphname, 'r') as f:
                    self.graph = json.load(f)


    def monitor_cut_node(self, queue):
        res = []
        for node in queue:
            for i in self.graph[node][1]:
                self.graph[i][0].remove(node)

            self.graph[node][1] = []

        while queue:
            n = queue.pop(0)
            res.append(n)
            if n not in self.graph: continue

            for i in self.graph[n][0]:
                self.graph[i][1].remove(n)
                if len(self.graph[i][1])==0: queue.append(i)
            del self.graph[n]
        return res


    def monitor_cut_link(self, queue):
        '''
        queue格式是【[a,b],[c,d]】
        '''
        tempQ = []
        for link in queue:
            a,b = link
            self.graph[a][1].remove(b)
            self.graph[b][0].remove(a)
            if len(self.graph[a][1])==0: tempQ.append(a)

        for link in queue:
            a, b = link
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        return tempQ


def monitor_cut_class2func_inter(f):
    print(f)
    index = f.rfind('/')
    if f[index+1:][:-4]+'.addDel.txt' not in os.listdir(f[:index]):
        monitor_cut(5,2,f,f[:-4]+'.addDel.txt', True)
    else:
        print('exist')
    return 0


def do_cut_by_cc(cc,path,asn_data):
    print('country: '+cc)
        # if cc=='US' or cc=='RU' or cc=='BR' or cc=='BY': continue
        # if cc=='US': continue
    file_name = os.listdir(os.path.join(path,cc))
    node_num = []
    for f in file_name:
        if f.find('.json')!=-1 or f.find('.txt')!=-1 or f[-4:]!='.npz' or f[:3]!='dco':
            continue
        try:
            m = np.load(path+cc+'/'+f)
        except Exception as e:
            print(e)
        # node_num.append([f,len(set(list(m['row'])))])
        if f[9:-4] not in asn_data:
            continue
        node_num.append([f,asn_data[f[9:-4]]])
    file = sorted(node_num, key=lambda x: x[1],reverse=True)
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    file_name = os.listdir(os.path.join(path,cc))
    # print(file)
    for f in file[:10]:
        if f[0][:-4]+'.addDel.txt' in file_name: 
            print('exist')
            continue
        try:
            thread_pool.apply(monitor_cut_class2func_inter,(os.path.join(path,cc,f[0]),))
        except Exception as e:
            print(e)
            raise e
        # fes.append(os.path.join(path,cc,f[0]))
    # pool.map(monitor_cut_class2func_inter,fes)
    thread_pool.close()
    thread_pool.join()

@record_launch_time
def monitor_country_internal(prefix,_type,asn_data):
    path = os.path.join(prefix,_type,'rtree/')
    file = os.listdir(path)
    file.reverse()
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in file:
        pool.apply_async(do_cut_by_cc,(cc,path,asn_data,))
    pool.close()
    pool.join()


# 需要定义path
# path含义和create_routingtree.py中p2的含义相当
# prefix = '/home/peizd01/for_dragon/new_data/'
# path = os.path.join(prefix,'asRank/rtree/')
# monitor_country_internal(path)