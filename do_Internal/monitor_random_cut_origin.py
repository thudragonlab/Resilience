#!usr/bin/env python
# _*_ coding:utf8 _*_
import multiprocessing
import os
import copy
import numpy as np
import json
import itertools
from multiprocessing.pool import ThreadPool
import random
from importlib import import_module
from other_script.my_types import *
from other_script.util import record_launch_time, record_launch_time_and_param

izip = zip


class monitor_cut():

    def __init__(self, n_node, n_link, file_path, dsn_path, asn, del_n=False):
        '''
        n_node 最多破坏节点个数
        n_link 最多破坏链接个树 (没用)
        file_path rtree路径
        dsn_path 记录破坏结果路径
        asn: as code
        del_n 根据路由树大小重置破坏节点数

        '''
        self.file_name = file_path
        self.n_node = n_node
        self.n_link = n_link
        self.asn = asn
        self.graph = {}
        self.dsn_path = dsn_path
        self.tempgraphname = file_path + '.graph.json'

        # 存储结果：{[]:节点总数量，[queue]:节点数量}
        self.res = {}

        # 创建图
        self.from_npz_create_graph()
        # print(file_path + ' graph created')
        # print(self.res[''])
        with open(self.tempgraphname, 'w') as f:
            json.dump(self.graph, f)

        with open(self.dsn_path, 'w') as f:
            f.write('#|' + str(self.res['']) + '\n')

        if del_n:
            if len(self.graph) < self.n_node:
                self.n_node = len(self.graph) // 2
                self.n_link = len(self.graph) // 2

        self.monitor_random_node_addDel()

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
        # monitor_cut_node(self.graph, list(self.graph.keys())[2])
        self.res[''] = len(self.graph)
        for i in self.graph[self.asn][1]:
            self.graph[i][0].remove(self.asn)
            self.graph[self.asn][1].clear()

    def monitor_random_node_addDel(self):
        '''
        随机破坏节点,破坏结果存入addDel.txt
        '''
        nodelist = set(self.row)
        tempG = len(self.graph)
        # print(self.dsn_path)

        cut_times = gl_get_cut_num(nodelist)
        with open(self.dsn_path, 'a') as f:
            for num in range(1, self.n_node):
                flag = 0
                while flag < cut_times:
                    # while flag<epoch:
                    flag += 1
                    node = random.sample(nodelist, num)
                    node = list(set(list(node)))
                    node.sort()
                    temp = ' '.join(list(map(str, node)))
                    linkres = self.monitor_cut_node(node)
                    # if '6471' in node:

                    # f.write(temp + '|' + str(linkres) + '\n')
                    f.write(temp + '|' + ' '.join(list(map(str, linkres))) +
                            '\n')
                    if len(self.graph) != tempG:
                        with open(self.tempgraphname, 'r') as ff:
                            self.graph = json.load(ff)

    def monitor_cut_node(self, queue):
        '''
        queue 随机破坏的节点列表

        从根据路由树生成的图中计算被影响的节点
        '''
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
                if len(self.graph[i][1]) == 0: queue.append(i)
            del self.graph[n]
        return res


def monitor_cut_class2func_inter(f):
    '''
    f 具体的.npz rtree文件 
    '''
    monitor_cut(gl_cut_node_depth, 1, f, f[:-4] + '.addDel.txt', f.split('/')[-1][9:-4], True)

    return 0


@record_launch_time_and_param(0)
def do_cut_by_cc(cc: COUNTRY_CODE, path: RTREE_PATH, asn_data: Dict[AS_CODE, int]):
    '''
    cc country code
    path  rtree 路径
    asn_data as-cone字典

    对这国家的路由树选取并破坏
    '''
    print('country: ' + cc)
    file_name = os.listdir(os.path.join(path, cc))
    node_num = []
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for f in file_name:
        if f.find('.json') != -1 or f.find(
                '.txt') != -1 or f[-4:] != '.npz' or f[:3] != 'dco':
            continue
        if f[9:-4] not in asn_data:
            continue
        m = np.load(path + cc + '/' + f)
        node_num.append([f, asn_data[f[9:-4]], len(set(list(m['row'])))])

    # 调用选取破坏节点模块
    for f in gl_get_destroy_trees(node_num):
        try:
            thread_pool.apply_async(monitor_cut_class2func_inter,
                                    (os.path.join(path, cc, f[0]),))
        except Exception as e:
            print(e)
            raise e
    thread_pool.close()
    thread_pool.join()


@record_launch_time
def monitor_country_internal(prefix: OUTPUT_PATH, _type: TOPO_TPYE, asn_data: Dict[AS_CODE, int], destroy_model_path: str, cut_rtree_model_path: str,
                             cut_node_depth: int, cc_list: List[COUNTRY_CODE]):
    '''
    prefix output 路径
    _type topo类型
    asn_data as-cone 字典
    destroy_model_path 选取破坏节点模块路径
    cut_rtree_model_path 破坏次数模块路径
    cc_list 国家列表
    '''
    path = os.path.join(prefix, _type, 'rtree/')
    global gl_get_destroy_trees
    global gl_get_cut_num
    global gl_cut_node_depth
    dynamic_module_1 = import_module(destroy_model_path)
    dynamic_module_2 = import_module(cut_rtree_model_path)

    gl_cut_node_depth = cut_node_depth + 1
    gl_get_destroy_trees = dynamic_module_1.get_destroy_trees
    gl_get_cut_num = dynamic_module_2.get_cut_num

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_list:
        do_cut_by_cc (
            cc,
            path,
            asn_data,
        )
        # pool.apply(do_cut_by_cc, (
        #     cc,
        #     path,
        #     asn_data,
        # ))
    pool.close()
    pool.join()
