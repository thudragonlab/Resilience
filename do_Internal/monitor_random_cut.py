#!usr/bin/env python
# _*_ coding:utf8 _*_
import multiprocessing
import os
from other_script.my_types import *
import numpy as np
import json
import itertools
from multiprocessing.pool import ThreadPool
import random
from importlib import import_module
from other_script.util import record_launch_time, record_launch_time_and_param

izip = zip
gl_get_destroy_trees: Callable[[List[Tuple[str, int]]], List[Tuple[str, int]]] = None
gl_get_cut_num: Callable[[List[AS_CODE]], List[AS_CODE]] = None
gl_cc2as_path: CC2AS_PATH = None


class monitor_cut():

    def __init__(self, n_node: int, file_path: str, dsn_path: ADDDEL_PATH, asn: AS_CODE, cc2as_list_path: CC_PATH):
        '''
        n_node 最多破坏节点个数
        n_link 最多破坏链接个树 (没用)
        file_path rtree路径
        dsn_path 记录破坏结果路径
        asn: as code
        cc2as_list_path cc2as下的json文件路径

        '''
        self.file_name: str = file_path
        self.n_node: int = n_node
        # self.n_link = n_link
        self.asn: AS_CODE = asn
        self.graph: Dict[AS_CODE, List[List[AS_CODE]]] = {}
        self.dsn_path: ADDDEL_PATH = dsn_path
        self.tempgraphname: str = file_path + '.graph.json'
        with open(cc2as_list_path, 'r') as ff:
            self.all_as_list: List[AS_CODE] = json.load(ff)

        # 存储结果：{[]:节点总数量，[queue]:节点数量}

        # 创建图
        self.from_npz_create_graph()
        with open(self.tempgraphname, 'w') as f:
            json.dump(self.graph, f)

        with open(self.dsn_path, 'w') as f:
            f.write('#|' + str(len(self.all_as_list)) + '\n')
        if len(self.graph) < self.n_node:
            self.n_node = len(self.graph) // 2

        self.monitor_random_node_addDel()

    def from_npz_create_graph(self):
        '''
        存routingTree 【【前向点】【后向点】】 后向点为空说>明就脱离了routingTree
        '''
        m = np.load(self.file_name)
        self.row: List[AS_CODE] = [str(i) for i in m['row']]
        self.col: List[AS_CODE] = [str(i) for i in m['col']]
        link: List[Dict[AS_CODE, AS_CODE]] = list(zip(self.row, self.col))
        for l in link:
            a, b = l
            if a not in self.graph: self.graph[a] = [[], []]
            if b not in self.graph: self.graph[b] = [[], []]
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        for i in self.graph[self.asn][1]:
            self.graph[i][0].remove(self.asn)
            self.graph[self.asn][1].clear()

    def monitor_random_node_addDel(self):
        '''
        随机破坏节点,破坏结果存入addDel.txt
        '''
        nodelist: List[AS_CODE] = set(self.row)
        tempG: int = len(self.graph)

        cut_times: int = gl_get_cut_num(nodelist)
        with open(self.dsn_path, 'a') as f:
            for num in range(1, self.n_node):
                flag = 0
                while flag < cut_times:
                    flag += 1
                    node: List[AS_CODE] = random.sample(nodelist, num)
                    node = list(set(list(node)))
                    node.sort()
                    temp: str = ' '.join(list(map(str, node)))
                    linkres: List[AS_CODE] = self.monitor_cut_node(node)
                    f.write(temp + '|' + ' '.join(list(map(str, linkres))) +
                            '\n')
                    if len(self.graph) != tempG:
                        with open(self.tempgraphname, 'r') as ff:
                            self.graph = json.load(ff)

    def monitor_cut_node(self, queue: List[AS_CODE]):
        '''
        queue 随机破坏的节点列表

        从根据路由树生成的图中计算被影响的节点
        '''
        res: List[AS_CODE] = []
        for node in queue:
            for i in self.graph[node][1]:
                self.graph[i][0].remove(node)

            self.graph[node][1] = []

        while queue:
            n: AS_CODE = queue.pop(0)
            res.append(n)
            if n not in self.graph: continue

            for i in self.graph[n][0]:
                self.graph[i][1].remove(n)
                if len(self.graph[i][1]) == 0: queue.append(i)
            del self.graph[n]
        return [ii for ii in self.all_as_list if ii not in self.graph.keys()]


def monitor_cut_class2func_inter(f: str, cc2as_list_path: CC_PATH):
    monitor_cut(gl_cut_node_depth, f, f[:-4] + '.addDel.txt', f.split('/')[-1][9:-4], cc2as_list_path)


# @record_launch_time
@record_launch_time_and_param(0)
def do_cut_by_cc(cc: COUNTRY_CODE, path: RTREE_PATH, asn_data: Dict[AS_CODE, int]):
    print('country: ' + cc)
    file_name: List[str] = os.listdir(os.path.join(path, cc))
    node_num: List[Tuple[str, int]] = []
    for f in file_name:
        if f.find('.json') != -1 or f.find(
                '.txt') != -1 or f[-4:] != '.npz' or f[:3] != 'dco':
            continue
        if f[9:-4] not in asn_data:
            continue
        node_num.append((f, asn_data[f[9:-4]]))
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for f in gl_get_destroy_trees(node_num):
        try:
            thread_pool.apply_async(monitor_cut_class2func_inter,
                                    (os.path.join(path, cc, f[0]), os.path.join(gl_cc2as_path, '%s.json' % cc)))
        except Exception as e:
            print(e)
            raise e
    thread_pool.close()
    thread_pool.join()


@record_launch_time
def monitor_country_internal(prefix: OUTPUT_PATH, _type: TOPO_TPYE, asn_data: Dict[AS_CODE, int], destroy_model_path: str, cut_rtree_model_path: str,
                             cut_node_depth: int, cc_list: List[COUNTRY_CODE], cc2as_path: CC2AS_PATH):
    path: RTREE_PATH = os.path.join(prefix, _type, 'rtree/')
    global gl_get_destroy_trees
    global gl_get_cut_num
    global gl_cut_node_depth
    global gl_cc2as_path
    dynamic_module_1 = import_module(destroy_model_path)
    dynamic_module_2 = import_module(cut_rtree_model_path)

    gl_cut_node_depth = cut_node_depth + 1
    gl_get_destroy_trees = dynamic_module_1.get_destroy_trees
    gl_get_cut_num = dynamic_module_2.get_cut_num
    gl_cc2as_path = cc2as_path

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_list:
        do_cut_by_cc(
            cc,
            path,
            asn_data,
        )
        # pool.apply_async(do_cut_by_cc, (
        #     cc,
        #     path,
        #     asn_data,
        # ))
    pool.close()
    pool.join()
