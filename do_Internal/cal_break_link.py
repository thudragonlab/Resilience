#!usr/bin/env python
# _*_ coding:utf8 _*_
from functools import wraps
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import json
import numpy as np

class monitor_cut():
    def __init__(self, file_path):
        self.file_name = file_path
        self.graph = {}
        self.tempgraphname = file_path + '.graph.json'

        #存储结果：{[]:节点总数量，[queue]:节点数量}
        self.res = {}
        self.prefix = {}
        self.nodeNum = {}

        #创建图
        self.from_npz_create_graph()



    def from_npz_create_graph(self):
        '''
        存routingTree 【【前向点】【后向点】】 后向点为空说明就脱离了routingTree
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
        self.res[''] = len(self.graph)



    def monitor_cut_node(self, queue):
        res = []
        for node in queue:
            if node in self.graph:
                for i in self.graph[node][1]:
                    self.graph[i][0].remove(node)
                self.graph[node][1] = []

        while queue:
            n = queue.pop(0)
            if n not in self.graph: continue
            res.append(n)
            for i in self.graph[n][0]:
                if n in self.graph[i][1]:
                    self.graph[i][1].remove(n)
                if len(self.graph[i][1])==0: queue.append(i)
            del self.graph[n]
        return res


class monitor_break():

    def __init__(self) -> None:
        self.sample_result = []

    def main_2(self, rtree_path, dsn_path):
        del_file = [file for file in os.listdir(rtree_path) if file.find('addDel')!=-1]
        
        '''
        sample_result
        找每个路由树里面破坏影响节点最多的链接
        数组下标是破坏节点数量
        [
            {
                具体破坏的节点:受影响的节点数量max
            }
        ]
        '''
        for df in del_file:
            with open(os.path.join(rtree_path, df)) as fp:
                line = fp.readline().strip()
                while line:
                    if line[0]!='#':
                        break_node = line.split('|')[0]
                        break_node_num = len(break_node.split(' '))
                        if break_node_num>len(self.sample_result):
                            self.sample_result.append({})
                        if break_node not in self.sample_result[break_node_num-1]:
                            self.sample_result[break_node_num-1][break_node] = 0
                        self.sample_result[break_node_num-1][break_node] = max(\
                            self.sample_result[break_node_num-1][break_node],\
                                len(line.split('|')[1].split(' ')))
                    line = fp.readline().strip()
        
        '''
        按破坏受影响的数量倒序,取TOP ${sample_num}
        '''
        for i in range(len(self.sample_result)):
            self.sample_result[i] = sorted(
                self.sample_result[i].items(), key=lambda d: d[1], reverse=True)
            self.sample_result[i] = self.sample_result[i][:100]
        rtree_file = [file for file in os.listdir(rtree_path) if file[-3:]=='npz' and file[0]=='d']
        

        '''
        从路由树里面剪掉sample_result里面的节点,被剪掉的节点存于.break_link.json
        '''

        
        
        break_result = {}
        pool = multiprocessing.Pool(processes=len(self.sample_result))
        results = []
        for i in range(len(self.sample_result)):
            
            result = pool.apply_async(func=self.cal_break_link,args=(i,rtree_file,rtree_path,))
            results.append(result)
        pool.close()
        pool.join()
        for i in results:
            real_result = i.get()
            for k in real_result:
                print(real_result)
                break_result[k] = real_result[k]
        print(dsn_path+'.break_link.json    created')
        with open(dsn_path+'.break_link.json', 'w') as f:
            json.dump(break_result, f)
    
        return break_result

    def cal_break_link(self,index,rtree_file,rtree_path):
        break_result = {}
        def cut_as(rf1,break_link_inner):
            _as1 = rf1.split('.')[0]
            if _as1[0]=='d': _as1 = _as1[9:]
            mc = monitor_cut(os.path.join(rtree_path, rf1))
            res = mc.monitor_cut_node(break_link_list)
            for _as2 in res:
                if [_as2, _as1] not in break_result[break_link_inner]:
                    break_result[break_link_inner].append([_as1, _as2])
            print('finish %s %s' % (rf1,break_link_inner))

        for break_link in self.sample_result[index]:
            break_link = break_link[0]
            break_result[break_link] = []
            break_link_list = break_link.split(' ')
            thread_pool = ThreadPool(processes=multiprocessing.cpu_count() * 10)
            for rf in rtree_file:
                try:
                    thread_pool.apply_async(func=cut_as,args=(rf,break_link,))
                except Exception as e:
                    print(e)
                    raise e
            thread_pool.close()
            thread_pool.join()
            
            break_result[break_link] = list(break_result[break_link])
        return break_result
        