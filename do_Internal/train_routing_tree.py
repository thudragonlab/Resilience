#!usr/bin/env python
# _*_ coding:utf8 _*_
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
import os
import random
from scipy import stats
import numpy as np
import time
from do_Internal.sort_rank import groud_truth_based_anova_for_single_country, country_internal_rank, internal_survival
from multiprocessing import Pool
from util import mkdir, record_launch_time, record_launch_time_and_param

# from ..base.internal.internal_security import *
# from base.internal.internal_security import *
# from ..internal_security import extract_connect_list, groud_truth_based_anova
'''
3、量化并排序 (internal_security.py)
'''

SUFFIX = 'optimize_link'
RESULT_SUFFIX = SUFFIX + '/new_optimize_result'
SORT_DSN_PATH_SUFFIX = RESULT_SUFFIX + '/anova'


def add_link_to_npz(add_link_file, old_npz_file, relas_file, dsn_npz_file,
                    add_link_relas_file, add_link_num):
    if os.path.exists(dsn_npz_file):
        print(dsn_npz_file + ' exist')
        return

    state = {'c2p': 0, 'p2p': 1, 'p2c': 2, 0: 'c2p', 1: 'p2p', 2: 'p2c'}
    match_state = {'1': {'1': 'p2p', '2': 'c2p'}, '2': {'1': 'p2c'}}

    graph = {}
    with open(relas_file) as fp:
        line = fp.readline().strip()
        while line:
            if line[0].isdigit():
                line = line.split('|')
                if line[0] not in graph:
                    graph[line[0]] = {'p2p': [], 'p2c': [], 'c2p': []}
                if line[1] not in graph:
                    graph[line[1]] = {'p2p': [], 'p2c': [], 'c2p': []}
                if line[-1][-1] == '1':
                    graph[line[0]]['p2c'].append(line[1])
                    graph[line[1]]['c2p'].append(line[0])
                else:
                    graph[line[0]]['p2p'].append(line[1])
                    graph[line[1]]['p2p'].append(line[0])
            line = fp.readline().strip()

    npz_file = np.load(old_npz_file)
    row = [str(i) for i in npz_file['row']]
    col = [str(i) for i in npz_file['col']]
    link = list(zip(row, col))
    cur_graph = {}
    for line in link:
        if line[0] == line[1]:
            print(1)
        if line[0] not in cur_graph:
            cur_graph[line[0]] = {'pre': [], 'nxt': []}
        if line[1] not in cur_graph:
            cur_graph[line[1]] = {'pre': [], 'nxt': []}
        cur_graph[line[0]]['pre'].append(line[1])
        cur_graph[line[1]]['nxt'].append(line[0])

    with open(add_link_file, 'r') as f:
        m = json.load(f)
    add_link = []
    if isinstance(m, dict):
        for line in m:
            if len(line) < 2:
                continue
            line = line.split(' ')
            line[0] = line[0][1:-1]
            line[1] = line[1][:-1]
            begin_state = line[2][2:-2]
            end_state = line[3][1:-2]
            add_link.append([
                str(line[0]),
                str(line[1]), state[match_state[begin_state][end_state]]
            ])
            if len(add_link) > add_link_num:
                break
    elif isinstance(m, list):
        for line in m:
            add_link.append([
                str(line[0]),
                str(line[1]), state[match_state[begin_state][end_state]]
            ])
            if len(add_link) > add_link_num:
                break

    def find_pre_state(node):
        if node not in graph or node not in cur_graph:
            return False
        cur_state = 2
        if len(cur_graph[node]['pre']) == 0:
            return 0
        for _node in cur_graph[node]['pre']:
            min_state = cur_state
            for i in range(cur_state - 1, -1, -1):
                if node in graph[_node][state[i]]:
                    min_state = min(i, cur_state)
                    if min_state == 0:
                        return 0
                    break
            cur_state = min_state
        return cur_state

    add_link_relas = {}
    n = 0
    begin_n = len(add_link)
    # print('!!!!')
    print(add_link)
    while add_link:
        link = add_link.pop(0)
        if link[0] == link[1]:
            continue
        if len(link) == 3:
            s = link[2]
        else:
            s0 = find_pre_state(link[0])
            s1 = find_pre_state(link[1])
            if not s0 or not s1: continue
            if s0 > s1: link[0], link[1] = link[1], link[0]
            s = min(s0, s1)
        add_link_relas[str(link[0]) + ' ' + str(link[1])] = state[s]
        if link[1] not in cur_graph or link[0] not in cur_graph:
            continue
        cur_graph[link[0]]['nxt'].append(link[1])
        try:
            cur_graph[link[1]]['pre'].append(link[0])
        except:
            print(link[1] not in cur_graph)
            print(cur_graph[link[1]])
            exit()

        n += 1
        for _s in range(s, 3):
            for _node in graph[link[1]][state[_s]]:
                if _node not in cur_graph[link[1]]['pre'] and _node not in cur_graph[link[1]]['nxt']\
                        and _node in cur_graph:
                    add_link.append([link[1], _node, _s])

    print('add link num:', begin_n, n)
    row, col = [], []
    for node in cur_graph:
        for _node in cur_graph[node]['pre']:
            row.append(node)
            col.append(_node)
    with open(add_link_relas_file, 'w') as f:
        json.dump(add_link_relas, f)
    np.savez(dsn_npz_file, row=row, col=col)


class monitor_cut():
    # 3、读取旧的.addDel文件 计算新routingTree下模拟的结果 （37服务器 routingTree.py）

    def __init__(self, file_path, old_del_path, dsn_path):
        self.file_name = file_path
        self.graph = {}
        self.dsn_path = dsn_path
        self.old_del_path = old_del_path
        self.tempgraphname = file_path + '.graph.json'

        # 存储结果：{[]:节点总数量，[queue]:节点数量}
        self.res = {}

        # 创建图
        self.from_npz_create_graph()
        print(self.file_name + ' graph created')
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
            self.graph[a][1].append(b)
            self.graph[b][0].append(a)
        self.res[''] = len(self.graph)

    def monitor_node_addDel(self):
        with open(self.dsn_path, "a+") as dsn_f:
            with open(self.old_del_path, 'r') as fp:
                line = fp.readline()
                while line:
                    if line[0].isdigit():
                        queue = line.split('|')[0].split(' ')
                        oldbreak = line.split('|')[1].split(' ')
                        linkres = self.monitor_cut_node(queue)

                        if len(linkres) > len(oldbreak):
                            print(self.tempgraphname, line, linkres,
                                  'bad error\n')

                        # elif len(linkres)<len(oldbreak):
                        #    print(self.tempgraphname, line, linkres, 'good error\n')

                        dsn_f.write(
                            line.split('|')[0] + '|' +
                            ' '.join(list(map(str, linkres))) + '\n')
                        with open(self.tempgraphname, 'r') as graph_f:
                            self.graph = json.load(graph_f)
                    line = fp.readline()

    def monitor_cut_node(self, queue):
        res = []
        for node in queue:
            for i in self.graph[node][1]:
                self.graph[i][0].remove(node)

            self.graph[node][1] = []

        while queue:
            n = queue.pop(0)
            res.append(n)
            if n not in self.graph:
                continue

            for i in self.graph[n][0]:
                self.graph[i][1].remove(n)
                if len(self.graph[i][1]) == 0:
                    queue.append(i)
            del self.graph[n]
        return res


class iSecutiry():

    def __init__(self, path, connect_dsn_path, sort_dsn_path) -> None:
        # connect_dsn_path = os.path.join(result_dsn_path, 'count_num/')
        # sort_dsn_path = os.path.join(result_dsn_path, 'anova/')
        # rank_dsn_path = os.path.join(result_dsn_path, 'rank/')

        self.path = path
        self.connect_dsn_path = connect_dsn_path
        self.sort_dsn_path = sort_dsn_path

    def extract_connect_list(self, begin_num=1):

        def basic_user_domain(line):
            as_list = line.split(' ')
            res1 = 0
            res2 = 0
            for _as in as_list:
                if _as in as_importance:
                    res1 += as_importance[_as][0]
                    res2 += as_importance[_as][1]
            return [line.count(' ') + 1, res1, res2]

        if not os.path.exists(self.path):
            print('%s  not exist' % self.path)
            return
        if begin_num < 1: begin_num = 1
        cc_name = os.listdir(self.path)
        for cc in cc_name:
            if cc + '.json' not in os.listdir(as_importance_path):
                print(cc + ' not have as_importance')
                continue
            with open(os.path.join(as_importance_path, cc + '.json'),
                      'r') as f:
                _as_importance = json.load(f)
            as_importance = {}
            for line in _as_importance:
                as_importance[line[0]] = line[1:]

            # for num in range(begin_num, Num):
            for num in Num_list:
                file_path = os.path.join(self.path, cc, 'all', str(num))
                if not os.path.exists(file_path):
                    print(file_path + ' del file not exist')
                    continue
                try:
                    if not os.path.exists(
                            os.path.join(self.connect_dsn_path, cc)):
                        os.makedirs(os.path.join(self.connect_dsn_path, cc))
                    if not os.path.exists(
                            os.path.join(self.connect_dsn_path, cc, str(num))):
                        os.makedirs(
                            os.path.join(self.connect_dsn_path, cc, str(num)))
                except:
                    time.sleep(5)
                file_name = os.listdir(file_path)
                file_name = [
                    i for i in os.listdir(file_path) if i[-4:] == '.txt'
                    and i.find('as-rel') == -1 and i[0] != '.'
                ]
                # file_name = [i for i in os.listdir(file_path) if i[-4:]=='.txt' and i.find('as-rel')==-1 and i[0]!='.' and i.find('.'+str(num)+'.')!=-1]
                if not len(file_name):
                    print(num)
                    break
                
                for file in file_name:
                    res = {}
                    asname = file.split('.')[0]
                    if os.path.exists(
                            os.path.join(self.connect_dsn_path, cc, str(num),
                                         asname + '.json')):
                        print('connect file exist')
                        continue
                    res[asname] = {}
                    res[asname]['asNum'] = -1
                    res[asname]['connect'] = [[], [], [], []]
                    with open(os.path.join(file_path, file), 'r') as f:
                        for line in f:
                            l = line.split('|')
                            if line[0] == '#':
                                res[asname]['asNum'] = int(l[1])
                            elif len(l) > 1 and line[0][0] != '(' and l[
                                    0] != '':
                                l1 = l[0].count(' ')
                                if l1 < len(res[asname]['connect']):
                                    l2 = basic_user_domain(l[1])
                                    res[asname]['connect'][l1].append(l2)
                    with open(
                            os.path.join(self.connect_dsn_path, cc, str(num),
                                         asname + '.json'), 'w') as df:
                        json.dump(res, df)


def cal_anova_change_for_single_country(connect_dsn_path, old_connect_path,
                                        num, _cc, m, output_path):
    '''
    4、计算某个国家优化后的排名
    '''

    if not os.path.exists(os.path.join(connect_dsn_path, _cc, str(num))):
        print(os.path.join(connect_dsn_path, _cc, str(num)) + ' not exist')
        return
    if len(os.listdir(os.path.join(connect_dsn_path, _cc, str(num)))) == 0:
        print(os.path.join(connect_dsn_path, _cc, str(num)) + ' file is 0')
        return

    for value in ['basic', 'user', 'domain']:
        new_anova_path = os.path.join(output_path, m, SORT_DSN_PATH_SUFFIX,
                                      value + '_' + _cc)
        mkdir(new_anova_path)
        groud_truth_based_anova_for_single_country(
            os.path.join(connect_dsn_path, _cc, str(num)), _cc,
            old_connect_path, new_anova_path, value, num)


@record_launch_time
def record_result(topo_list, output_path, type_path, _type):
    '''
    5、记录排名的变化
    '''
    middle_index_dict = {
        'asRank': [0, 2],
        'problink': [3, 5],
        'toposcope': [6, 8],
        'toposcope_hidden': [9, 11]
    }

    change_res = {}
    for _cc in cc_list:
        print(_cc)
        if _cc in ['BR', 'US', 'RU']:
            continue
        change_res[_cc] = {}
        # for num in range(1, Num):
        for num in Num_list:
            change_res[_cc][str(num)] = {}
            country_internal_rank(cc_list, topo_list, output_path,
                                  RESULT_SUFFIX, type_path, _type, _cc,
                                  str(num))

    #         for m in topo_list:
    #             result_dsn_path = os.path.join(output_path,'public')
    #             rank_dsn_path = os.path.join(result_dsn_path, 'optimize/')
    #             mkdir(rank_dsn_path)
    #             for value in ['basic', 'user', 'domain']:

    #                 if not os.path.exists(os.path.join(output_path, m, RESULT_SUFFIX,type_path, value+'_'+_cc, 'sorted_country_'+value+'.'+str(num)+'.json')):
    #                     print(m, _cc, '%s/sorted_country_%s'%(type_path,value) + '.'+str(num)+'.json', ' not exist')

    #             country_internal_rank(cc_list, topo_list,output_path,RESULT_SUFFIX, type_path,_type,_cc,num)

    #             ccres = internal_survival(os.path.join(
    #                 rank_dsn_path, '%s_rank.%s.%s.json' % (_type,_cc,str(num))), 0, 2)

    #             ccres_old = internal_survival(
    #                 old_rank_file, middle_index_dict[m][0], middle_index_dict[m][1])
    #             ccres = {line[0]: line[1:] for line in ccres}
    #             ccres_old = {line[0]: line[1:] for line in ccres_old}
    #             change_res[_cc][str(num)][m] = {
    #                 'new': ccres[_cc], 'old': ccres_old[_cc]}
    # optimize_dst_path = os.path.join(output_path, 'public', SUFFIX)
    # mkdir(optimize_dst_path)

    # with open(os.path.join(optimize_dst_path, '%s_change_res.json' % _type), 'w') as f:
    #     json.dump(change_res, f)


@record_launch_time_and_param(2, 1)
def add_npz_and_monitor_cut_pool(output_path, m, cname):
    dst_path = os.path.join(output_path, m)
    new_path = os.path.join(dst_path, SUFFIX, 'new_optimize')
    floyed_path = os.path.join(dst_path, SUFFIX, 'floyed')
    rtree_path = os.path.join(dst_path, 'rtree/')
    mkdir(new_path)
    mkdir(os.path.join(new_path, cname))
    mkdir(os.path.join(new_path, cname, 'rtree'))
    mkdir(os.path.join(new_path, cname, 'all'))

    relas_file = os.path.join(rtree_path, cname, 'as-rel.txt')
    add_link_file = os.path.join(floyed_path,
                                 cname + '.opt_add_link_rich.json')

    def add_npz_and_monitor_cut_thread(file, _add_link_num):
        old_npz_file = os.path.join(rtree_path, cname,
                                    file.split('.')[0] + '.npz')
        new_npz_path = os.path.join(new_path, cname, 'rtree',
                                    str(_add_link_num) + '/')
        new_npz_file = os.path.join(new_npz_path, file.split('.')[0] + '.npz')
        add_link_relas_file = os.path.join(
            new_path, cname, 'rtree',
            file.split('.')[0] + '.add_link_relas.json')
        old_del_path = os.path.join(rtree_path, cname, file)
        new_del_path = os.path.join(new_path, cname, 'all',
                                    str(_add_link_num) + '/')
        new_del_file = os.path.join(new_del_path, file)
        mkdir(new_npz_path)
        mkdir(new_del_path)

        if os.path.exists(old_npz_file) and not os.path.exists(new_del_file):
            add_link_to_npz(add_link_file, old_npz_file, relas_file,
                            new_npz_file, add_link_relas_file, _add_link_num)
            monitor_cut(new_npz_file, old_del_path, new_del_file)
        elif not os.path.exists(old_npz_file):
            print(old_npz_file + " not exist")
        else:
            print(new_del_file + " exist")

    thread_pool_inner = ThreadPool(multiprocessing.cpu_count() * 10)
    for file in os.listdir(os.path.join(rtree_path, cname)):
        if file.find('addDel') == -1:
            continue
        # for _add_link_num in range(1, Num):
        for _add_link_num in Num_list:
            thread_pool_inner.apply_async(add_npz_and_monitor_cut_thread, (
                file,
                _add_link_num,
            ))
    thread_pool_inner.close()
    thread_pool_inner.join()


@record_launch_time_and_param(1, 0, 2)
def cal_anova_for_single_cc_pool(m, _cc, num, output_path):
    connect_dsn_path = os.path.join(output_path, m, SUFFIX,
                                    'new_optimize_result', 'count_num')
    old_connect_path = os.path.join(output_path, m, 'result/count_num/')
    cal_anova_change_for_single_country(connect_dsn_path, old_connect_path,
                                        num, _cc, m, output_path)


@record_launch_time_and_param(1, 0, 2)
def cal_var_for_single_cc_pool(m, _cc, num, output_path):
    old_count_num_path = os.path.join(output_path, m, 'result/count_num/')
    new_count_num_path = os.path.join(output_path, m, SUFFIX,
                                      'new_optimize_result', 'count_num', _cc,
                                      str(num))
    for value in ['basic', 'user', 'domain']:
        new_var_path = os.path.join(output_path, m, RESULT_SUFFIX, 'var',
                                    value + '_' + _cc)
        mkdir(new_var_path)
        cal_var_change_for_single_country(new_count_num_path,
                                          old_count_num_path, value, _cc,
                                          new_var_path, num)


def judge_var(target_list, result):
    if len(target_list) == 0:
        return
    source_list = target_list[0]
    target_list.remove(source_list)
    result.append([])
    result[-1].append(source_list['key'])
    if len(target_list) == 0:
        return
    for ii in target_list:
        F = np.var(source_list['list']) / np.var(ii['list'])
        # p = 1 - stats.f.cdf(F, 60, 60)
        stat, p = stats.levene(source_list['list'], ii['list'])
        if ii['key'] == source_list['key']:
            continue
        if p > 0.05:
            result[-1].append(ii['key'])
            target_list.remove(ii)
    judge_var(target_list, result)


def cal_var_change_for_single_country(new_count_num_path, old_count_num_path,
                                      _type, single_country_name, new_var_path,
                                      num):

    var_result = {}
    result = []
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)

    def groud_truth_based_var_old_thread(cc):

        def basic_value_map(x):
            if _type == 'basic':
                return x[value_dict[_type]] / N
            else:
                return x[value_dict[_type]]

        if cc == single_country_name: return
        if cc.find('.json') != -1 or cc[0] == '.': return
        if cc[-4:] == 'json':
            return
        as_list = os.listdir(os.path.join(old_count_num_path, cc))
        for as_file_name in as_list:
            if as_file_name.find('.json') == -1 or as_file_name[0] == '.':
                continue
            with open(os.path.join(old_count_num_path, cc, as_file_name),
                      'r') as as_file:
                as_data = json.load(as_file)
                for _as in as_data:
                    N = as_data[_as]['asNum']
                    if N < 0:
                        continue
                    if N < 20:
                        continue
                    for i in as_data[_as]['connect']:
                        if len(i) == 0:
                            continue
                        var_result['%s-%s' % (cc, as_file_name[:-5])] = {
                            'list': list(map(basic_value_map, i)),
                            'key': '%s-%s' % (cc, as_file_name[:-5])
                        }

    def groud_truth_based_var_new_thread(file):

        def basic_value_map(x):
            if _type == 'basic':
                return x[value_dict[_type]] / N
            else:
                return x[value_dict[_type]]

        with open(os.path.join(new_count_num_path, file), 'r') as as_file:
            as_data = json.load(as_file)
            for _as in as_data:
                N = as_data[_as]['asNum']
                if N < 0:
                    continue
                if N < 20:
                    continue
                for i in as_data[_as]['connect']:
                    if len(i) == 0:
                        continue
                    var_result['%s-%s' % (single_country_name, file[:-5])] = {
                        'list': list(map(basic_value_map, i)),
                        'key': '%s-%s' % (single_country_name, file[:-5])
                    }

    cc_list = os.listdir(old_count_num_path)
    file_list = os.listdir(new_count_num_path)
    for cc in cc_list:
        thread_pool.apply_async(groud_truth_based_var_old_thread, (cc, ))
    for file in file_list:
        thread_pool.apply_async(groud_truth_based_var_new_thread, (file, ))
    thread_pool.close()
    thread_pool.join()
    var_list = list(var_result.values())
    var_list.sort(key=lambda x: np.var(x['list']))

    judge_var(var_list, result)

    with open(
            os.path.join(new_var_path,
                         'sorted_country_%s.%s.json' % (_type, str(num))),
            'w') as sorted_var_f:
        json.dump(result, sorted_var_f)





@record_launch_time
def part1(topo_list, _cc_list, output_path, _as_importance_path):
    pool = Pool(multiprocessing.cpu_count())
    for m in topo_list:
        for cname in cc_list:
            path = os.path.join(output_path, m, 'rtree/')
            floyed_path = os.path.join(output_path, m, SUFFIX, 'floyed/')
            if not os.path.exists(os.path.join(path, cname, 'as-rel.txt')):
                print(cname + ' 没有as-rel')
                continue
            if not os.path.exists(
                    os.path.join(floyed_path,
                                 cname + '.opt_add_link_rich.json')):
                print(cname + ' 没有opt_add_link_rich')
                continue
            # input.append([os.path.join(output_path,m),cname])
            pool.apply_async(add_npz_and_monitor_cut_pool, (
                output_path,
                m,
                cname,
            ))
    pool.close()
    pool.join()


@record_launch_time
def part2(output_path, topo_list):
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)

    def make_dir_thread(m):
        new_path = os.path.join(output_path, m, SUFFIX, 'new_optimize')
        result_dsn_path = os.path.join(output_path, m, SUFFIX,
                                       'new_optimize_result')
        connect_dsn_path = os.path.join(result_dsn_path, 'count_num')
        sort_dsn_path = os.path.join(result_dsn_path, 'anova')

        mkdir(result_dsn_path)
        mkdir(connect_dsn_path)
        mkdir(sort_dsn_path)

        iS = iSecutiry(new_path, connect_dsn_path, sort_dsn_path)
        iS.extract_connect_list()

    for m in topo_list:
        thread_pool.apply_async(make_dir_thread, (m, ))
    thread_pool.close()
    thread_pool.join()


@record_launch_time
def part3(topo_list, output_path):

    pool = Pool(multiprocessing.cpu_count())
    for _cc in cc_list:
        if _cc in ['BR', 'US', 'RU']: continue
        for m in topo_list:
            # print(m)
            connect_dsn_path = os.path.join(output_path, m, SUFFIX,
                                            'new_optimize_result', 'count_num')
            # for num in range(1, Num):
            for num in Num_list:
                if not os.path.exists(
                        os.path.join(connect_dsn_path, _cc, str(num))):
                    print(
                        os.path.join(connect_dsn_path, _cc, str(num)) +
                        ' not exist')
                    continue
                if len(
                        os.listdir(
                            os.path.join(connect_dsn_path, _cc,
                                         str(num)))) == 0:
                    print(
                        os.path.join(connect_dsn_path, _cc, str(num)) +
                        ' file is 0')
                    continue

                pool.apply_async(cal_anova_for_single_cc_pool, (
                    m,
                    _cc,
                    str(num),
                    output_path,
                ))
                pool.apply_async(cal_var_for_single_cc_pool, (
                    m,
                    _cc,
                    str(num),
                    output_path,
                ))
    pool.close()
    pool.join()




# Num = 6
cc_list = []
as_importance_path = None
old_rank_file = None

@record_launch_time
def train_routing_tree(topo_list, _cc_list, output_path, _as_importance_path,optimize_link_list):
    global as_importance_path
    global cc_list
    global old_rank_file
    global Num
    global Num_list
    Num = 6
    Num_list = optimize_link_list
    cc_list = _cc_list
    as_importance_path = _as_importance_path
    old_rank_file = os.path.join(output_path, 'public', 'med_rank.json')
    part1(topo_list,_cc_list,output_path,_as_importance_path)
    part2(output_path,topo_list)
    part3(topo_list, output_path)
    # print(topo_list)
    record_result(topo_list, output_path, 'anova', 'med')
    record_result(topo_list, output_path, 'var', 'var')
