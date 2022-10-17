#!usr/bin/env python
# _*_ coding:utf8 _*_
import os
import json
import csv
from math import floor
from sklearn import linear_model
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from pyecharts import options as opts
from pyecharts.charts import Page, Tree, Bar, Line
from util import mkdir, record_launch_time, record_launch_time_and_param
from pyecharts.globals import WarningType
import multiprocessing
from multiprocessing.pool import ThreadPool

WarningType.ShowWarning = False
'''
10.25
提取.addDel.txt文件的连通性数据，获得各个国家的连通性数据的文件
格式：
key:AS号, value:[[],[],[]], asNum:AS的数量
value索引为i存储破坏i个节点的所有连通性结果
'''
# old_rank_file = '/home/peizd01/for_dragon/public/rank_2.json'
as_importance_path = ''


def extract_connect_list_async(
    cc,
    path,
    dsn_path,
):

    def basic_user_domain(line):
        as_list = line.split(' ')
        res1 = 0
        res2 = 0
        for _as in as_list:
            if _as in as_importance:
                # print(as_importance)
                res1 += as_importance[_as][0]
                res2 += as_importance[_as][1]
        return [line.count(' ') + 1, res1, res2]

    if cc + '.json' not in os.listdir(as_importance_path):
        print(cc + ' not have as_importance')
        return
    with open(os.path.join(as_importance_path, cc + '.json'), 'r') as f:
        _as_importance = json.load(f)
    as_importance = {}
    for line in _as_importance:
        as_importance[line[0]] = line[1:]
    file_name = os.listdir(os.path.join(path, cc))
    file_name = [i for i in file_name if i.find('.addDel') != -1]

    if not os.path.exists(os.path.join(dsn_path, cc)):
        os.makedirs(os.path.join(dsn_path, cc))
    for file in file_name:
        res = {}
        asname = file.split('.')[0]
        res[asname] = {}
        res[asname]['asNum'] = -1
        res[asname]['connect'] = [[], [], [], []]
        f = open(os.path.join(path, cc, file), 'r')
        for line in f:
            l = line.split('|')
            if line[0] == '#':
                res[asname]['asNum'] = int(l[1])
            elif len(l) > 1 and line[0][0] != '(' and l[0] != '':
                l1 = l[0].count(' ')
                if l1 < len(res[asname]['connect']):
                    l2 = basic_user_domain(l[1])
                    res[asname]['connect'][l1].append(l2)
        with open(os.path.join(dsn_path, cc, asname + '.json'), 'w') as df:
            json.dump(res, df)


def extract_connect_list(path, dsn_path):
    cc_name = os.listdir(path)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_name:
        pool.apply_async(extract_connect_list_async, (
            cc,
            path,
            dsn_path,
        ))
    pool.close()
    pool.join()


def anova_sort(dsn_path, VALUE, debug_path, reader):
    # if os.path.exists(os.path.join(dsn_path, 'sorted_country_'+VALUE+'.json')): return
    # with open(os.path.join(dsn_path, 'anova_' + VALUE + '_multi_comparison.json'), 'r') as f:
    #     reader = json.load(f)
    res = {}  # 记录{国家:【比其更安全的国家】【无差异国家】【更不安全的国家】}
    for line in reader:
        if line[0] not in res: res[line[0]] = [[], [], []]
        if line[1] not in res: res[line[1]] = [[], [], []]
        if line[-1] == 0:
            res[line[0]][1].append(line[1])
            res[line[1]][1].append(line[0])
        elif line[-1] < 0:
            res[line[0]][0].append(line[1])
            res[line[1]][2].append(line[0])
        else:
            res[line[0]][2].append(line[1])
            res[line[1]][0].append(line[0])

    sorted_country = [[]]
    for k in res:
        if len(res[k][0]) == 0:
            sorted_country[-1].append(k)

    while len(res):
        for cc in sorted_country[-1]:
            for _cc in res[cc][2]:
                try:
                    res[_cc][0].remove(cc)
                except:
                    print(_cc)
                    print(cc)
                    exit()
            for _cc in res[cc][1]:
                res[_cc][1].remove(cc)
            del res[cc]
        temp = []
        for cc in res:
            if len(res[cc][0]) == 0:
                temp.append(cc)
        sorted_country.append(temp)

    with open(os.path.join(dsn_path, 'sorted_country_' + VALUE + '.json'), 'w') as f:
        json.dump(sorted_country, f)

    if debug_path:
        with open(os.path.join(debug_path, 'sorted_country_' + VALUE + '.json'), 'w') as f:
            json.dump(sorted_country, f)


def anova(dict_l, dsn_path, VALUE, debug_path):
    '''
    输入L:[[],...,[]]，列表每个元素为国家所有的破坏性度量
    '''
    # if os.path.exists(os.path.join(dsn_path, 'anova_' + VALUE + '_multi_comparison.json')):
    #     return
    l = [v for _, v in dict_l.items()]
    print(l)
    f, p = stats.f_oneway(*l)
    print('\nOne-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p)
    if p > 0.05:
        print('无显著性差异 p>0.05')
        # return
    else:
        print('有显著性差异')

    nums, groups = [], []
    for k, v in dict_l.items():
        nums += v
        groups += len(v) * [k]
    mc = MultiComparison(nums, groups)
    result = mc.tukeyhsd()

    res = []
    for line in result._results_table.data[1:]:
        # print('line => %s \n '% line )
        if line[-1]:
            res.append([line[0], line[1], line[2]])
        else:
            res.append([line[0], line[1], 0])

    # with open(os.path.join(dsn_path, 'anova_' + VALUE + '_multi_comparison.json'), 'w') as f:
    #     json.dump(res, f)
    anova_sort(dsn_path, VALUE, debug_path, res)


@record_launch_time_and_param(2)
def groud_truth_based_anova(path, dsn_path, value, debug_path):

    def groud_truth_based_anova_thread(_cc):
        if _cc.find('.json') != -1 or _cc[0] == '.': return
        cc_name = os.listdir(os.path.join(path, _cc))
        for file in cc_name:
            _l = []
            try:
                with open(os.path.join(path, _cc, file), 'r') as f:
                    r = json.load(f)
            except Exception as e:
                print(file)
            
            for _as in r:
                N = r[_as]['asNum']
                if N < 0: continue
                
                for i in r[_as]['connect']:
                    if value == 'basic':
                        _l += [_i[value_dict[value]] / N for _i in i]
                    else:
                        _l += [_i[value_dict[value]] for _i in i]
            asname = file.split('.')[0]
            if len(_l) > 0:
                l[_cc + '-' + asname] = _l

    # file_name = os.listdir(path)
    '''
    l 的数据结构
    {
        AD_dcomplete{as_code}:[connect[0][受影响的节点数量]/asNum,....connect[100][受影响的节点数量]/asNum]
    }
    '''
    l = {}
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for _cc in cc_list:
        thread_pool.apply(groud_truth_based_anova_thread, (_cc, ))
    thread_pool.close()
    thread_pool.join()
    # print(l[list(l.keys())[0]])
    anova(l, dsn_path, value, debug_path)
    # anova_sort(dsn_path, value, debug_path)


@record_launch_time
def country_internal_rank(path, rank_path, _type, topo_list, debug_path):

    def country_internal_rank_thread(value2, value):
        del_path = os.path.join(path, value, 'rtree')
        npz_file_name = del_path
        anova_path = os.path.join(path, value, 'result', _type, 'sorted_country_' + value2 + '.json')
        with open(anova_path, 'r') as f:
            reader = json.load(f)
        res = {}
        for index in range(len(reader)):
            for _v in reader[index]:
                _cc, _as = _v.split('-')
                if _cc not in res: res[_cc] = {}
                if len(_as) == 0: continue
                if _as[0] == 'd': _as = _as[9:]
                res[_cc][_as] = index + 1  # 每个国家下每个AS的排名
        for _cc in res:
            if _cc not in rank: continue
            file_name = os.listdir(os.path.join(del_path, _cc))
            file_name = [i for i in file_name if i.find('.graph') != -1]
            if len(file_name) == 0:  # 如果没有找到生成的.graph数据
                temp = 0
                for _as in res[_cc]:
                    if _as[0] == 'd': _as = _as[9:]
                    temp += res[_cc][_as]  # temp是每个国家下所有AS排名之和
                temp = temp / len(res[_cc])  # temp 再除以这个国家下AS的数量
            else:  # 否则
                ans = {}
                for _as in res[_cc]:
                    if len(_as) == 0: continue
                    if _as[0] == 'd': _as = _as[9:]  #格式化每个国家下的AS号
                    if 'dcomplete' + _as + '.npz.graph.json' in file_name:  # 如果有graph文件
                        with open(os.path.join(del_path, _cc, 'dcomplete' + _as + '.npz.graph.json'), 'r') as f:
                            n = json.load(f)
                        ans[_as] = len(set(list(n.keys())))  # 统计每个路由树下的链接个数
                    elif 'dcomplete' + _as + '.npz' in os.path.join(npz_file_name, _cc):  #否则解析npz文件
                        m = np.load(os.path.join(npz_file_name, _cc, 'dcomplete' + _as + '.npz'))
                        ans[_as] = len(set(m['row']))  # 统计每个路由树下的链接个数
                    else:
                        ans[_as] = 0  # 否则链接数为0
                temp = 0
                for _as in res[_cc]:
                    if len(_as) == 0: continue
                    if _as[0] == 'd': _as = _as[9:]
                    if ans[_as] == 0: continue
                    # temp += res[_cc][_as] * ans[_as] #  sum(国家下所有AS的排名 * 各自链接数量)
                    temp += res[_cc][_as]  #  sum(国家下所有AS的排名 * 各自链接数量)
                    if _cc == 'KH' and value2 == 'user' and value == 'asRank':
                        print(_as, res[_cc][_as])

                    # print(temp,len(res[_cc]))
                # try:
                #     temp /= sum(list(ans.values())) # 再除以国家所有链接的个数
                # except:
                temp = sum(res[_cc].values()) / len(res[_cc])  # 如果ans是空的 每个国家的AS排名之和除以拥有的AS数量

            if _cc in rank:
                if temp < 1:
                    temp = 1
                rank[_cc].append(temp)
                # print(temp)

    rank = {}
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for _cc in cc_list:
        rank[_cc] = []
    for value2 in ['basic', 'user', 'domain']:
        for value in topo_list:
            #### 开多线程数组会乱套
            #         thread_pool.apply_async(country_internal_rank_thread, (
            #             value2,
            #             value,
            #         ))
            # thread_pool.close()
            # thread_pool.join()
            country_internal_rank_thread(value2, value)
    with open(rank_path, 'w') as f:
        json.dump(rank, f)
    if debug_path:
        with open(debug_path, 'w') as f:
            json.dump(rank, f)


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
        if ii['key'] == source_list['key']:
            continue
        if np.var(source_list['list']) == 0 and np.var(ii['list']):
            result[-1].append(ii['key'])
            target_list.remove(ii)
        else:
            stat, p = stats.levene(source_list['list'], ii['list'])
            if p > 0.05:
                result[-1].append(ii['key'])
                target_list.remove(ii)
    judge_var(target_list, result)


@record_launch_time_and_param(1)
def groud_truth_based_var(path, _type, debug_path):

    def groud_truth_based_var_thread(cc):

        def basic_value_map(x):
            if _type == 'basic':
                return x[value_dict[_type]] / N
            else:
                return x[value_dict[_type]]

        if cc[-4:] == 'json':
            return
        as_list = os.listdir(os.path.join(count_num_path, cc))
        if len(as_list) != 0:
            for as_file_name in as_list:
                with open(os.path.join(count_num_path, cc, as_file_name), 'r') as as_file:
                    as_data = json.load(as_file)
                    for _as in as_data:
                        N = as_data[_as]['asNum']
                        if N < 0:
                            continue
                        # if N < 20:
                        #     continue
                        for i in as_data[_as]['connect']:
                            if len(i) == 0:
                                continue
                            var_result['%s-%s' % (cc, as_file_name[:-5])] = {
                                'list': list(map(basic_value_map, i)),
                                'key': '%s-%s' % (cc, as_file_name[:-5])
                            }

    var_result = {}
    result = []
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    count_num_path = os.path.join(path, 'count_num')
    cc_list = os.listdir(count_num_path)

    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for cc in cc_list:
        thread_pool.apply_async(groud_truth_based_var_thread, (cc, ))
    thread_pool.close()
    thread_pool.join()
    var_list = list(var_result.values())
    var_list.sort(key=lambda x: np.var(x['list']))

    judge_var(var_list, result)

    var_path = os.path.join(path, 'var')
    if not os.path.exists(var_path):
        os.mkdir(var_path)

    with open(os.path.join(var_path, 'sorted_country_%s.json' % _type), 'w') as sorted_var_f:
        json.dump(result, sorted_var_f)

    if debug_path:
        with open(os.path.join(debug_path, 'sorted_country_%s.json' % _type), 'w') as sorted_var_f:
            json.dump(result, sorted_var_f)


# 需要定义rtree_path、count_path、as_importance_path
# rtree_path和create_routingtree.py中p2的含义相当
# count_path为输出路径
# as_importance_path为区域AS的资源权重

# prefix = '/home/peizd01/for_dragon/new_data/'


# for rela in ['asRank', 'problink', 'toposcope', 'toposcope_hidden']:
# for rela in ['toposcope', 'toposcope_hidden']:
@record_launch_time
def do_extract_connect_list(prefix, rela, weight_data_path):
    rtree_path = os.path.join(prefix, rela + '/rtree/')
    count_path = os.path.join(prefix, rela + '/result/count_num')
    global as_importance_path
    as_importance_path = weight_data_path
    extract_connect_list(rtree_path, count_path)


# 需要定义count_path、anova_path
# count_path和本文件extract_connect_list函数中count_path含义相当
# anova_path为输出路径
# for rela in ['asRank', 'problink', 'toposcope', 'toposcope_hidden']:
# for rela in ['toposcope', 'toposcope_hidden']:
@record_launch_time
def do_groud_truth_based_anova(prefix, rela, debug_path,_cc_list):
    global cc_list
    cc_list = _cc_list
    print("?")
    if debug_path:
        debug_path = os.path.join(debug_path, rela, 'anova')
        mkdir(debug_path)
    count_path = os.path.join(prefix, rela + '/result/count_num')
    anova_path = os.path.join(prefix, rela + '/result/anova')
    if not os.path.exists(anova_path):
        os.mkdir(anova_path)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for value in ['basic', 'user', 'domain']:
        # groud_truth_based_anova(count_path, anova_path, value,debug_path)
        pool.apply_async(groud_truth_based_anova, (
            count_path,
            anova_path,
            value,
            debug_path,
        ))
    pool.close()
    pool.join()


# 需要定义old_rank_file、rank_path
# 从old_rank_file拿到需要计算的区域，可以自己定义，具体在country_internal_rank函数使用
# rank_path为输出路径


@record_launch_time
def do_groud_truth_based_var(prefix, rela, debug_path,_cc_list):
    global cc_list
    cc_list = _cc_list
    path = os.path.join(prefix, rela, 'result')
    if debug_path:
        debug_path = os.path.join(debug_path, rela, 'var')
        mkdir(debug_path)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for value in ['basic', 'user', 'domain']:
        pool.apply_async(groud_truth_based_var, (
            path,
            value,
            debug_path,
        ))
    pool.close()
    pool.join()


@record_launch_time
def do_country_internal_rank(path, _cc_list, topo_list, debug_path):
    # old_prefix = '/home/peizd01/for_dragon/new_data/'
    global cc_list
    cc_list = _cc_list
    new_rank_path = os.path.join(path, 'public/med_rank.json')
    var_rank_path = os.path.join(path, 'public/var_rank.json')
    med_debug_path = None
    var_debug_path = None
    if debug_path:
        med_debug_path = os.path.join(debug_path, 'med_rank.json')
        var_debug_path = os.path.join(debug_path, 'var_rank.json')
    # 函数country_internal_rank内部的del_path、npz_file_name、anova_path需要定义
    # del_path和monitor_random_cut.py中的path含义相当
    # npz_file_name和create_routingtree.py中p2的含义相当
    # anova_path和本文件groud_truth_based_anova函数中anova_path含义相当
    country_internal_rank(path, new_rank_path, 'anova', topo_list, med_debug_path)
    country_internal_rank(path, var_rank_path, 'var', topo_list, var_debug_path)
