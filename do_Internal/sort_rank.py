#!usr/bin/env python
# _*_ coding:utf8 _*_
import multiprocessing
import os
import json
from typing import Callable, Dict, List
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
from multiprocessing.pool import ThreadPool
from pyecharts.globals import WarningType
from other_script.util import mkdir
from importlib import import_module

WarningType.ShowWarning = False
gl_cal_rank_model:Callable

def set_gl_cal_rank_model(cal_rank_model_path) -> None:
    global gl_cal_rank_model
    dynamic_module_1= import_module(cal_rank_model_path)
    gl_cal_rank_model = dynamic_module_1.do_cal


def anova(dict_l, dsn_path, VALUE, num):
    '''
    输入L:[[],...,[]]，列表每个元素为国家所有的破坏性度量
    '''
    l = [v for _, v in dict_l.items()]
    f, p = stats.f_oneway(*l)
    print('One-way ANOVA')
    print('=============')
    print('F value:', f)
    print('P value:', p, '\n')
    if p > 0.05:
        print('无显著性差异 p>0.05')
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
        if line[-1]:
            res.append([line[0], line[1], line[2]])
        else:
            res.append([line[0], line[1], 0])

    anova_sort(dsn_path, VALUE, num, res)


def anova_sort(dsn_path, VALUE, num, reader):
    if os.path.exists(os.path.join(dsn_path, 'sorted_country_' + VALUE + '.' + num + '.json')): return
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
                    exit()
            for _cc in res[cc][1]:
                res[_cc][1].remove(cc)
            del res[cc]
        temp = []
        for cc in res:
            if len(res[cc][0]) == 0:
                temp.append(cc)
        sorted_country.append(temp)

    with open(os.path.join(dsn_path, 'sorted_country_' + VALUE + '.' + num + '.json'), 'w') as f:
        json.dump(sorted_country, f)


def cal_rank_weight(_cc: str, rtree_path: str, as_dict: Dict[str, int]) -> int:
    '''
    _cc: country code
    rtree_path: 路由树路径
    as_dict : as排名字典

    计算国家最终排名
    返回排名权重
    '''
    file_name = os.listdir(os.path.join(rtree_path, _cc))
    file_name = [i for i in file_name if i.find('.graph') != -1]
    npz_file_name = [i for i in os.listdir(os.path.join(rtree_path, _cc)) if i[-1] == 'z']
    if len(file_name) == 0:
        temp = 0
        for _as in as_dict:
            temp += as_dict[_as]  # temp是每个国家下所有AS排名之和
        temp = temp / len(as_dict)  # temp 再除以这个国家下AS的数量
    else:
        ans = {}
        for _as in as_dict:
            if 'dcomplete' + _as + '.npz.graph.json' in file_name:  # 如果有graph文件
                with open(os.path.join(rtree_path, _cc, 'dcomplete' + _as + '.npz.graph.json'), 'r') as f:
                    n = json.load(f)
                ans[_as] = len(set(list(n.keys())))  # 统计每个路由树下的链接个数
            elif 'dcomplete' + _as + '.npz' in npz_file_name:  #否则解析npz
                m = np.load(os.path.join(rtree_path, _cc, 'dcomplete' + _as + '.npz'))
                ans[_as] = len(set(m['row']))  # 统计每个路由树下的链接个数
            else:
                ans[_as] = 0
        temp = gl_cal_rank_model(as_dict,ans)
    if temp < 1: temp = 1
    return temp


def country_internal_rank( topo_list, output_path, RESULT_SUFFIX, type_path, _type, country_name, num,data_dim):
    '''
    topo_list topo类型列表
    output_path output路径
    RESULT_SUFFIX 优化后排序结果存储路径
    type_path : anova var
    _type : med var
    country_name : country code
    num 优化连接数量
    data_dim : basic|user|domain列表
    '''
    rank_dir_path = os.path.join(output_path, 'public', 'optimize')
    mkdir(rank_dir_path)

    rank_dsn_path = os.path.join(rank_dir_path, '%s_rank.%s.%s.json' % (_type, country_name, str(num)))
    old_rank_path = os.path.join(output_path, 'public', '%s_rank.json' % _type)
    with open(old_rank_path, 'r') as orf:
        rank = json.load(orf) #读取之前的排名数据
    
    new_cc_rank_weight = {}
    for value2 in data_dim:
        for m in topo_list:
            old_graph_path = os.path.join(output_path, m, 'rtree')
            sort_dsn_path = os.path.join(output_path, m, RESULT_SUFFIX, type_path)

            path = os.path.join(sort_dsn_path, value2 + '_' + country_name,
                                'sorted_country_' + value2 + '.' + str(num) + '.json')
            with open(path, 'r') as f: #读取优化后的AS排名结果
                reader = json.load(f)
            res = {}
            for index in range(len(reader)):
                for _v in reader[index]:
                    _cc, _as = _v.split('-')
                    if _cc != country_name: #只记录优化的国家排名
                        continue
                    if not _as[0].isdigit(): _as = _as[9:]
                    res[_as] = index + 1 #记录国家内所有AS的排名

            temp = cal_rank_weight(country_name, old_graph_path, res)
            new_cc_rank_weight[f'{value2}-{m}'] = temp

    rank[country_name] = new_cc_rank_weight
    with open(rank_dsn_path, 'w') as f:
        json.dump(rank, f)


def groud_truth_based_anova_for_single_country(single_country_path, single_country_name, old_connect_path, dsn_path, value,
                                               num):
    num = str(num)
    file_name = os.listdir(old_connect_path)
    l = {}
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    cc_name = os.listdir(single_country_path)
    if not len(cc_name): print(single_country_name, value, num, single_country_path, ' file len==0 error!!!')

    def groud_truth_based_anova_for_single_country_old_thread(_cc):
        if _cc == single_country_name: return
        if _cc.find('.json') != -1 or _cc[0] == '.': return
        cc_name = os.listdir(os.path.join(old_connect_path, _cc))
        for file in cc_name:
            if file.find('.json') == -1 or file[0] == '.':
                continue
            _l = []
            try:
                with open(os.path.join(old_connect_path, _cc, file), 'r') as f:
                    r = json.load(f)
            except:
                exit()
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

    def groud_truth_based_anova_for_single_country_new_thread(file):
        _l = []
        file_path = os.path.join(single_country_path, file)
        with open(file_path, 'r') as f:
            r = json.load(f)
        for _as in r:
            N = r[_as]['asNum']
            for i in r[_as]['connect']:
                if value == 'basic':
                    _l += [_i[value_dict[value]] / N for _i in i]
                else:
                    _l += [_i[value_dict[value]] for _i in i]
        asname = file.split('.')[0]
        if len(_l) > 0:
            l[single_country_name + '-' + asname] = _l
            print(single_country_name, value, num, file_path + ' added')
        else:
            print(single_country_name, value, num, file_path + ' len==0 not added')

    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)

    for _cc in file_name:
        thread_pool.apply_async(groud_truth_based_anova_for_single_country_old_thread, (_cc, ))

    for file in cc_name:
        thread_pool.apply_async(groud_truth_based_anova_for_single_country_new_thread, (file, ))

    thread_pool.close()
    thread_pool.join()

    anova(l, dsn_path, value, num)


def internal_survival(rank_file, begin_index, end_index):
    with open(rank_file, 'r') as f:
        r = json.load(f)
    for k in r:
        if len(r[k]) == 24: r[k] = r[k][-12:]
    k = list(r.keys())
    for _k in k:
        if len(r[_k]) == 0: del r[_k]
    r = list(r.items())
    ccres = []
    for i in range(len(r)):
        ccres.append([])
    for i in range(len(r)):
        ccres[i].append(r[i][0])
    for index in range(begin_index, end_index + 1):
        for i in range(len(r)):
            ccres[i].append(r[i][1][index])
            if ccres[i][-1] < 1: ccres[i][-1] = 1

    for locate in range(end_index - begin_index + 1):
        ccres = sorted(ccres, key=(lambda x: sum(x[1:])), reverse=False)
        ccres = sorted(ccres, key=(lambda x: x[locate + 1]), reverse=False)
        temp = ccres[0][locate + 1]
        for index in range(len(ccres)):
            if index > 0 and ccres[index][locate + 1] == temp:
                ccres[index][locate + 1] = ccres[index - 1][locate + 1]
            else:
                temp = ccres[index][locate + 1]
                if index > 0:
                    ccres[index][locate + 1] = ccres[index - 1][locate + 1] + 1
                else:
                    ccres[index][locate + 1] = 1

    ccres = sorted(ccres, key=(lambda x: sum(x[1:])), reverse=False)
    return ccres
