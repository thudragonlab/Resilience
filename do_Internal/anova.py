#!usr/bin/env python
# _*_ coding:utf8 _*_
import os
import json
import csv
from math import floor
from sklearn import linear_model
from do_Internal.sort_rank import cal_rank_weight
from other_script.my_types import *
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from pyecharts import options as opts
from importlib import import_module
from pyecharts.charts import Page, Tree, Bar, Line
from other_script.util import mkdir, record_launch_time, record_launch_time_and_param
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
as_importance_pat: WEIGHT_PATH = ''
gl_cut_node_depth: int
cc_list: List[COUNTRY_CODE] = []
gl_cal_rank_model:Callable[[Dict[AS_CODE,int],Dict[AS_CODE,int]],int] = None

def extract_connect_list_async(
    cc: COUNTRY_CODE,
    path: RTREE_PATH,
    dsn_path: COUNT_NUM_PATH,
):
    '''
    cc: country code
    path rtree 路径
    dsn_path count_num存储路径

    计算as对应的权重数据
    '''
    def basic_user_domain(line: str) -> Tuple[AFFECTED_AS_NUMBER, USER_IMPORTANT_WEIGHT, DOMAIN_IMPORTANT_WEIGHT]:
        '''
        line addDel.txt的每行记录(第一行除外)
        '''
        as_list: List[AS_CODE] = line.split(' ')
        res1: USER_IMPORTANT_WEIGHT = 0
        res2: DOMAIN_IMPORTANT_WEIGHT = 0
        for _as in as_list:
            if _as in as_importance:
                # print(as_importance)
                res1 += as_importance[_as][0]
                res2 += as_importance[_as][1]
        return (line.count(' ') + 1, res1, res2)

    if cc + '.json' not in os.listdir(as_importance_path):
        print(cc + ' not have as_importance')
        return
    with open(os.path.join(as_importance_path, cc + '.json'), 'r') as f:
        _as_importance: List[Tuple[COUNTRY_CODE, USER_IMPORTANT_WEIGHT, DOMAIN_IMPORTANT_WEIGHT]] = json.load(f)
    as_importance: Dict[COUNTRY_CODE, Tuple[USER_IMPORTANT_WEIGHT, DOMAIN_IMPORTANT_WEIGHT]] = {}
    for line in _as_importance:
        as_importance[line[0]] = line[1:]
    file_name: str = os.listdir(os.path.join(path, cc))
    file_name = [i for i in file_name if i.find('.addDel') != -1]

    if not os.path.exists(os.path.join(dsn_path, cc)):
        mkdir(os.path.join(dsn_path, cc))
    for file in file_name:
        res: Dict[DCOMPLETE_AS_CODE, Dict] = {}
        asname: DCOMPLETE_AS_CODE = file.split('.')[0]
        res[asname] = {}
        res[asname]['asNum']: int = -1
        # 对应破坏节点数量
        res[asname]['connect']: List[List[Tuple[AFFECTED_AS_NUMBER, USER_IMPORTANT_WEIGHT,
                                                DOMAIN_IMPORTANT_WEIGHT]]] = [[]] * gl_cut_node_depth
        f = open(os.path.join(path, cc, file), 'r')
        for line in f:
            l: List[str] = line.split('|')
            if line[0] == '#':
                res[asname]['asNum'] = int(l[1])
            elif len(l) > 1 and line[0][0] != '(' and l[0] != '':
                l1: int = l[0].count(' ')
                if l1 < len(res[asname]['connect']):
                    l2: Tuple[AFFECTED_AS_NUMBER, USER_IMPORTANT_WEIGHT, DOMAIN_IMPORTANT_WEIGHT] = basic_user_domain(l[1])
                    res[asname]['connect'][l1].append(l2)
        with open(os.path.join(dsn_path, cc, asname + '.json'), 'w') as df:
            json.dump(res, df)


def extract_connect_list(path: RTREE_PATH, dsn_path: COUNT_NUM_PATH):
    '''
    path 路由树路径
    dsn_path count_num存储路径
    '''
    cc_name: List[COUNTRY_CODE] = os.listdir(path)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for cc in cc_name:
        pool.apply_async(extract_connect_list_async, (
            cc,
            path,
            dsn_path,
        ))
        # extract_connect_list_async (
        #     cc,
        #     path,
        #     dsn_path,
        # )
    pool.close()
    pool.join()


def anova_sort(dsn_path:ANOVA_NUM_PATH, VALUE:ASPECT_TPYE, reader:List[Tuple[str,str,int]]):
    '''
    dsn_path: anova存储路径
    VALUE : basic|user|domain 维度类型
    reader : as两两比对的结果集

    进行最终排序,结果存到dsn_path
    '''
    # if os.path.exists(os.path.join(dsn_path, 'sorted_country_'+VALUE+'.json')): return
    # with open(os.path.join(dsn_path, 'anova_' + VALUE + '_multi_comparison.json'), 'r') as f:
    #     reader = json.load(f)
    res:Dict[str,Tuple[List[str],List[str],List[str]]] = {}  # 记录{AS:【比其更安全的AS】【无差异AS】【更不安全的AS】}
    for line in reader:
        if line[0] not in res: res[line[0]] = ([], [], [])
        if line[1] not in res: res[line[1]] = ([], [], [])
        if line[-1] == 0:
            res[line[0]][1].append(line[1])
            res[line[1]][1].append(line[0])
        elif line[-1] < 0:
            res[line[0]][0].append(line[1])
            res[line[1]][2].append(line[0])
        else:
            res[line[0]][2].append(line[1])
            res[line[1]][0].append(line[0])

    sorted_country:List[List[str]] = [[]]
    for k in res:
        # 如果没有比其更安全的AS（找最安全的AS）
        if len(res[k][0]) == 0:
            #排第一名
            sorted_country[-1].append(k)

    while len(res):
        for cc in sorted_country[-1]:
            for _cc in res[cc][2]:
                try:
                    #把 在排名里面的AS 从 比其更不安全的AS 列表对应的 比其更安全的AS列表中删除
                    res[_cc][0].remove(cc)
                except:
                    print(_cc)
                    print(cc)
                    exit()
                    #把 在排名里面的AS 从 无差异AS 列表对应的 无差异AS列表中删除
            for _cc in res[cc][1]:
                res[_cc][1].remove(cc)
            
            #把在排名中的AS从所有AS列表中删除
            del res[cc]
        temp:List[str] = []
        for cc in res:
            #如果此时 没有比其更安全的AS（找最安全的AS）
            if len(res[cc][0]) == 0:
                #则往下继续排名
                temp.append(cc)
        if len(temp) != 0:
            sorted_country.append(temp)

    with open(os.path.join(dsn_path, 'sorted_country_' + VALUE + '.json'), 'w') as f:
        json.dump(sorted_country, f)


def anova(dict_l: Dict[str, List[float]], dsn_path: ANOVA_NUM_PATH, VALUE: ASPECT_TPYE):
    '''
    dict_l 输入L:[[],...,[]]，列表每个元素为国家所有的破坏性度量
    dsn_path : anova存储路径
    VALUE : basic|user|domain 维度类型

    对AS进行两两对比,结果存到本地文件
    '''
    # if os.path.exists(os.path.join(dsn_path, 'anova_' + VALUE + '_multi_comparison.json')):
    #     return
    l: List[List[float]] = [v for _, v in dict_l.items()]
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

    nums: List[float] = []
    groups: List[str] = []
    for k, v in dict_l.items():
        nums += v
        groups += len(v) * [k]
    mc = MultiComparison(nums, groups)
    result = mc.tukeyhsd()

    res: List[Tuple[str,str,int]] = []
    for line in result._results_table.data[1:]:
        # print('line => %s \n '% line )
        if line[-1]:
            res.append((line[0], line[1], line[2]))
        else:
            res.append((line[0], line[1], 0))

    with open(os.path.join(dsn_path, 'anova_' + VALUE + '_multi_comparison.json'), 'w') as f:
        json.dump(res, f)
    anova_sort(dsn_path, VALUE, res)


@record_launch_time_and_param(2)
def groud_truth_based_anova(path: COUNT_NUM_PATH, dsn_path: ANOVA_NUM_PATH, value: ASPECT_TPYE):
    '''
    path count_num路径
    dsn_path anova结果存储路径
    value user|domain|basic 三个维度类型

    从basic,user,domain,三个维度进行anova排序

    '''
    def groud_truth_based_anova_thread(_cc: COUNTRY_CODE):
        '''
        cc: country code
        '''
        if _cc.find('.json') != -1 or _cc[0] == '.': return
        cc_name: List[str] = os.listdir(os.path.join(path, _cc))
        for file in cc_name:
            _l: List[float] = []
            try:
                with open(os.path.join(path, _cc, file), 'r') as f:
                    r = json.load(f)
            except Exception as e:
                print(file)

            for _as in r:
                N: int = r[_as]['asNum']
                if N < 0: continue

                for i in r[_as]['connect']:
                    if value == 'basic':
                        _l += [_i[value_dict[value]] / N for _i in i]
                    else:
                        _l += [_i[value_dict[value]] for _i in i]
            asname: DCOMPLETE_AS_CODE = file.split('.')[0]
            if len(_l) > 0:
                l[_cc + '-' + asname] = _l

    # file_name = os.listdir(path)
    '''
    l 的数据结构
    {
        AD_dcomplete{as_code}:[connect[0][受影响的节点数量]/asNum,....connect[100][受影响的节点数量]/asNum]
    }
    '''
    l: Dict[str, List[float]] = {}
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for _cc in cc_list:
        thread_pool.apply(groud_truth_based_anova_thread, (_cc, ))
    thread_pool.close()
    thread_pool.join()
    # print(l[list(l.keys())[0]])
    anova(l, dsn_path, value)
    # anova_sort(dsn_path, value, debug_path)


@record_launch_time
def country_internal_rank(path:OUTPUT_PATH, rank_path, _type, topo_list:List[TOPO_TPYE], data_dim:List[ASPECT_TPYE]):
    '''
    path output 路径
    rank_path : 最终排序文件存储路径
    _type : 排序类型 (anova var)
    topo_list : topo类型
    data_dim basic|user|domain 维度列表
    '''
    def country_internal_rank_thread(value2, value):
        '''
        value2: basic|user|domain 维度类型
        value:topo类型
        '''
        del_path = os.path.join(path, value, 'rtree')
        npz_file_name = del_path
        anova_path = os.path.join(path, value, 'result', _type, 'sorted_country_' + value2 + '.json')
        with open(anova_path, 'r') as f:
            reader = json.load(f)
        res:Dict[COUNTRY_CODE,Dict[AS_CODE,int]] = {}
        for index in range(len(reader)):
            for _v in reader[index]:
                _cc, _as = _v.split('-')
                if _cc not in res: res[_cc] = {}
                if len(_as) == 0: continue
                if _as[0] == 'd': _as = _as[9:]
                res[_cc][_as] = index + 1  # 每个国家下每个AS的排名
        for _cc in res:
            if _cc not in rank: continue
            temp = cal_rank_weight(_cc,del_path,res[_cc])
           
            if _cc in rank:
                if temp < 1:
                    temp = 1
                rank[_cc][f'{value2}-{value}'] = temp
                # print(temp)

    rank = {}
    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for _cc in cc_list:
        rank[_cc] = {}
    for value2 in data_dim:
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


def judge_var(target_list, result):
    '''
    target_list 待排序的as列表
    result 用来存储最终结果

    用递归的方式，按照方差从小到大为数据排序
    

    '''
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
def groud_truth_based_var(path:RESULT_PATH, _type:ASPECT_TPYE):
    '''
    path result文件夹路径
    _type basic|user|domain 维度路径

    计算方差排名结果，结果存储在本地文件
    '''
    def groud_truth_based_var_thread(cc):
        '''
        cc: country code

        '''
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

    var_result:Dict[str,Dict] = {}
    result:List[List[str]] = []
    value_dict = {'basic': 0, 'user': 1, 'domain': 2}
    count_num_path = os.path.join(path, 'count_num')
    cc_list = os.listdir(count_num_path)

    thread_pool = ThreadPool(multiprocessing.cpu_count() * 10)
    for cc in cc_list:
        thread_pool.apply(groud_truth_based_var_thread, (cc, ))
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



# 需要定义rtree_path、count_path、as_importance_path
# rtree_path和create_routingtree.py中p2的含义相当
# count_path为输出路径
# as_importance_path为区域AS的资源权重

# prefix = '/home/peizd01/for_dragon/new_data/'


# for rela in ['asRank', 'problink', 'toposcope', 'toposcope_hidden']:
# for rela in ['toposcope', 'toposcope_hidden']:
@record_launch_time
def do_extract_connect_list(prefix: OUTPUT_PATH, rela: TOPO_TPYE, weight_data_path: WEIGHT_PATH, cut_node_depth: int):
    '''
    prefix output路径
    rela topo类型
    weight_data_path 权重数据路径
    cut_node_depth 破坏节点数量

    创建count_num文件夹,统计所有被破坏As的权重数据
    '''
    rtree_path: RTREE_PATH = os.path.join(prefix, rela + '/rtree/')
    count_path: COUNT_NUM_PATH = os.path.join(prefix, rela + '/result/count_num')
    global as_importance_path
    global gl_cut_node_depth
    as_importance_path = weight_data_path
    gl_cut_node_depth = cut_node_depth
    extract_connect_list(rtree_path, count_path)


# 需要定义count_path、anova_path
# count_path和本文件extract_connect_list函数中count_path含义相当
# anova_path为输出路径
# for rela in ['asRank', 'problink', 'toposcope', 'toposcope_hidden']:
# for rela in ['toposcope', 'toposcope_hidden']:
@record_launch_time
def do_groud_truth_based_anova(prefix: OUTPUT_PATH, rela: TOPO_TPYE, _cc_list: List[COUNTRY_CODE]):
    '''
    prefix output路径
    rela topo类型
    _cc_list 国际列表

    用anova排序
    '''
    global cc_list
    cc_list = _cc_list
    count_path: COUNT_NUM_PATH = os.path.join(prefix, rela + '/result/count_num')
    anova_path: ANOVA_NUM_PATH = os.path.join(prefix, rela + '/result/anova')
    if not os.path.exists(anova_path):
        os.mkdir(anova_path)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for value in ['basic', 'user', 'domain']:
        # groud_truth_based_anova(count_path, anova_path, value,debug_path)
        pool.apply_async(groud_truth_based_anova, (
            count_path,
            anova_path,
            value,
        ))
    pool.close()
    pool.join()


# 需要定义old_rank_file、rank_path
# 从old_rank_file拿到需要计算的区域，可以自己定义，具体在country_internal_rank函数使用
# rank_path为输出路径


@record_launch_time
def do_groud_truth_based_var(prefix:OUTPUT_PATH, rela:TOPO_TPYE, _cc_list: List[COUNTRY_CODE]):
    '''
    prefix: output 路径
    rela : topo类型
    _cc_list : 国家列表
    '''
    global cc_list
    cc_list = _cc_list
    path:RESULT_PATH = os.path.join(prefix, rela, 'result')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for value in ['basic', 'user', 'domain']:
        pool.apply_async(groud_truth_based_var, (
            path,
            value,
        ))
    pool.close()
    pool.join()


@record_launch_time
def do_country_internal_rank(path:OUTPUT_PATH, _cc_list:List[COUNTRY_CODE], topo_list:List[TOPO_TPYE], data_dim:List[ASPECT_TPYE]):
    '''
    path output路径
    _cc_list 国家列表
    topo_list topo类型列表
    data_dim user|domain|basic 维度列表
    '''
    # old_prefix = '/home/peizd01/for_dragon/new_data/'
    global cc_list
    # global gl_cal_rank_model
    cc_list = _cc_list
    # dynamic_module_1= import_module(cal_rank_model_path)
    # gl_cal_rank_model = dynamic_module_1.do_cal

    new_rank_path = os.path.join(path, 'public/med_rank.json')
    var_rank_path = os.path.join(path, 'public/var_rank.json')
    med_debug_path = None
    var_debug_path = None
    # if debug_path:
    #     med_debug_path = os.path.join(debug_path, 'med_rank.json')
    #     var_debug_path = os.path.join(debug_path, 'var_rank.json')
    # 函数country_internal_rank内部的del_path、npz_file_name、anova_path需要定义
    # del_path和monitor_random_cut.py中的path含义相当
    # npz_file_name和create_routingtree.py中p2的含义相当
    # anova_path和本文件groud_truth_based_anova函数中anova_path含义相当
    # 计算anova最终排名
    country_internal_rank(path, new_rank_path, 'anova', topo_list, data_dim)
    # 计算方差最终排名
    country_internal_rank(path, var_rank_path, 'var', topo_list, data_dim)
