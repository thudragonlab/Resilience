#!usr/bin/env python
# _*_ coding:utf8 _*_
import os
import json
import csv
from math import floor
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

from pyecharts.globals import WarningType
WarningType.ShowWarning = False


def anova(dict_l, dsn_path, VALUE, num):
    '''
    输入L:[[],...,[]]，列表每个元素为国家所有的破坏性度量
    '''
    if os.path.exists(os.path.join(dsn_path, 'anova_'+VALUE+'_multi_comparison.'+num+'.json')):
        return
    l = [v for _,v in dict_l.items()]
    f,p =stats.f_oneway(*l)
    print ('One-way ANOVA')
    print ('=============')
    print ('F value:', f)
    print ('P value:', p, '\n')
    if p > 0.05:
        print('无显著性差异 p>0.05')
        return
    else:
        print('有显著性差异')
    
    nums,groups = [], []
    for k, v in dict_l.items():
        nums+=v
        groups += len(v)*[k]
    mc = MultiComparison(nums, groups)
    result = mc.tukeyhsd()

    res = []
    for line in result._results_table.data[1:]:
        if line[-1]:
            res.append([line[0], line[1], line[2]])
        else:
            res.append([line[0], line[1], 0])

    with open(os.path.join(dsn_path, 'anova_'+VALUE+'_multi_comparison.'+num+'.json'), 'w')  as f:
        json.dump(res, f)
    print(os.path.join(dsn_path, 'anova_'+VALUE+'_multi_comparison.'+num+'.json')+' created')
 
def anova_sort(dsn_path,VALUE, num):
    if os.path.exists(os.path.join(dsn_path, 'sorted_country_'+VALUE+'.'+num+'.json')): return
    with open(os.path.join(dsn_path, 'anova_'+VALUE+'_multi_comparison.'+num+'.json'), 'r')  as f:
        reader = json.load(f)
    res = {} # 记录{国家:【比其更安全的国家】【无差异国家】【更不安全的国家】}
    for line in reader:
        if line[0] not in res: res[line[0]] = [[], [], []]
        if line[1] not in res: res[line[1]] = [[], [], []]
        if line[-1]==0:
            res[line[0]][1].append(line[1])
            res[line[1]][1].append(line[0])
        elif line[-1]<0:
            res[line[0]][0].append(line[1])
            res[line[1]][2].append(line[0])
        else:
            res[line[0]][2].append(line[1])
            res[line[1]][0].append(line[0])

    sorted_country = [[]]
    for k in res:
        if len(res[k][0])==0:
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
            if len(res[cc][0])==0:
                temp.append(cc)
        sorted_country.append(temp)

    with open(os.path.join(dsn_path, 'sorted_country_'+VALUE+'.'+num+'.json'), 'w')  as f:
        json.dump(sorted_country, f)
    

def country_internal_rank(cc_list, old_graph_path, sort_dsn_path,country_name,rank_dsn_path,num):
    rank = {k:[] for k in cc_list}
    for value2 in ['basic', 'user', 'domain']:
        path = os.path.join(sort_dsn_path, value2+'_'+country_name, 'sorted_country_'+value2+'.'+str(num)+'.json')
        with open(path, 'r') as f:
            reader = json.load(f)
        res = {}
        for index in range(len(reader)):
            for _v in reader[index]:
                _cc, _as = _v.split('-')
                if not _as[0].isdigit(): _as = _as[9:]
                if _cc not in res: res[_cc] = {}
                res[_cc][_as] = index+1
        for _cc in res:
            if _cc not in rank: continue
            file_name = os.listdir(os.path.join(old_graph_path, _cc))
            file_name =[i for i in file_name if i.find('.graph')!=-1]
            npz_file_name = [i for i in os.listdir(os.path.join(old_graph_path, _cc)) if i[-1]=='z']
            if len(file_name)==0:
                temp = 0
                for _as in res[_cc]:
                    temp+=res[_cc][_as]
                temp = temp/len(res[_cc])
            else:
                ans = {}
                for _as in res[_cc]:
                    if 'dcomplete'+_as+'.npz.graph.json' in file_name:
                        with open(os.path.join(old_graph_path, _cc,'dcomplete'+_as+'.npz.graph.json'), 'r') as f:
                            n = json.load(f)
                        ans[_as] = len(set(list(n.keys())))
                    elif 'dcomplete'+_as+'.npz' in npz_file_name:
                        m = np.load(os.path.join(old_graph_path, _cc,'dcomplete'+_as+'.npz'))
                        ans[_as] = len(set(m['row']))
                    else:
                        ans[_as]=0
                temp = 0
                for _as in res[_cc]:
                    if ans[_as] == 0: continue
                    temp += res[_cc][_as]*ans[_as]
                if sum(list(ans.values()))==0:
                    temp = 0
                else:
                    temp /= sum(list(ans.values()))
            if _cc in rank:
                if temp<1: temp = 1
                rank[_cc].append(temp)

    with open(rank_dsn_path, 'w') as f:
        json.dump(rank, f)

def groud_truth_based_anova_for_single_country(single_country_path, single_country_name, old_connect_path, dsn_path, value, num):
    if os.path.exists(os.path.join(dsn_path, 'sorted_country_'+value+'.'+num+'.json')): return
    num = str(num)
    file_name = os.listdir(old_connect_path)
    l = {}
    value_dict = {'basic':0, 'user':1, 'domain':2}
    for _cc in file_name:
        if _cc==single_country_name: continue
        if _cc.find('.json')!=-1 or _cc[0]=='.': continue
        cc_name = os.listdir(os.path.join(old_connect_path, _cc))
        for file in cc_name:
            if file.find('.json')==-1 or file[0]=='.': 
                continue
            _l = []
            try:
                with open(os.path.join(old_connect_path, _cc, file), 'r') as f:
                    r = json.load(f)
            except:
                print(os.path.join(old_connect_path, _cc, file)+' read error')
                print(os.path.join(dsn_path, 'sorted_country_'+value+'.'+num+'.json'))
                exit()
            for _as in r:
                N = r[_as]['asNum']
                if N<0: continue
                if N<20: continue
                for i in r[_as]['connect']:
                    _l+=[_i[value_dict[value]]/N for _i in i]
            asname = file.split('.')[0]
            if len(_l)>0:
                l[_cc+'-'+asname] = _l
    cc_name = os.listdir(single_country_path)
    if not len(cc_name): print(single_country_name, value, num, \
                            single_country_path, ' file len==0 error!!!')
    for file in cc_name:
        _l = []
        with open(os.path.join(single_country_path, file), 'r') as f:
            r = json.load(f)
        for _as in r:
            N = r[_as]['asNum']
            if N<20: continue
            for i in r[_as]['connect']:
                _l+=[_i[value_dict[value]]/N for _i in i]
        asname = file.split('.')[0]
        if len(_l)>0:
            l[single_country_name+'-'+asname] = _l
            print(single_country_name, value, num, file+' added')
        else: print(single_country_name, value, num, file+' len==0 not added')
    anova(l, dsn_path, value, num)
    anova_sort(dsn_path, value, num)


def internal_survival(rank_file, begin_index, end_index):

    VALUE = ['_b', '_u', '_d']
    # PATH = ['Asrank', 'Problink', 'Toposcope', 'hToposcope']
    PATH = ['1', '2', '3', '4']
    yaxis = []
    for v in VALUE:
        for p in PATH:
            yaxis.append(p+v)
    
    with open(rank_file, 'r') as f:
        r = json.load(f)
    for k in r:
        if len(r[k])==24: r[k] = r[k][-12:]
    k = list(r.keys())
    for _k in k:
        if len(r[_k])==0: del r[_k]
    r = list(r.items())
    locate = [0,3,6,9,1,4,7,10,2,5,8,11]
    ccres = []
    for i in range(len(r)): ccres.append([])
    for i in range(len(r)):
        ccres[i].append(r[i][0])
    for index in range(begin_index, end_index+1):
        for i in range(len(r)):
            ccres[i].append(r[i][1][index])
            if ccres[i][-1]<1: ccres[i][-1]=1

    for locate in range(end_index-begin_index+1):
        ccres = sorted(ccres, key=(lambda x: sum(x[1:])), reverse=False)
        ccres = sorted(ccres, key=(lambda x: x[locate+1]), reverse=False)
        temp = ccres[0][locate+1]
        for index in range(len(ccres)):
            if index>0 and ccres[index][locate+1]==temp:
                ccres[index][locate+1] = ccres[index-1][locate+1]
            else:
                temp = ccres[index][locate+1]
                # ccres[index][locate+1] = index+1
                if index>0:
                    ccres[index][locate+1] = ccres[index-1][locate+1]+1
                else: ccres[index][locate+1]=1
            # ccres[index][locate+1] = index+1
                
    ccres = sorted(ccres, key=(lambda x: sum(x[1:])), reverse=False)
    return ccres

