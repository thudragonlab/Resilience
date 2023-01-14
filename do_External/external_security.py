#!usr/bin/env python
# _*_ coding:utf8 _*_
import os
import json
from math import floor
import numpy as np
from scipy import stats
from other_script.my_types import *
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import time
from pyecharts.globals import WarningType
WarningType.ShowWarning = False


def second_order(path, file, dsn_file, value):
    def get_anova_sort(dict_l):
        l = [v for _,v in dict_l.items()]
        nums,groups = [], []
        for k, v in dict_l.items():
            nums+=v
            groups += len(v)*[k]
        f,p =stats.f_oneway(*l)
        if p > 0.05:
            print('无显著性差异 p={}'.format(p))
            return None
        else:
            print('有显著性差异')

        mc = MultiComparison(nums, groups)
        result = mc.tukeyhsd()

        res = []
        for line in result._results_table.data[1:]:
            if line[-1]:
                res.append([line[0], line[1], line[2]])
            else:
                res.append([line[0], line[1], 0])

        Rres = res
        res = {} # 记录{国家:【比其更安全的国家】【无差异国家】【更不安全的国家】}
        for line in Rres:
            if line[0] not in res: res[line[0]] = [set(), set(), set()]
            if line[1] not in res: res[line[1]] = [set(), set(), set()]
            if line[-1]==0:
                res[line[0]][1].add(line[1])
                res[line[1]][1].add(line[0])
            elif line[-1]<0:
                res[line[0]][0].add(line[1])
                res[line[1]][2].add(line[0])
            else:
                res[line[0]][2].add(line[1])
                res[line[1]][0].add(line[0])
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
                        print(cc)
                        exit()
                for _cc in res[cc][1]:
                    res[_cc][1].remove(cc)
                del res[cc]
            temp = []
            for cc in res:
                if len(res[cc][0])==0:
                    temp.append(cc)
            flag = 0
            sorted_country.append(temp)
        return sorted_country
    def get_line_sort(line):
        _l = []
        for _as in line:
            _file = _as+'.json'
            
            with open(os.path.join(path, _file), 'r') as f:
                r = json.load(f)
            for i in r:
                _l += [_i[value_dict[value]] for _i in i]

            if len(_l)>0:
                l[_file.split('.')[0]] = _l
        temp = get_anova_sort(l)
        if temp: return temp
        else: return [line[0:1], line[1:]]
        
    def merge(m, n):
        i_m = 0
        i_n = 0
        res = []
        while i_m < len(m) or i_n < len(n):
            if i_m>=len(m):
                res += n[i_n:]
                break
            if i_n>=len(n):
                res += m[i_m:]
                break
            if set(m[i_m]) & set(n[i_n]):
                res.append(list(set(m[i_m]) | set(n[i_n])))
                i_m += 1
                i_n += 1
            else:
                res.append(n[i_n])
                i_n+=1
        i = len(res)-1
        while i>=0:
            if len(res[i])==0:
                res.pop(i)
            else:
                break
            i-=1
        return res
            

    value_dict = {'gdp':0, 'domain':1, 'democracy':2}
    # gdp, domain, democracy = 0,1,2
    with open(file, 'r') as f:
        r = json.load(f)
    index = 0
    N = 10
    while index < len(r):
        l = {}
        line = r[index]
        if index+1<len(r): line += r[index+1]
        if len(line)>N:
            _sorted = []
            line = [line[i:N+i] for i in range(0,len(line),N)]
            for _i in range(len(line)):
                _line = line[_i]
                for __sorted in _sorted:
                    if len(__sorted): _line.append(__sorted[0])
                _r = get_line_sort(_line)
                _sorted = merge(_sorted, _r)
            r[index:index+2] = _sorted
        else:
            r[index:index+2] = get_line_sort(line)
        index += 1
    with open(dsn_file, 'r') as f:
        json.dump(r, f)



def anova(dict_l, dsn_path, VALUE, begin, end):
    '''
    输入L:[[],...,[]]，列表每个元素为边界AS所有的破坏性度量
    '''
    l = [v for _,v in dict_l.items()]
    
    # f,p =stats.f_oneway(*l)
    # print ('One-way ANOVA')
    # print ('=============')
    # print ('F value:', f)
    # print ('P value:', p, '\n')
    # if p > 0.05:
    #     print('无显著性差异 p>0.05')
    #     return
    # else:
    #     print('有显著性差异')
    
    # _k = list(dict_l.keys())
    # k = [ [] for i in range(8)]
    # for i,e in enumerate(_k):
    #     k[i%n].append(e)

    # for i in range(8):
    #     for j in range(i+1, 8):
    # nums,groups = [], []
    # for _k in k[i]:
    #     nums+=dict_l[_k]
    #     groups += len(dict_l[_k])*[k]
    # for _k in k[j]:
    #     nums+=dict_l[_k]
    #     groups += len(dict_l[_k])*[k]
    res = []
    nums,groups = [], []
    for k, v in dict_l.items():
        print(k,v)
        nums+=v
        groups += len(v)*[k]
    # print(groups,nums)
    if len(set(groups)) > 1:
        mc = MultiComparison(nums, groups)
        result = mc.tukeyhsd()

        res = []
        for line in result._results_table.data[1:]:
            if line[-1]:
                res.append([line[0], line[1], line[2]])
            else:
                res.append([line[0], line[1], 0])
    elif len(groups) > 0:
        res.append([groups[0], groups[0], 0])
    with open(os.path.join(dsn_path, 'anova_'+VALUE+'_multi_comparison.'+str(begin)+'_'+str(end)+'.json'), 'w')  as f:
        json.dump(res, f)
 
    # print(result)
    # print(result._results_table.data)            


def anova_sort(dsn_path, VALUE):
    # with open(os.path.join(dsn_path, 'anova'+VALUE+'_multi_comparison.json'), 'r')  as f:
    #     reader = json.load(f)
    reader = set()
    res = {}
    for file in os.listdir(dsn_path):
        if file.find('multi')!=-1 and file.find(VALUE)!=-1:
            with open(os.path.join(dsn_path,file), 'r') as f:
                reader = json.load(f)
    # 记录{国家:【比其更安全的国家】【无差异国家】【更不安全的国家】}
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
                res[_cc][0].remove(cc)
            for _cc in res[cc][1]:
                res[_cc][1].remove(cc)
            del res[cc]
        temp = []
        for cc in res:
            if len(res[cc][0])==0:
                temp.append(cc)
        flag = 0
        sorted_country.append(temp)
    with open(os.path.join(dsn_path, 'sorted_country_'+VALUE+'.json'), 'w')  as f:
    # with open(os.path.join(dsn_path, 'sorted_country'+VALUE+'.json'), 'w')  as f:
        json.dump(sorted_country, f)



def groud_truth_based_anova_single(path, dsn_path,value):
    _dir = os.listdir(path)
    value_dict = {'gdp':0, 'domain':1, 'democracy':2}
    N = 30
    begin = -1
    l = {}
    for _dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, _dir)):
            print(begin, len(l))
            begin += 1
            if len(l)==N:
                anova(l, dsn_path, value, begin-N, begin)
                print(begin,' create')
                if len(l)>0:
                    remove_k = list(l.keys())[:int(1*len(l)/5)]
                    for _k in remove_k:
                        l.pop(_k)
            _l=[]
            with open(os.path.join(path, _dir ,file), 'r') as f:
                r = json.load(f)
            for i in r:
                _l += [_i[value_dict[value]] for _i in i]

            if len(_l)>0:
                l[_dir+'-'+file.split('.')[0]] = _l
            else:
                with open('len_zero.txt', 'a+') as f:
                    f.write(_dir+'-'+file.split('.')[0]+'\n')
    if len(l)>0:
        anova(l, dsn_path, value, begin-N-len(l), begin)
        print(begin,' create')
    anova_sort(dsn_path, value)


def groud_truth_based_anova(source_path,):
    '''
    STEP1 通过anova分析，输入各国的全部连通性度量，看是否有显著性差异。
    STEP2 如有，判断国家两两间差异。
    STEP3 对国家进行排序
    '''
    path = os.path.join(source_path,'result','count_num')
    dsn_path = os.path.join(source_path,'result','anova')
    os.makedirs(dsn_path,exist_ok=True)
    file_name = os.listdir(path)
    gdp, domain, democracy = 0,1,2
    N = 30
    begin = 0
    while begin<len(file_name):
        l = {}
        end = begin+N
        a = time.time()
        for file in file_name[begin:end]:
            # print(file)
            
            _l = []
            with open(os.path.join(path, file), 'r') as f:
                r = json.load(f)
            for i in r:
                _l += [_i[democracy] for _i in i]

            # if len(_l)>0:
            l[file.split('.')[0]] = _l
            # else:
            #     print(_l)
        # if len(l.keys()) == 0:
        #     print(file_name[begin:end])
        anova(l, dsn_path, 'democracy', begin, end)
        begin += int(N/2)
        print(end,' create')
        b = time.time()
        # print(b-a)
    anova_sort(dsn_path, 'democracy')



def judge_var(target_list, result):
    print(len(target_list))
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



def groud_truth_based_var(source_path,):
    '''
    STEP1 通过anova分析，输入各国的全部连通性度量，看是否有显著性差异。
    STEP2 如有，判断国家两两间差异。
    STEP3 对国家进行排序
    '''
    path = os.path.join(source_path,'result','count_num')
    dsn_path = os.path.join(source_path,'result','var')
    _type = 'democracy'
    os.makedirs(dsn_path,exist_ok=True)
    file_name = os.listdir(path)
    gdp, domain, democracy = 0,1,2
    var_result:Dict[str,Dict] = {}
    result:List[List[str]] = []
    N = 30
    begin = 0
    while begin<len(file_name):
        l = {}
        end = begin+N
        a = time.time()
        for file in file_name[begin:end]:
            asn = file.split('.')[0]
            _l = []
            with open(os.path.join(path, file), 'r') as f:
                r = json.load(f)
            for i in r:
                _l += [_i[democracy] for _i in i]

            var_result[asn] = {
                                'list': [0] if len(_l) == 0 else _l,
                                'key': asn
                            }
        # anova(l, dsn_path, 'democracy', begin, end)
        begin += int(N/2)
        print(end,' create')
        b = time.time()
        # print(b-a)
    # anova_sort(dsn_path, 'democracy')
    var_list = list(var_result.values())
    
    var_zero_list =  list(filter(lambda x:np.var(x['list']) == 0.0,var_list))
    var_no_zero_list = list(filter(lambda x:np.var(x['list']) != 0.0,var_list))
    var_no_zero_list.sort(key=lambda x: np.var(x['list']))
    
    result.append(list(map(lambda x:x['key'],var_zero_list)))
    judge_var(var_no_zero_list, result)
    with open(os.path.join(dsn_path, 'sorted_country_%s.json' % _type), 'w') as sorted_var_f:
        json.dump(result, sorted_var_f)


def extract_connect_list(source_path,gdp_domain_democracy_path,cut_node_depth):
    path = os.path.join(source_path,'monitor')
    dsn_path = os.path.join(source_path,'result','count_num')
    json_path = os.path.join(source_path,'json')
    os.makedirs(dsn_path,exist_ok=True)
    def gdp_domain_democracy(line):
        as_list = line.split(' ')
        res1 = 0
        res2 = 0
        res3 = 0
        for _as in as_list:
            if _as in country_importance:
                res1 += country_importance[_as][0]
                res2 += country_importance[_as][1]
                res3 += country_importance[_as][2]
        return [len(as_list),res1,res2,res3]

    with open(gdp_domain_democracy_path, 'r') as f:
        _country_importance = json.load(f)
    country_importance = {}
    for line in _country_importance:
        country_importance[line[0]] = line[1:]
    file_name = os.listdir(path)
    json_file_list = os.listdir(json_path)
    for file in file_name:
        if file.find('.txt')==-1: continue
        
        asname = file.split('.')[0][9:]
        if asname+'.json' in os.listdir(dsn_path):
            print('exist')
            # continue
        if file.split('.')[0]+'.cc_rela.json' not in json_file_list:
            continue
        with open(os.path.join(json_path, file.split('.')[0]+'.cc_rela.json'), 'rU') as f_rela:
            relas = json.load(f_rela)
        with open(os.path.join(json_path, file.split('.')[0]+'.nonconnect.json'), 'rU') as f_non:
            nonconnect = json.load(f_non)
        nonconnect = ' '.join(nonconnect)
        cclist = set()
        for k in relas:
            cclist.add(k)
            for line in relas[k]:
                cclist|=set(line)
        allinfo = gdp_domain_democracy(' '.join(cclist))

        res = cut_node_depth * [[]]
        f = open(os.path.join(path, file), 'rU')
        for line in f:
            
            l = line.strip('\n').split('|')
            if len(l)>1 and line[0][0]!='(' and l[0]!='':
                if len(l[1])==0: continue
                
                l1 = l[0].count(' ')
                if l1<len(res):
                    l2 = gdp_domain_democracy(l[1]+' '+nonconnect)
                    l2 = [float(a)/float(b) for a,b in zip(l2, allinfo)]
                    res[l1].append(l2)

        with open(os.path.join(dsn_path,asname+'.json'), 'w') as df:
            json.dump(res, df)


def extract_connect_list_single(path, dsn_path, file_path):

    def gdp_domain_democracy(line):
        as_list = line.split(' ')
        res1 = 0
        res2 = 0
        res3 = 0
        for _as in as_list:
            if _as in country_importance:
                res1 += country_importance[_as][0]
                res2 += country_importance[_as][1]
                res3 += country_importance[_as][2]
        return [len(as_list),res1,res2,res3]

    with open('/data/lyj/shiyan_database/prefix_weight/country_importance/gdp_domain_democracy.json', 'r') as f:
        _country_importance = json.load(f)
    country_importance = {}
    for line in _country_importance:
        country_importance[line[0]] = line[1:]
    file_name = os.listdir(path)
    for _dir in os.listdir(file_path):
        if _dir not in os.listdir(dsn_path):
            os.popen('mkdir '+os.path.join(dsn_path, _dir))
        for file in os.listdir(os.path.join(file_path, _dir)):
            if file.find('.txt')==-1: continue
            asname = file.split('.')[0][9:]
            if asname+'.json' in os.listdir(dsn_path):
                print('exist')
            if file.split('.')[0]+'.cc_rela.json' not in file_name:
                continue
            with open(os.path.join(path, file.split('.')[0]+'.cc_rela.json'), 'rU') as f_rela:
                relas = json.load(f_rela)
            with open(os.path.join(path, file.split('.')[0]+'.nonconnect.json'), 'rU') as f_non:
                nonconnect = json.load(f_non)
            nonconnect = ' '.join(nonconnect)
            cclist = set()
            for k in relas:
                cclist.add(k)
                for line in relas[k]:
                    cclist|=set(line)
            allinfo = gdp_domain_democracy(' '.join(cclist))
            res = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
            f = open(os.path.join(file_path, _dir, file), 'rU')
            for line in f:
                l = line.strip('\n').split('|')
                if len(l)>1 and line[0][0]!='(' and l[0]!='':
                    if len(l[1])==0: continue
                    l1 = l[0].count(' ')
                    if l1<len(res):
                        l2 = gdp_domain_democracy(l[1]+' '+nonconnect)
                        l2 = [float(a)/float(b) for a,b in zip(l2, allinfo)]
                        res[l1].append(l2)
            with open(os.path.join(dsn_path,_dir, asname+'.json'), 'w') as df:
                json.dump(res, df)



def external_isp():
    # 计算边界AS下ISP属于和不所属本国的用户比例排名和安全性排名
    with open('/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/as-country-code.json', 'r') as f:
        encoder = json.load(f)
    decoder = {encoder[k]:k for k in encoder}
    with open('/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/result/v1/anova/sorted_country_gdp.json', 'r') as f:
        anova = json.load(f)
    with open('/home/peizd01/for_dragon/asns.json', 'r') as f:
        asns = json.load(f)
    asorg = {}
    for _as in asns:
        asorg[_as] = asns[_as]['country']['iso']

    security = {}
    cces = []
    user = {}

    for index in range(len(anova)):
        for a in anova[index]:
            security[decoder[a]] = index+1
    
    with open('/home/peizd01/for_dragon/ccInternal/public/rank.json', 'r') as f:
        rank = json.load(f)
    cces = list(rank.keys())
    
    path = 'F:/下一代互联网实验数据/ccExternal/user_influence/AUR/'
    file_name = os.listdir(path)
    for file in file_name:
        if file.find('AUR')==-1: continue
        cc = file.find('_')[0]
        if cc not in cces: continue
        with open(path+file, 'r') as f:
            aur = json.load(f)
        for _as in aur:
            user[_as+'-'+cc] = aur[_as]
    # 提取国家的结果排名
    # 【【属于本国边界AS用户比例】【不属于本国边界AS用户比例】【属于本国边界AS AUR列表】【不属于本国边界AS AUR列表】】
    res = {}
    for _cc in cces:
        res[_cc] = [[],[],[],[]]
    
    for _cc in res:
        _is,_not = 0, 0
        key = [_key.split('-')[1]  for _key in user if _cc in _key]
        # 计算用户比例
        for _as in key:
            if _as in asorg and asorg[_as] == _cc:
                _is+=user[_as+'-'+'_cc']
            elif _as in asorg and asorg[_as] != _cc:
                _not+=user[_as+'-'+'_cc']
        res[_cc][0] = _is/(_is+_not)
        res[_cc][1] = _not/(_is+_not)
    
    for _cc in res:
        # 计算AUR列表
        key = [_key.split('-')[1]  for _key in security if _cc in _key]
        for _as in key:
            if _as in asorg and asorg[_as] == _cc:
                res[_cc][2].append(security[_as+'-'+'_cc'])
            elif _as in asorg and asorg[_as] != _cc:
                res[_cc][3].append(security[_as+'-'+'_cc'])


# mbd == Ture 用来判断as是否属于且位于这个国家
def country_broadas_rank(file, dsn_file, user_path, encoder_file,_type, mbd=False):
    os.makedirs(dsn_file,exist_ok=True)
    def get_sum(n):
        r = 0
        for i in range(1,n+1):
            r += 1/i
        return r
    with open('/home/peizd01/for_dragon/ccInternal/public/rank.json', 'r') as f:
        ccInternal = json.load(f)
    with open(encoder_file, 'r') as f:
        encoder = json.load(f)
    decoder = {encoder[k]:k for k in encoder}

    with open('/home/peizd01/for_dragon/basicData/asn-iso.json', 'r') as f:
        asn_iso = json.load(f)
    with open(file, 'r') as f:
        print(file)
        sorted_broad = json.load(f)
    cbrank = {}
    fenmu = get_sum(len(sorted_broad)-1)
    for i in range(len(sorted_broad)):
        for m in sorted_broad[i]:
            a = decoder[m]
            _as, _cc = a.split('-')
            if _cc not in cbrank:
                cbrank[_cc] = {}
            # cbrank[_cc][_as] = (1/(i+1))/fenmu
            cbrank[_cc][_as] = len(sorted_broad)-(i+1)
    country_mean = {}
    for _cc in cbrank:
        # print(_cc)
        if _cc+'_AUR.json' in os.listdir(user_path):
            with open(os.path.join(user_path, _cc+'_AUR.json'), 'r') as f:
                aur = json.load(f)
            k = cbrank[_cc].keys()
            _min = 1
            _sum = 0
            for _as in k:
                if _as in aur:
                    if mbd and asn_iso.get(_as, None) != _cc: 
                        print(_as,asn_iso.get(_as, None) ,_cc)
                        continue
                    _min = min(_min, aur[_as])
                    cbrank[_cc][_as] = [cbrank[_cc][_as], aur[_as]]
                    _sum += cbrank[_cc][_as][-1]
            if _sum==0:
                if _cc in ccInternal:
                    print(_cc,'??')
                continue
            country_mean[_cc] = 0
            for _as in k:
                if isinstance(cbrank[_cc][_as], list):
                    country_mean[_cc] += cbrank[_cc][_as][-2]*cbrank[_cc][_as][-1]/_sum
                    # country_mean[_cc] += cbrank[_cc][_as][-2]*cbrank[_cc][_as][-1]
    
    country_mean = list(country_mean.items())
    country_mean = sorted(country_mean, key=(lambda x: x[1]), reverse=True)
    with open(os.path.join(dsn_file,f'{_type}.json'), 'w') as f:
        json.dump(country_mean, f)

def count_anova_rank_num(value):
    user_path = '/home/peizd01/for_dragon/ccExternal/user_influence/AUR/'
    file = '/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/v2/anova_nonconnect/sorted_country_'+value+'.json'
    encoder_file = '/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/as-country-code.json'
    with open('/home/peizd01/for_dragon/ccInternal/public/rank.json', 'r') as f:
        ccInternal = json.load(f)
    with open(encoder_file, 'r') as f:
        encoder = json.load(f)
    decoder = {encoder[k]:k for k in encoder}

    with open('/home/peizd01/for_dragon/basicData/asn-iso.json', 'r') as f:
        asn_iso = json.load(f)
    with open(file, 'r') as f:
        sorted_broad = json.load(f)
    cbrank = {}
    for i in range(len(sorted_broad)):
        for m in sorted_broad[i]:
            a = decoder[m]
            _as, _cc = a.split('-')
            if _cc not in cbrank:
                cbrank[_cc] = {}
            cbrank[_cc][_as] = i+1
    res = {}
    for _cc in cbrank:
        res[_cc] = {}
        if _cc+'_AUR.json' in os.listdir(user_path):
            with open(os.path.join(user_path, _cc+'_AUR.json'), 'r') as f:
                aur = json.load(f)
            k = cbrank[_cc].keys()
            for _as in k:
                if _as in asn_iso and asn_iso[_as]!=_cc: continue
                if _as in aur:
                    if cbrank[_cc][_as] not in res[_cc]:
                        res[_cc][cbrank[_cc][_as]] = 0
                    # res[_cc][cbrank[_cc][_as]] += 1
                    res[_cc][cbrank[_cc][_as]] += aur[_as]
    # for _cc in res:
    #     print(_cc, res[_cc][1]/sum(list(res[_cc].values())))
    with open('/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/v2/anova_nonconnect/mbd_anova_rank_user_ratio_'+value+'.json', 'w') as f:
        json.dump(res, f)





# extract_connect_list('/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/rtree/', '/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/count_num/')
# groud_truth_based_anova('/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/v2/count_num/', '/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/v2/anova/')
# anova_sort('/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/v2/anova/', 'democracy')

# # 对所有边界AS抗毁性排序
# second_order('/home/peizd01/for_dragon/pzd_External/globalCountryLabel/add_hidden_link/result/v2/count_num/', \
#     '/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/result/v2/anova/sorted_countrydemocracy165_195.json', \
#         '/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/result/v2/anova/sorted_country_democracy.json', \
#             'democracy')


# for value in ['gdp', 'domain', 'democracy']:
#     groud_truth_based_anova_single('/data/lyj/shiyan_database/ccExternal/globalCountryLabel/add_hidden_link/result/count_num_nonconnect_single_2', \
#         '/data/lyj/shiyan_database/ccExternal/globalCountryLabel/add_hidden_link/result/anova_nonconnect_single_2', \
#             value)


# # 以国家为单位，抗毁性排名
# country_broadas_rank('/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/result/v2/var_nonconnect/sorted_country_domain.json',\
#     '/home/peizd01/for_dragon/ccExternal/globalCountryLabel/add_hidden_link/result/v2/var_nonconnect/normalization/country_mbd_mean_rank_domain.json', \
#         '/home/peizd01/for_dragon/ccExternal/user_influence/AUR/', \
#             '/home/peizd01/for_dragonfor_dragon/ccExternal/globalCountryLabel/add_hidden_link/as-country-code.json', \
#                 True)

    
# for value in ['gdp', 'domain', 'democracy']:
#     count_anova_rank_num(value)