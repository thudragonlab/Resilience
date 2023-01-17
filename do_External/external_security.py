#!usr/bin/env python
# _*_ coding:utf8 _*_
import os
import json
import numpy as np
from scipy import stats
from other_script.my_types import *
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
    输入L:[[],...,[]],列表每个元素为边界AS所有的破坏性度量
    生成AS两两比对的结果
    '''
    l = [v for _,v in dict_l.items()]
    
    res = []
    nums,groups = [], []
    for k, v in dict_l.items():
        nums+=v
        groups += len(v)*[k]

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
 


def anova_sort(dsn_path, VALUE):
    '''
    dsn_path 排序结果存储路径
    VALUE 排序对比维度（只决定文件名）
    对之前生成的两两比对的结果进行最终排序
    '''
    reader = set()
    res = {}
    # 先读取所有两两对比的文件，整合成一个json
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

    # 没有比其更好的AS排第一
    for k in res:
        if len(res[k][0])==0:
            sorted_country[-1].append(k)
    
    # 从res中把已经记录过的连接从其他节点的比起更安全和无差异的国家列表中删掉
    while len(res):
        for cc in sorted_country[-1]:
            for _cc in res[cc][2]:
                res[_cc][0].remove(cc)
            for _cc in res[cc][1]:
                res[_cc][1].remove(cc)
            del res[cc]
        temp = []
        # 没有比其更好的AS排第一
        for cc in res:
            if len(res[cc][0])==0:
                temp.append(cc)
        flag = 0
        sorted_country.append(temp)
    with open(os.path.join(dsn_path, 'sorted_country_'+VALUE+'.json'), 'w')  as f:
        json.dump(sorted_country, f)



def groud_truth_based_anova_single(path, dsn_path,value):
    _dir = os.listdir(path)
    value_dict = {'gdp':0, 'domain':1, 'democracy':2}
    N = 30
    begin = -1
    l = {}
    for _dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, _dir)):
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
    STEP1 通过anova分析,输入各国的全部连通性度量,看是否有显著性差异。
    STEP2 如有，判断国家两两间差异。
    STEP3 对国家进行排序
    '''
    path = os.path.join(source_path,'result','count_num')
    dsn_path = os.path.join(source_path,'result','anova')
    os.makedirs(dsn_path,exist_ok=True)
    file_name = os.listdir(path)
    gdp, domain, democracy = 0,1,2
    # 每30个排一次序
    N = 30
    begin = 0
    while begin<len(file_name):
        l = {}
        end = begin+N
        a = time.time()
        for file in file_name[begin:end]:
            
            _l = []
            with open(os.path.join(path, file), 'r') as f:
                r = json.load(f)
            for i in r:
                _l += [_i[democracy] for _i in i]

            l[file.split('.')[0]] = _l
        anova(l, dsn_path, 'democracy', begin, end)
        begin += int(N/2)
        print(end,' create')
        b = time.time()
    anova_sort(dsn_path, 'democracy')



def judge_var(target_list, result):
    '''
    result 存储最终结果
    target_list 剩余没有排名的数据

    递归查找target_list中第一个元素,用levene方法计算是否有显著性差异,如果有就加入result
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



def groud_truth_based_var(source_path,):
    '''
    STEP1 输入各国的全部连通性度量
    STEP2 对国家连通性度量方差进行排序
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
    # 整理count_num数据到字典中
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
        begin += int(N/2)
        print(end,' create')
        b = time.time()
    var_list = list(var_result.values())
    
    # 区分方差是0的数据
    var_zero_list =  list(filter(lambda x:np.var(x['list']) == 0.0,var_list))
    var_no_zero_list = list(filter(lambda x:np.var(x['list']) != 0.0,var_list))
    var_no_zero_list.sort(key=lambda x: np.var(x['list']))
    # 方差为0的排在第一位
    result.append(list(map(lambda x:x['key'],var_zero_list)))
    # 区分方差不是0的数据进行排序，结果存在var_no_zero_list
    judge_var(var_no_zero_list, result)
    with open(os.path.join(dsn_path, 'sorted_country_%s.json' % _type), 'w') as sorted_var_f:
        json.dump(result, sorted_var_f)


def extract_connect_list(source_path,cut_node_depth):
    '''
    source_path 根路径
    gdp_domain_democracy_path 权重路径
    cut_node_depth 破坏节点数量
    生成count_num文件夹
    '''
    path = os.path.join(source_path,'monitor')
    dsn_path = os.path.join(source_path,'result','count_num')
    json_path = os.path.join(source_path,'json')
    os.makedirs(dsn_path,exist_ok=True)
    def gdp_domain_democracy(line) -> List:
        '''
        line 国家列表，用空格隔开
        '''
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

    # 读取权重数据
    with open('static/gdp_domain_democracy.json', 'r') as f:
        _country_importance = json.load(f)
    country_importance:Dict[COUNTRY_CODE,List] = {}
    # 生成权重字典
    for line in _country_importance:
        country_importance[line[0]] = line[1:]
    file_name = os.listdir(path)
    json_file_list = os.listdir(json_path)
    for file in file_name:
        # file 是.addDel文件
        if file.find('.txt')==-1: continue
        
        asname:AS_CODE = file.split('.')[0][9:]
        with open(os.path.join(json_path, file.split('.')[0]+'.cc_rela.json'), 'rU') as f_rela:
            relas = json.load(f_rela)
        with open(os.path.join(json_path, file.split('.')[0]+'.nonconnect.json'), 'rU') as f_non:
            nonconnect = json.load(f_non)
        nonconnect = ' '.join(nonconnect)
        cclist:Set[COUNTRY_CODE] = set()
        for k in relas:
            cclist.add(k)
            for line in relas[k]:
                # 求ccList和line的并集并赋值给ccList
                cclist|=set(line)

        #国家列表以及对应的权重信息    
        allinfo = gdp_domain_democracy(' '.join(cclist))
        print(allinfo)
        # 生成 最多破坏节点数量 个数组
        res = max(cut_node_depth) * [[]]
        f = open(os.path.join(path, file), 'rU')
        # 处理破坏结果文件
        for line in f:
            
            l = line.strip('\n').split('|')
            if len(l)>1 and line[0][0]!='(' and l[0]!='':
                # 破坏节点数量
                l1 = l[0].count(' ')
                if l1<len(res):
                    # 通过受影响的国家和不连通的国家，计算权重数据
                    l2 = gdp_domain_democracy(l[1]+' '+nonconnect)
                    # 当前受影响的国家权重数据/所有受影响国家的权重数据
                    l2 = [float(a)/float(b) for a,b in zip(l2, allinfo)]
                    res[l1].append(l2)

        with open(os.path.join(dsn_path,asname+'.json'), 'w') as df:
            json.dump(res, df)

def country_broadas_rank(file, dsn_file, user_path, encoder_file,_type):
    '''
    file 排名数据
    dsn_file 权重数据
    user_path user_influence/AUR数据
    encoder_file 新生成的asn数据
    _type anova排名 or var排名

    计算最终排名
    '''
    os.makedirs(dsn_file,exist_ok=True)
    
    # 读取新生成的as号数据，生成字典
    with open(encoder_file, 'r') as f:
        encoder = json.load(f)
    decoder = {encoder[k]:k for k in encoder}

    # 读取AS地理位置数据
    with open('static/asn-iso.json', 'r') as f:
        asn_iso = json.load(f)
    # 地区排名数据
    with open(file, 'r') as f:
        sorted_broad = json.load(f)
    cbrank:Dict[COUNTRY_CODE,Dict[AS_CODE,int]] = {}
    # 遍历排名数据
    # i是排名
    for i in range(len(sorted_broad)):
        for m in sorted_broad[i]:
            # 转换为真实AS-country
            a = decoder[m]
            _as, _cc = a.split('-')
            # country加入cbrank字典
            if _cc not in cbrank:
                cbrank[_cc] = {}
            # 生成倒序排名 比如一共5名，第一名是倒数第五名
            cbrank[_cc][_as] = len(sorted_broad)-(i+1)
    country_mean = {}
    for _cc in cbrank:
        if _cc+'_AUR.json' in os.listdir(user_path):
            # 读取路由影响力数据
            with open(os.path.join(user_path, _cc+'_AUR.json'), 'r') as f:
                aur:Dict[AS_CODE,float] = json.load(f)
            k:List[AS_CODE] = list(cbrank[_cc].keys())
            _sum = 0
            for _as in k:
                if _as in aur:
                    cbrank[_cc][_as] = [cbrank[_cc][_as], aur[_as]]
                    # 对AUR的数据求和
                    _sum += cbrank[_cc][_as][-1]
            if _sum==0:
                continue
            country_mean[_cc] = 0
            for _as in k:
                if isinstance(cbrank[_cc][_as], list):
                    print('cbrank[_cc][_as]',cbrank[_cc][_as])
                    # 排名 * AUR的数据 / AUR数据总和就是最终排名
                    country_mean[_cc] += cbrank[_cc][_as][-2]*cbrank[_cc][_as][-1]
    
    country_mean = list(country_mean.items())
    country_mean = sorted(country_mean, key=(lambda x: x[1]), reverse=True)
    result = {}
    for i in range(len(country_mean)):
        result[country_mean[i][0]] = {
            'rank':i + 1,
            'weight':country_mean[i][1]

        }
    with open(os.path.join(dsn_file,f'{_type}.json'), 'w') as f:
        json.dump(result, f)
