#!usr/bin/env python
# _*_ coding:utf8 _*_

import os
import json
import copy
import matplotlib.pyplot as plt
from collections import deque, Counter
from typing import List, Tuple
import time
import numpy as np
import itertools
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import random

# only_cc_list = ['AF', 'PH']
'''
传输AS影响+AS霸权思想
上面都是想法一的处理过程，想法二由增加了计算国内所有AS的数据，所以得重新写所有相关代码
'''

N = 10
class user_radio_v2():

    def __init__(self):
        pass

    @staticmethod
    def extract_country_broad_path(Path, dsn_path):
        res = {}
        ccas = {}
        ccasfile = os.listdir(prefix + '/ccInternal/cc2as')
        for _file in ccasfile:
            with open(prefix + '/ccInternal/cc2as/' + _file, 'r') as f:
                ccas[_file[:2]] = json.load(f)

        for path in Path:
            file_name = os.listdir(path)
            for file in file_name:
                try:
                    with open(path + file, 'r') as f:
                        reader = json.load(f)
                except:
                    print(file + ' error')
                for aslist in reader:
                    value = reader[aslist][0].split('|')
                    value[1] = value[1].split(' ')
                    value[2] = value[2].split(' ')
                    tempas = []
                    tempcc = []
                    for i in range(len(value[2])):
                        value[2][i] = value[2][i].split('-')
                        for cc in value[2][i]:
                            tempas.append(value[1][i])
                            tempcc.append(cc)
                    # c = Counter(tempcc).most_common()

                    # if len(c) == 1 or (len(c) == 2 and 'None' in tempcc):
                    #     continue

                    bindex = 0  #记录观测点
                    for bindex in range(len(tempcc)):
                        if tempcc[bindex] != 'None' and tempas[bindex] != 'None':
                            break
                    for i in range(len(tempcc) - 1):
                        if tempcc[i] != tempcc[i+1] and 'None' not in tempcc[i:i+2]\
                                and 'None' not in tempas[i:i+2]:
                            if tempcc[i] not in ccas:
                                print('not have ' + tempcc[i])
                                continue
                            if tempcc[i] not in res:
                                res[tempcc[i]] = {}
                            if tempas[bindex] not in res[tempcc[i]]:
                                res[tempcc[i]][tempas[bindex]] = {}
                            _bindex = i
                            while i >= 0 and (tempcc[_bindex] == tempcc[i] or
                                              (tempcc[_bindex] == 'None' and tempcc[_bindex] in ccas[tempcc[i]])):
                                _bindex -= 1
                            _bindex += 1
                            if ' '.join(tempas[_bindex:i + 2]) + '|' + ' '.join(tempcc[_bindex:i + 2]) not in \
                                    res[tempcc[i]][tempas[bindex]]:
                                res[tempcc[i]][tempas[bindex]][' '.join(tempas[_bindex:i + 2]) + '|' +
                                                               ' '.join(tempcc[_bindex:i + 2])] = 0
                            res[tempcc[i]][tempas[bindex]][' '.join(tempas[_bindex:i + 2]) + '|' +
                                                           ' '.join(tempcc[_bindex:i + 2])] += reader[aslist][1]
        for cc in res:
            with open(dsn_path + cc + '.json', 'w') as f:
                json.dump(res[cc], f)

    @staticmethod
    def calculate_ratio(Path, dsn_path):
        #从上面函数提取的出国路由中，提取出每个边界AS在各个观测点出现的次数、观测点的总路径、观测点观察到的AS数量
        #每个观测点记录：‘broadAS’broadAS信息：(‘num’:各个broad AS的次数, 'interASnum'各个broadAS出现国内AS出现的数量)、'pathSum'观察的总路径、'asSum'观测点的AS数量、
        file_name = os.listdir(Path)
        for file in file_name:
            # try:
            with open(Path + file, 'r') as f:
                reader = json.load(f)
            # except:
            #     print(file)
            #     continue
            res = {}
            res['all'] = {}
            res['all']['asSum'] = set()
            for a in reader:
                res[a] = {}
                res[a]['broadAS'] = {}
                res[a]['broadAS']['num'] = {}
                res[a]['broadAS']['interASnum'] = {}
                res[a]['pathSum'] = 0
                res[a]['asSum'] = set()
                for path in reader[a]:
                    aslist = path.split('|')[0].split(' ')
                    cclist = path.split('|')[1].split(' ')
                    if len(aslist) < 2: continue
                    if aslist[-2] not in res[a]['broadAS']['num']:
                        res[a]['broadAS']['num'][aslist[-2]] = 0
                    res[a]['broadAS']['num'][aslist[-2]] += reader[a][path]
                    res[a]['pathSum'] += reader[a][path]
                    res[a]['asSum'] = res[a]['asSum'] | set(aslist)
                    res['all']['asSum'] = res['all']['asSum'] | set(aslist)

                    # 增加的部分 记录国内AS的次数
                    if aslist[-2] not in res[a]['broadAS']['interASnum']:
                        res[a]['broadAS']['interASnum'][aslist[-2]] = {}

                    for index in range(len(aslist) - 2):
                        if a == aslist[index] or aslist[index] == 'None': continue
                        if aslist[index] not in res[a]['broadAS']['interASnum'][aslist[-2]]:
                            res[a]['broadAS']['interASnum'][aslist[-2]][aslist[index]] = 0
                        res[a]['broadAS']['interASnum'][aslist[-2]][aslist[index]] += reader[a][path]

                res[a]['asSum'] = len(res[a]['asSum'])
            res['all']['asSum'] = len(res['all']['asSum'])

            with open(dsn_path + file, 'w') as f:
                json.dump(res, f)

    @staticmethod
    def user_radio(path, dsn_path, alpha1, alpha2):
        file_name = os.listdir(path)
        res = {}
        for file in file_name:
            res[file[:2]] = {}
            with open(path + file, 'r') as f:
                reader = json.load(f)
                f.close()

            if file not in os.listdir(prefix + '/prefix_weight/as_importance/'):
                print(file + ' not have user ratio')
                continue

            with open(prefix + '/prefix_weight/as_importance/' + file, 'r') as f:
                as_importance = json.load(f)
            uratio = {}
            for line in as_importance:
                uratio[line[0]] = line[1]

            # 对于每个测量点
            # 1、先计算保存各个边界AS的BC值：‘BC’
            # 2、计算每个边界AS下，国内AS经过这个边界AS的比例：‘interASratio’
            # 3、计算vp观测到的边界AS比例：‘bdASratio’

            vp = {}
            bdAS = set()
            interAS = set()
            for _vp in reader:
                if len(reader[_vp]) <= 2: continue
                vp[_vp] = {}
                vp[_vp]['BC'] = {}
                vp[_vp]['interASratio'] = {}
                vp[_vp]['bdASratio'] = len(reader[_vp]['broadAS']['num'])

                bdAS |= set(reader[_vp]['broadAS']['interASnum'].keys())

                for bdas in reader[_vp]['broadAS']['interASnum']:
                    vp[_vp]['interASratio'][bdas] = {}
                    for _interas in reader[_vp]['broadAS']['interASnum'][bdas]:
                        vp[_vp]['interASratio'][bdas][_interas] = reader[_vp]['broadAS']['interASnum'][bdas][_interas] / sum(
                            list(reader[_vp]['broadAS']['interASnum'][bdas].values()))
                    interAS |= set(reader[_vp]['broadAS']['interASnum'][bdas].keys())

                    vp[_vp]['BC'][bdas] = reader[_vp]['broadAS']['num'][bdas] / reader[_vp]['pathSum']

            for _vp in vp:
                vp[_vp]['bdASratio'] = vp[_vp]['bdASratio'] / len(bdAS)

            # 对每个边界AS
            # 1、提取每个观测点下国内AS的比例
            # 2、每个观测点下，该边界AS的BC值
            # 3、按照alpha过滤边界AS的观测点信息
            # 4、计算边界AS对国内AS的传输影响
            # 5、国内AS获取用户比例，计算边界AS安全性

            res = {}
            for _bdas in bdAS:
                res[_bdas] = {}

            for _vp in vp:
                for _bdas in vp[_vp]['BC']:
                    res[_bdas][_vp] = {}
                    res[_bdas][_vp]['BC'] = vp[_vp]['BC'][_bdas]
                for _bdas in vp[_vp]['interASratio']:
                    res[_bdas][_vp]['interASratio'] = vp[_vp]['interASratio'][_bdas]

            # 存储UR(_baas,_interas)
            Res = {}
            # 存储AUR
            AUR = {}
            for _bdas in res:
                Res[_bdas] = {}
                AUR[_bdas] = 0
                for _interas in interAS:
                    Res[_bdas][_interas] = 0

                    # 获取一个List：某个观测点下BC值，路径比例，测量点观察到的边界AS比例
                    temp = []
                    for _vp in res[_bdas]:
                        temp.append([])
                        temp[-1].append(res[_bdas][_vp]['BC'])
                        if _interas in res[_bdas][_vp]['interASratio']:
                            temp[-1].append(res[_bdas][_vp]['interASratio'][_interas])
                        else:
                            temp[-1].append(0.0)
                        temp[-1].append(vp[_vp]['bdASratio'])
                    temp = sorted(temp, key=(lambda x: x[0]), reverse=True)
                    bindex = int((len(temp) - 1) * alpha1)
                    eindex = int((len(temp) - 1) * alpha2)

                    #计算UR(_baas,_interas)
                    for i in range(bindex, eindex + 1):
                        # Res[_bdas][_interas] += temp[i][-1]*temp[i][-2]
                        Res[_bdas][_interas] += max(temp[i][-2], 0.05)

                    Res[_bdas][_interas] = Res[_bdas][_interas] / (eindex - bindex + 1)

                    if _interas in uratio:
                        AUR[_bdas] += uratio[_interas] * Res[_bdas][_interas]

            with open(dsn_path + file[:2] + '_UR.json', 'w') as f:
                json.dump(Res, f)
            with open(dsn_path + file[:2] + '_AUR.json', 'w') as f:
                json.dump(AUR, f)


'''
分析国家边界AS的安全性
STEP1 计算国家边界AS的routingTree，加入各个国家内部的拓扑，以及各个国家边界之间的拓扑
STEP2 去除国家内部的边，只保留国家边界之间的连接，增加国家内部国家边界之间连接的虚拟边
STEP3 模拟断开
'''


class broad_as_routingtree():

    def __init__(self,source_path,cc_list,as_rela_file):
        self.file_name = os.path.join(source_path,'as_rela_code.txt')
        self.as_country_code_path = os.path.join(source_path, 'as-country-code.json')
        self.path = os.path.join(source_path,'cc2as')
        self.only_cc_list = cc_list
        self.code = 1
        self.cal_rtree_code_v2_path = os.path.join(source_path,'cal_rtree_code_v2.json')
        self.as_rela_file = as_rela_file
        with open(self.file_name, 'w') as f:
            f.write('')
        self.country_code = {}
        self.make_country_code()
        self.add_internal_link()
        self.add_external_link()
        print(self.as_country_code_path)
        with open(self.as_country_code_path, 'w') as f:
            json.dump(self.country_code, f)

    def make_country_code(self):
        cc2as = os.listdir(self.path)
        for cc_file in cc2as:
            with open(os.path.join(self.path, cc_file), 'r') as f:
                ases = json.load(f)
            cc = cc_file[:-5]
            for a in ases:
                self.country_code[a + '-' + cc] = str(self.code)
                self.code += 1

    def as2code(self, a):
        if a not in self.country_code:
            a, b = a.split('-')
            with open(os.path.join(self.path, b + '.json'), 'r') as f:
                r = json.load(f)
            r.append(a)
            with open(os.path.join(self.path, b + '.json'), 'w') as f:
                json.dump(r, f)
            # print(a, b)
            self.country_code[a + '-' + b] = str(self.code)
            self.code += 1
            # self.country_code[a + '-' + b] = 'new'
            return self.country_code[a + '-' + b]
        return self.country_code[a]

    def add_internal_link(self):
        with open(os.path.join(self.as_rela_file), 'r') as f:
            # with open('/data3/lyj/shiyan_database/basicData/as_rela_from_asInfo.json', 'r') as f:
            as_rela = json.load(f)

        cc2as = os.listdir(self.path)
        print(self.file_name)
        with open(self.file_name, 'a') as f_dsn:
            for cc_file in cc2as:
                with open(os.path.join(self.path, cc_file), 'r') as f:
                    ases = json.load(f)
                    cc = cc_file[:-5]
                    
                    for c in ases:
                        if c in as_rela:
                            temp = [i for i in as_rela[c][0] if i in ases]
                            for b in temp:
                                if c <= b:
                                    if c == 'None' or b == 'None': continue
                                    f_dsn.write(
                                        self.as2code(str(c) + '-' + cc) + '|' + self.as2code(str(b) + '-' + cc) + '|0\n')
                            temp = [i for i in as_rela[c][1] if i in ases]
                            for b in temp:
                                if c == 'None' or b == 'None' : continue
                                f_dsn.write(self.as2code(str(c) + '-' + cc) + '|' + self.as2code(str(b) + '-' + cc) + '|-1\n')
            f_dsn.close()

    def add_external_link(self):
        with open(self.as_rela_file, 'r') as f:
            # with open('/data3/lyj/shiyan_database/basicData/as_rela_from_asInfo.json', 'r') as f:
            as_rela = json.load(f)

        file_name = os.listdir('/home/peizd01/for_dragon/beginAsFile/broadPath_v2/')
        temp = []
        for file in file_name:
            with open('/home/peizd01/for_dragon/beginAsFile/broadPath_v2/' + file, 'r') as ff:
                reader = json.load(ff)
            for k in reader:
                for i in reader[k].keys():
                    a_c, b_c = i.split('|')[1].split(' ')
                    # print(a_c,b_c)
                    if a_c not in self.only_cc_list or b_c not in self.only_cc_list:
                        continue
                    # temp+=list(reader[k].keys())
                    temp.append(i)
                # print(temp)
        with open('/home/peizd01/for_dragon/ccExternal/hiddenLink/hidden_link.json', 'r') as ff:
            reader = json.load(ff)
        for k in reader:
            a1, c1, a2, c2 = k.split(' ')[0].split('-')[1], k.split(' ')[0].split('-')[0], k.split(' ')[1].split(
                '-')[1], k.split(' ')[1].split('-')[0]
            if 'None' in [a1, c1, a2, c2] or not a1.isdigit() or not a2.isdigit(): continue
            temp.append(a1 + ' ' + a2 + '|' + c1 + ' ' + c2)
        temp = set(temp)
        with open(self.file_name, 'a') as f:
            for line in temp:
                a, b = line.split('|')[0].split(' ')
                a_cc, b_cc = line.split('|')[1].split(' ')
                if 'None' in [a, a_cc, b, b_cc] or not a.isdigit() or not b.isdigit(): continue
                if a_cc == '' or b_cc == '' or self.as2code(str(a) + '-' + a_cc) == 'none' or self.as2code(str(b) + '-' + b_cc) == 'none':
                    continue
                if a == b:
                    f.write(self.as2code(str(a) + '-' + a_cc) + '|' + self.as2code(str(b) + '-' + b_cc) + '|0\n')
                    f.write(self.as2code(str(a) + '-' + a_cc) + '|' + self.as2code(str(b) + '-' + b_cc) + '|-1\n')
                    f.write(self.as2code(str(b) + '-' + b_cc) + '|' + self.as2code(str(a) + '-' + a_cc) + '|-1\n')
                else:
                    if b in as_rela and a in as_rela[b][0]:
                        f.write(self.as2code(str(a) + '-' + a_cc) + '|' + self.as2code(str(b) + '-' + b_cc) + '|0\n')
                    elif b in as_rela and a in as_rela[b][1]:
                        f.write(self.as2code(str(a) + '-' + a_cc) + '|' + self.as2code(str(b) + '-' + b_cc) + '|-1\n')
                    elif b in as_rela and a in as_rela[b][1]:
                        f.write(self.as2code(str(b) + '-' + b_cc) + '|' + self.as2code(str(a) + '-' + a_cc) + '|-1\n')
                    else:
                        if a not in as_rela and b not in as_rela:
                            f.write(self.as2code(str(b) + '-' + b_cc) + '|' + self.as2code(str(a) + '-' + a_cc) + '|0\n')
                        elif a not in as_rela:
                            f.write(self.as2code(str(b) + '-' + b_cc) + '|' + self.as2code(str(a) + '-' + a_cc) + '|-1\n')
                        elif b not in as_rela:
                            f.write(self.as2code(str(a) + '-' + a_cc) + '|' + self.as2code(str(b) + '-' + b_cc) + '|-1\n')
                        else:
                            a_c_len, b_c_len = len(as_rela[a][1]), len(as_rela[b][1])
                            mvalue = min(a_c_len, b_c_len) if min(a_c_len, b_c_len) > 0 else 0.01
                            if abs(a_c_len - b_c_len) / mvalue < 0.1:
                                f.write(self.as2code(str(b) + '-' + b_cc) + '|' + self.as2code(str(a) + '-' + a_cc) + '|0\n')
                            elif a_c_len > b_c_len:
                                f.write(self.as2code(str(a) + '-' + a_cc) + '|' + self.as2code(str(b) + '-' + b_cc) + '|-1\n')
                            else:
                                f.write(self.as2code(str(b) + '-' + b_cc) + '|' + self.as2code(str(a) + '-' + a_cc) + '|-1\n')

            f.close()

    # @staticmethod
    def cal_rtree_code(self):
        with open(self.as_country_code_path, 'r') as f:
            code = json.load(f)
        res = []
        Res = []
        file_name = os.listdir('/home/peizd01/for_dragon/beginAsFile/broadPath_v2/')
        for file in file_name:
            with open('/home/peizd01/for_dragon/beginAsFile/broadPath_v2/' + file, 'r') as f:
                r = json.load(f)
            for p in r:
                for line in r[p]:
                    a1, a2 = line.split('|')[0].split(' ')
                    c1, c2 = line.split('|')[1].split(' ')
                    if c1 == '' or c2 == '' or f'{a1}-{c1}' not in code or c1 not in self.only_cc_list: continue
                    res.append(a1 + '-' + c1)
        with open('/home/peizd01/for_dragon/ccExternal/hiddenLink/hidden_link.json', 'r') as ff:
            reader = json.load(ff)
        for k in reader:
            a1, c1, a2, c2 = k.split(' ')[0].split('-')[1], k.split(' ')[0].split('-')[0], k.split(' ')[1].split(
                '-')[1], k.split(' ')[1].split('-')[0]
            if c1 == '' or c2 == '' or c1 not in self.only_cc_list: continue
            res.append(a1 + '-' + c1)
        # print(len(set(res)))
        # with open('E:/Study/实验室/沙盘最新数据/mergeIP_1013/国家边界-相同AS-1020.json', 'r') as f:
        #     r2 = json.load(f)
        # for line in r2:
        #     if line['begin']['CY'] != '' and line['end']['CY'] != '':
        #         res.append(line['begin']['AS'] + '-' + line['begin']['CY'])
        #         res.append(line['end']['AS'] + '-' + line['end']['CY'])

        res = list(set(res))
        print('len(res)',len(res))
        for i in res:
            Res.append(code[i])

        with open(self.cal_rtree_code_v2_path, 'w') as f:
            json.dump(Res, f)

    # @staticmethod
    def remove_cc_internal_link(self,source_path):
        
        encoder_path = os.path.join(source_path,'as-country-code.json')
        npz_path = os.path.join(source_path,'rtree')

        with open(encoder_path, 'r') as f:
            encoder = json.load(f)
        incoder = {encoder[i]: i for i in encoder}
        file_name = os.listdir(npz_path)
        for file in file_name:
            # 过滤条件，只破坏我想破坏的

            # if file.find('2.npz')==-1: continue
            remove_internal_link(source_path, incoder, file)

def remove_internal_link(source_path, incoder, file):
    npz_path = os.path.join(source_path,'rtree')
    json_path = os.path.join(source_path,'json')
    os.makedirs(json_path, exist_ok=True)
    
    def resolve(s):
            return [s.split('-')[0], s.split('-')[1]]
    broad = []
    m = np.load(os.path.join(npz_path, file))
    link = list(zip(m['row'], m['col']))

        # 第一次遍历，存储边界AS代码
    for a, b in link:
        as1, cy1 = resolve(incoder[str(a)])
        as2, cy2 = resolve(incoder[str(b)])
        if cy1 != cy2 and 'None' not in [as1, as2, cy1, cy2]:
                # if cy1!=cy2:
            broad.append(str(a))
            broad.append(str(b))

    for index in range(len(link) - 1, -1, -1):
        as1, cy1 = resolve(incoder[str(link[index][0])])
        as2, cy2 = resolve(incoder[str(link[index][1])])
        if str(link[index][0]) in broad and str(link[index][1]) in broad:
            link[index] = [incoder[str(link[index][0])], incoder[str(link[index][1])]]
        else:
            link.pop(index)

        # 记录link列表（国家A-》国家B）：[link]
    cc_pair_link = {}
        # 以国家为单位，找到前向国家，后向国家
    cc_rela = {}
    for a, b in link:
        as1, cy1 = resolve(a)
        as2, cy2 = resolve(b)
        if cy1 not in cc_rela:
            cc_rela[cy1] = [set(), set()]
        if cy2 not in cc_rela:
            cc_rela[cy2] = [set(), set()]
        cc_rela[cy1][1].add(cy2)
        cc_rela[cy2][0].add(cy1)

        if cy1 + ' ' + cy2 not in cc_pair_link:
            cc_pair_link[cy1 + ' ' + cy2] = set()
        cc_pair_link[cy1 + ' ' + cy2].add(as1 + ' ' + as2)

    for key in cc_pair_link.keys():
        cc_pair_link[key] = list(cc_pair_link[key])
    for key in cc_rela:
        cc_rela[key][0] = list(cc_rela[key][0])
        cc_rela[key][1] = list(cc_rela[key][1])

    with open(os.path.join(json_path, file[:-4] + '.cc_pair_link.json'), 'w') as f:
        json.dump(cc_pair_link, f)
    with open(os.path.join(json_path, file[:-4] + '.cc_rela.json'), 'w') as f:
        json.dump(cc_rela, f)

        # for key in cc_pair_link:
        #     a, b = key.split(' ')
        #     if a not in cc_rela[b][0] or b not in cc_rela[a][1]:
        #         print('error')


class monitor_remove_as():

    def __init__(self, file_path, dsn_path, specific_country=None):
        self.file_path = file_path
        self.n_as = N
        self.dsn_path = dsn_path
        self.specific_country = specific_country

    def remove_country_link(self, queue):
        res = []
        # 删除边
        tempQ = []
        for link in queue:
            a, b = link
            self.cc_rela[a][1].remove(b)
            self.cc_rela[b][0].remove(a)
            if len(self.cc_rela[a][1]) == 0 or self.cc_rela[a][1] == [a]:
                tempQ.append(a)
        for link in queue:
            a, b = link
            self.cc_rela[a][1].append(b)
            self.cc_rela[b][0].append(a)

        # 删除点
        for node in tempQ:
            for i in self.cc_rela[node][1]:
                self.cc_rela[i][0].remove(node)
            self.cc_rela[node][1] = []
        while tempQ:
            n = tempQ.pop(0)
            res.append(n)
            if n not in self.cc_rela: continue
            for i in self.cc_rela[n][0]:
                self.cc_rela[i][1].remove(n)
                if len(self.cc_rela[i][1]) == 0 or self.cc_rela[i][1] == [i]: tempQ.append(i)
            del self.cc_rela[n]
        return res

    def _remove_as(self, node):
        remove_cc_link = []
        for key in self.cc_pair_link:
            flag = 1
            for line in self.cc_pair_link[key]:
                a, b = line.split(' ')
                if a not in node and b not in node:
                    flag = 0
                    break
            if flag:
                remove_cc_link.append(key.split(' '))
        return self.remove_country_link(remove_cc_link)

    def remove_as(self, file):
        # print(self.file_path)
        print(file)
        with open(os.path.join(self.file_path, file), 'r') as f:
            self.cc_rela = json.load(f)
        with open(os.path.join(self.file_path, file.split('.')[0] + '.cc_pair_link.json'), 'r') as f:
            self.cc_pair_link = json.load(f)
        nodelist = set()
        for key in self.cc_pair_link:
            for line in self.cc_pair_link[key]:
                nodelist.add(line.split(' ')[0])
                nodelist.add(line.split(' ')[1])
        nodelist = list(nodelist)

        # 挑选出只有某个国家的nodelist
        with open(prefix + '/basicData/asn-iso.json', 'r') as f:
            asn_iso = json.load(f)

        if self.specific_country:
            for i in range(len(nodelist) - 1, -1, -1):
                if nodelist[i] not in asn_iso or asn_iso[nodelist[i]] != self.specific_country:
                    nodelist.pop(i)

        # print(len(nodelist))

        if len(nodelist) < 500:
            epoch = len(nodelist)
        else:
            epoch = 500

        node2link = {}
        for node in nodelist:
            remove_cc_link = []
            for key in self.cc_pair_link:
                flag = 1
                for line in self.cc_pair_link[key]:
                    a, b = line.split(' ')
                    if a not in node and b not in node:
                        flag = 0
                        break
                if flag:
                    remove_cc_link.append(key.split(' '))
            if len(remove_cc_link):
                node2link[node] = remove_cc_link
        # if len(nodelist) <= self.n_as:
        if len(nodelist) <= 0:
            return
        with open(os.path.join(self.dsn_path, file.split('.')[0] + '.del.txt'), 'w') as dsn_f:
            # for num in range(3, self.n_as):
            for num in range(1, 2):
                # num = 1
                flag = 0
                epoch = len(nodelist) * num
                while flag < epoch:
                    flag += 1
                    node = random.sample(nodelist, num)
                    node = list(set(list(node)))
                    node.sort()
                    with open(os.path.join(self.file_path, file), 'r') as f:
                        self.cc_rela = json.load(f)
                    res = []
                    linklist = []
                    for _node in node:
                        if _node in node2link:
                            for __node in node2link[_node]:
                                if __node not in linklist:
                                    linklist.append(__node)
                    if linklist:
                        res = self.remove_country_link(linklist)
                    # res = self._remove_as(node)
                    print(node)
                    dsn_f.write(' '.join(node) + '|' + ' '.join(res) + '\n')
                # break

        print(file + ' success')
        # size = 1000
        # flag = size-1
        # for i in itertools.permutations(nodelist, num):
        #     if flag>=size-1:
        #         choise = np.random.uniform(0, len(nodelist)**num+1, size)
        #         flag = -1
        #     flag += 1
        #     if choise[flag]<=len(nodelist):
        #         node = list(set(list(i)))
        #         node.sort()
        #         # print(node)
        #         with open(os.path.join(self.file_path, file), 'r') as f:
        #             self.cc_rela = json.load(f)
        #         res = self._remove_as(node)
        #         dsn_f.write(' '.join(node)+'|'+' '.join(res)+'\n')





def _monitor_remove_as(source_path,cut_node_depth):
    global N
    N = cut_node_depth
    file_path = os.path.join(source_path,'json')
    dsn_path = os.path.join(source_path,'monitor')
    os.makedirs(file_path,exist_ok=True)
    os.makedirs(dsn_path,exist_ok=True)

    def pool_monitor_as(file):
        # N = 10
        if file.split('.')[0] + '.del.txt' in os.listdir(dsn_path):
            print(file + ' exist')
            return
        mra = monitor_remove_as(file_path, dsn_path, None)
        mra.remove_as(file)
        
    file_name = os.listdir(file_path)
    file_run = [[]]
    for file in file_name:
        if file.find('cc_rela') == -1: continue
        if file.split('.')[0] + '.cc_pair_link.json' not in file_name: continue
        if file.split('.')[0] + '.del.txt' in os.listdir(dsn_path):
            print(file + ' exist')
            continue
        if len(file_run[-1]) < 6:
            file_run[-1].append(file)
        else:
            file_run.append([file])
    #file_run.reverse()
    for _file_run in file_run:
        pool = ThreadPool()
        pool.map(pool_monitor_as, _file_run)
    pool.close()
    pool.join()


# 计算出边界AS routingTree中本身没有通信的国家
def create_nonconnect(source_path):
    exist = [
        "PG", "HN", "GP", "PR", "LU", "KG", "CG", "GG", "ST", "ME", "CN", "TT", "KY", "BV", "GR", "MQ", "VU", "IT", "UA", "EG",
        "BS", "PK", "RE", "LR", "MF", "PM", "CW", "BN", "VA", "BD", "UG", "BT", "IL", "SN", "HK", "KW", "HT", "JM", "AT", "SC",
        "IE", "BZ", "UY", "ES", "SZ", "RS", "AS", "GF", "NI", "JE", "MX", "AI", "WF", "BL", "BJ", "TZ", "CL", "CD", "AX", "BQ",
        "NZ", "BE", "AU", "BW", "KH", "ZM", "TK", "PE", "JO", "ER", "NF", "GM", "CI", "MA", "NA", "CM", "NC", "RO", "KZ", "BF",
        "SR", "GW", "AW", "SK", "AG", "DK", "ET", "GQ", "MU", "NP", "UZ", "VI", "SM", "FO", "KI", "CK", "GA", "IR", "PW", "CV",
        "VE", "MN", "PY", "MV", "GI", "ZA", "MP", "TJ", "MS", "BM", "SB", "MM", "MH", "TO", "CA", "BB", "KN", "CF", "BH", "TD",
        "NG", "DJ", "EE", "HU", "TR", "KR", "DO", "TG", "MG", "RU", "MW", "JP", "HR", "DM", "GU", "EC", "TC", "TW", "BO", "TV",
        "PS", "IM", "BI", "GT", "AE", "SI", "ML", "FR", "VN", "BR", "DE", "GD", "VG", "MD", "CU", "BY", "LC", "SS", "MO", "CZ",
        "CH", "PA", "DZ", "FJ", "SD", "US", "FI", "WS", "GY", "TH", "TL", "RW", "CR", "SO", "LV", "FM", "AR", "BA", "LT", "AL",
        "CO", "GH", "IS", "BG", "LA", "TN", "IQ", "MZ", "LK", "LI", "PT", "LY", "ZW", "AD", "NR", "MT", "LB", "KE", "SG", "KM",
        "TM", "PL", "AM", "GE", "OM", "AZ", "GL", "MR", "SX", "YE", "AF", "ID", "QA", "IN", "PH", "SA", "NE", "MY", "GN", "VC",
        "NO", "NL", "GB", "FK", "LS", "MC", "SV", "SL", "KP", "NU", "SE", "CY", "MK", "AO", "PF"
    ]
    file_path = os.path.join(source_path,'json')
    dsn_path = os.path.join(source_path,'json')
    exist = set(exist)
    for file in os.listdir(file_path):
        if 'cc_rela' not in file: continue
        else:
            with open(file_path + '/' + file, 'r') as f:
                m = json.load(f)
            r = exist - set(m.keys())
            with open(file_path + '/' + file.split('.')[0] + '.nonconnect.json', 'w') as f:
                json.dump(list(r), f)


# 计算边界AS所属组织的分布，具体为各个国家管理的边界AS数量、分布的广泛情况
# 国家：{国家1：数量， 国家2：数量。。。}
def broad_as_isp_basic():
    res = {}
    with open(prefix + '/basicData/asn-iso.json', 'r') as f:
        asn_iso = json.load(f)
    path = prefix + '/ccExternal/user_influence/AUR/'
    for file in os.listdir(path):
        if file.find('AUR') == -1: continue
        _cc = file.split('_')[0]
        with open(os.path.join(path, file), 'r') as f:
            reader = json.load(f)
        for _as in reader:
            if _as in asn_iso:
                if asn_iso[_as] not in res: res[asn_iso[_as]] = {}
                if _cc not in res[asn_iso[_as]]: res[asn_iso[_as]][_cc] = 0
                res[asn_iso[_as]][_cc] += 1

    with open(prefix + '/ccExternal/analyse/asn_own_bdas_num.json', 'w') as f:
        json.dump(res, f)


prefix = '/home/peizd01/for_dragon/'
# # 跑全局抗毁性前期计算
# bar = broad_as_routingtree(os.path.join(prefix, \
#                 'ccExternal/globalCountryLabel/add_hidden_link/as_rela_code.txt'))
# broad_as_routingtree.cal_rtree_code()

# #create rtree

# # # 优化生成rtree

# broad_as_routingtree.remove_cc_internal_link(
#     '/home/peizd01/for_dragon/pzd_external_2/rtree',
#     os.path.join(prefix, 'ccExternal/globalCountryLabel/add_hidden_link/as-country-code.json'),
#     '/home/peizd01/for_dragon/pzd_external_2/json')

# # # # 跑全局抗毁性
# # # 模拟破坏
# file_path = '/home/peizd01/for_dragon/pzd_external_2/json'
# dsn_path = '/home/peizd01/for_dragon/pzd_external_2/monitor'
# os.makedirs(dsn_path,exist_ok=True)
# _monitor_remove_as(source_path)

# # 计算边界AS用户影响力
# # ur = user_radio_v2()
# # Path = [os.path.join(prefix,'beginAsFile/mergeData4_bgp/'),os.path.join(prefix+'beginAsFile/mergeData4/')]
# # ur.extract_country_broad_path(Path, os.path.join(prefix,'ccExternal/user_influence/path/'))
# # ur.calculate_ratio(os.path.join(prefix,'ccExternal/user_influence/path/'), os.path.join(prefix,'ccExternal/user_influence/info/'))
# # ur.user_radio(os.path.join(prefix,'ccExternal/user_influence/info/'), os.path.join(prefix,'ccExternal/user_influence/AUR/'), 0.1, 0.9)
