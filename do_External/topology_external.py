#!usr/bin/env python
# _*_ coding:utf8 _*_

import os
import json
from other_script.my_types import *
import numpy as np
from importlib import import_module
from multiprocessing.pool import ThreadPool

import random

'''
传输AS影响+AS霸权思想
上面都是想法一的处理过程，想法二由增加了计算国内所有AS的数据，所以得重新写所有相关代码
'''

N = 10
gl_get_cut_num: Callable[[List[AS_CODE]], List[AS_CODE]] = None

'''
分析国家边界AS的安全性
STEP1 计算国家边界AS的routingTree,加入各个国家内部的拓扑,以及各个国家边界之间的拓扑
STEP2 去除国家内部的边，只保留国家边界之间的连接，增加国家内部国家边界之间连接的虚拟边
STEP3 模拟断开
'''


class broad_as_routingtree():

    def __init__(self, source_path, cc_list, as_rela_file):
        '''
        as_rela_code.txt 新生成的路由数据
        as-country-code.json 新生成的as号
        cal_rtree_code_v2.json 边界as
        '''
        self.file_name = os.path.join(source_path, 'as_rela_code.txt')
        self.as_country_code_path = os.path.join(source_path, 'as-country-code.json')
        self.path = os.path.join(source_path, 'cc2as')
        self.only_cc_list = cc_list
        self.code = 1
        self.cal_rtree_code_v2_path = os.path.join(source_path, 'cal_rtree_code_v2.json')
        self.as_rela_file = as_rela_file
        with open(self.file_name, 'w') as f:
            f.write('')
        self.country_code = {}
        self.make_country_code()
        self.add_internal_link()
        self.add_external_link()
        with open(self.as_country_code_path, 'w') as f:
            json.dump(self.country_code, f)

    def make_country_code(self):
        '''
        重新生成as号,对应as和国家
        '''
        cc2as = os.listdir(self.path)
        for cc_file in cc2as:
            with open(os.path.join(self.path, cc_file), 'r') as f:
                ases = json.load(f)
            cc = cc_file[:-5]
            for a in ases:
                self.country_code[a + '-' + cc] = str(self.code)
                self.code += 1

    def as2code(self, a):
        '''
        a : as-country
        去新生成的as号字典中找对应的新的as号
        -如果没有找到,就在后面追加一个新的as
        '''
        if a not in self.country_code:
            a, b = a.split('-')
            with open(os.path.join(self.path, b + '.json'), 'r') as f:
                r = json.load(f)
            r.append(a)
            with open(os.path.join(self.path, b + '.json'), 'w') as f:
                json.dump(r, f)
            self.country_code[a + '-' + b] = str(self.code)
            self.code += 1
            return self.country_code[a + '-' + b]
        return self.country_code[a]

    def add_internal_link(self):
        '''
        读取国家对应as文件,重新生成路由txt文件
        txt中把as号替换成新生成的as号
        '''
        with open(os.path.join(self.as_rela_file), 'r') as f:
            as_rela = json.load(f)

        cc2as = os.listdir(self.path)
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
                                if c == 'None' or b == 'None': continue
                                f_dsn.write(self.as2code(str(c) + '-' + cc) + '|' + self.as2code(str(b) + '-' + cc) + '|-1\n')
            f_dsn.close()

    def add_external_link(self):
        '''
        从broadPath_v2读取外部as关系,补充到路由txt中
        txt中把as号替换成新生成的as号
        '''
        with open(self.as_rela_file, 'r') as f:
            as_rela = json.load(f)

        file_name = os.listdir('static/broadPath_v2/')
        temp = []
        for file in file_name:
            with open('static/broadPath_v2/' + file, 'r') as ff:
                reader = json.load(ff)
            for k in reader:
                for i in reader[k].keys():
                    a_c, b_c = i.split('|')[1].split(' ')
                    if a_c not in self.only_cc_list or b_c not in self.only_cc_list:
                        continue
                    temp.append(i)

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

    def cal_rtree_code(self):
        '''
        生成边界as
        从broadPath_v2中读取所有国家边界as的数据
        存到v2.json
        '''
        with open(self.as_country_code_path, 'r') as f:
            code = json.load(f)
        res = []
        Res = []

        file_name = os.listdir('static/broadPath_v2/')
        for file in file_name:
            with open('static/broadPath_v2/' + file, 'r') as f:
                r = json.load(f)
            for p in r:
                for line in r[p]:
                    a1, a2 = line.split('|')[0].split(' ')
                    c1, c2 = line.split('|')[1].split(' ')
                    if c1 == '' or c2 == '' or f'{a1}-{c1}' not in code or c1 not in self.only_cc_list: continue
                    res.append(a1 + '-' + c1)

        res = list(set(res))
        for i in res:
            Res.append(code[i])

        with open(self.cal_rtree_code_v2_path, 'w') as f:
            json.dump(Res, f)

    def remove_cc_internal_link(self, source_path):

        encoder_path = os.path.join(source_path, 'as-country-code.json')
        npz_path = os.path.join(source_path, 'rtree')

        with open(encoder_path, 'r') as f:
            encoder = json.load(f)
        incoder = {encoder[i]: i for i in encoder}
        file_name = os.listdir(npz_path)
        for file in file_name:
            # 过滤条件，只破坏我想破坏的
            remove_internal_link(source_path, incoder, file)


def remove_internal_link(source_path, incoder, file):
    npz_path = os.path.join(source_path, 'rtree')
    json_path = os.path.join(source_path, 'json')
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


class MonitorRemoveAs():

    def __init__(self, file_path, dsn_path, specific_country=None):
        self.file_path = file_path
        self.n_as = N
        self.dsn_path = dsn_path
        self.specific_country = specific_country

    def remove_country_link(self, queue: List[COUNTRY_CODE]) -> List[COUNTRY_CODE]:
        '''
        queue 破坏的节点列表
        删除国家内部链接
        '''
        res: List[COUNTRY_CODE] = []
        # 删除边
        tempQ: List[COUNTRY_CODE] = []
        # 找后向点只包含queue中的国家的国家，存入tempQ
        for link in queue:
            # 左节点 右节点
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
            # 删除node的后向点
            for i in self.cc_rela[node][1]:
                self.cc_rela[i][0].remove(node)
            self.cc_rela[node][1] = []
        while tempQ:
            n = tempQ.pop(0)
            res.append(n)
            if n not in self.cc_rela: continue
            # 删除node的前向点
            for i in self.cc_rela[n][0]:
                self.cc_rela[i][1].remove(n)
                # 如果删除完node的前向点的后向点列表为空，那把这个点也删除
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
        with open(os.path.join(self.file_path, file), 'r') as f:
            # 读取国家的前后向国家
            self.cc_rela = json.load(f)
        # 读取国家的具体连接节点
        with open(os.path.join(self.file_path, file.split('.')[0] + '.cc_pair_link.json'), 'r') as f:
            self.cc_pair_link = json.load(f)
        nodelist = set()
        for key in self.cc_pair_link:
            for line in self.cc_pair_link[key]:
                nodelist.add(line.split(' ')[0])
                nodelist.add(line.split(' ')[1])
        # 把所有节点存成list
        nodelist = list(nodelist)

        # 挑选出只有某个国家的nodelist
        with open('static/asn-iso.json', 'r') as f:
            asn_iso = json.load(f)

        if self.specific_country:
            for i in range(len(nodelist) - 1, -1, -1):
                if nodelist[i] not in asn_iso or asn_iso[nodelist[i]] != self.specific_country:
                    nodelist.pop(i)

        # 就最多破坏500次

        node2link: Dict[str, List[COUNTRY_CODE]] = {}
        # 找所有cc_pair_link中和node有关的保留
        for node in nodelist:
            remove_cc_link: List[COUNTRY_CODE] = []
            for key in self.cc_pair_link:
                flag = 1
                for line in self.cc_pair_link[key]:
                    a, b = line.split(' ')
                    if a not in node and b not in node:
                        flag = 0
                        break
                if flag:
                    # 不包含node的国家加入到remove_cc_link
                    remove_cc_link.append(key.split(' '))
            if len(remove_cc_link):
                node2link[node] = remove_cc_link
        # if len(nodelist) <= self.n_as:
        if len(nodelist) <= 0:
            return

        with open(os.path.join(self.dsn_path, file.split('.')[0] + '.del.txt'), 'w') as dsn_f:

            # 破坏N次，每次破坏1-N个节点
            for num in N:
                flag = 0
                cut_times: int = gl_get_cut_num(nodelist)
                # print(cut_times, flag)
                while flag < cut_times:
                    # 每次破坏时最多破坏epoch次
                    flag += 1
                    # 如果nodeList比破坏的节点数要少，就破坏nodeLis全部节点
                    if num > len(nodelist):
                        num = len(nodelist)
                    # 随机破坏的节点
                    node = random.sample(nodelist, num)
                    node = list(set(list(node)))
                    node.sort()
                    # 读取cc_rela.json文件,获取国家的前向点和后向点
                    with open(os.path.join(self.file_path, file), 'r') as f:
                        self.cc_rela = json.load(f)
                    res = []
                    linklist: List[COUNTRY_CODE] = []
                    for _node in node:
                        # 如果node在cc_pair_link中，表示node时连接两个国家的节点的其中一个
                        if _node in node2link:
                            # 把和node无关的节点加入到linklist中
                            for __node in node2link[_node]:
                                if __node not in linklist:
                                    linklist.append(__node)
                    if linklist:
                        # 受影响的节点
                        res: List[COUNTRY_CODE] = self.remove_country_link(linklist)
                    dsn_f.write(' '.join(node) + '|' + ' '.join(res) + '\n')

        print(file + ' success')


def monitor_remove_as(source_path, cut_node_depth, cut_rtree_model_path):
    '''
    模拟破坏,破坏cut_node_depth个节点
    '''
    global N
    global gl_get_cut_num
    N = cut_node_depth
    file_path = os.path.join(source_path, 'json')
    dsn_path = os.path.join(source_path, 'monitor')
    dynamic_module_2 = import_module(cut_rtree_model_path)
    gl_get_cut_num = dynamic_module_2.get_cut_num
    os.makedirs(file_path, exist_ok=True)
    os.makedirs(dsn_path, exist_ok=True)

    def pool_monitor_as(file):
        if file.split('.')[0] + '.del.txt' in os.listdir(dsn_path):
            print(file + ' exist')
            return
        mra = MonitorRemoveAs(file_path, dsn_path, None)
        mra.remove_as(file)

    file_name = os.listdir(file_path)
    file_run = [[]]
    for file in file_name:
        if file.find('cc_rela') == -1: continue
        if file.split('.')[0] + '.cc_pair_link.json' not in file_name: continue

        # file是.cc_rela.json文件
        if len(file_run[-1]) < 6:
            file_run[-1].append(file)
        else:
            file_run.append([file])
    for _file_run in file_run:
        pool = ThreadPool()
        pool.map(pool_monitor_as, _file_run)
    pool.close()
    pool.join()


# 计算出边界AS routingTree中本身没有通信的国家
def create_nonconnect(source_path):
    # 已存在的所有国家
    exist = list(map(lambda x: x[:2], os.listdir(os.path.join(source_path, 'cc2as'))))
    file_path = os.path.join(source_path, 'json')
    dsn_path = os.path.join(source_path, 'json')
    exist = set(exist)
    for file in os.listdir(file_path):
        if 'cc_rela' not in file:
            continue
        else:
            # 读取cc_rela.json文件
            with open(file_path + '/' + file, 'r') as f:
                m = json.load(f)
            # 和当前国家没有as连接的国家列表
            r: Set[COUNTRY_CODE] = exist - set(m.keys())
            with open(file_path + '/' + file.split('.')[0] + '.nonconnect.json', 'w') as f:
                json.dump(list(r), f)
