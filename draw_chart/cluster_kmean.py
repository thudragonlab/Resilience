import sys

import matplotlib.pyplot as plt
from draw_chart.generator_rank_json import generator_rank_json_by_topo_type, generator_external_rank_json_by_topo_type
import os
import json
from pylab import mpl
from sklearn.cluster import KMeans
import numpy as np

c2 = ['asRank']

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
mpl.rcParams['font.family'] = 'times'
# mpl.rcParams['figure.constrained_layout.use'] = True


# cm = {}
#
# with open('country_map.json', 'r') as cmf:
#     cm = json.load(cmf)

mark_list = ['+', 's', 'o', 'P', 'x', '.', 'D', 'D', ]
facecolors_list = ['black', 'none', 'none', 'none', 'black', 'black', 'black', 'black', ]
color_list = ['red', 'green', 'black', 'blue', 'orange', 'pink', 'purple']

title_font_size = 25
label_font_size = 22
table_font_size = 20
row0_font_size = 20
n_clusters = 6


def make_result(path, file_name, _type, result):
    real_path = os.path.join(path, file_name)

    with open(real_path) as f:
        all_data = json.load(f)
        for cc in all_data:
            start_index = 0
            lens = len(all_data[cc])
            end_index = lens
            if _type == 'basic':
                end_index = int((lens / 3) * 1)
            elif _type == 'user':
                start_index = int((lens / 3) * 1)
                end_index = int((lens / 3) * 2)
            elif _type == 'domain':
                start_index = int((lens / 3) * 2)
            if cc not in result.keys():
                result[cc] = [0, 0]
            if file_name == 'med_rank.json':
                result[cc][0] = sum(all_data[cc][start_index:end_index]) * 4
            if file_name == 'var_rank.json':
                result[cc][1] = sum(all_data[cc][start_index:end_index])


def get_all_num(cc_as):
    cc = cc_as.split('-')[0]
    with open('ccInternal/cc2as/%s.json' % cc, 'r') as f:
        as_list = json.load(f)
        return len(as_list)


def get_as_num(path, cc_as, _dir):
    cc_and_as = cc_as.split('-')
    if _dir != 'var':
        cc_and_as[1] = 'dcomplete%s' % cc_and_as[1]
    if os.path.exists('%s/count_num/%s/%s.json' % (path, cc_and_as[0], cc_and_as[1])):
        with open('%s/count_num/%s/%s.json' % (path, cc_and_as[0], cc_and_as[1]), 'r') as f:
            data = json.load(f)
            return int(data['%s' % cc_and_as[1]]['asNum'])
    return 0


def make_result_by_rank(path, _type):
    real_result = {}
    topo_list = c2
    result = generator_rank_json_by_topo_type(path, save=False)['result']
    for topo in topo_list:
        t = result['%s-%s' % (_type, topo)]
        for k in t:
            if k not in real_result:
                real_result[k] = [0, 0]
            real_result[k][0] += t[k][0]
            real_result[k][1] += t[k][1]

    return real_result


def make_result_by_rank_all(path):
    real_result = {}
    topo_list = c2

    result = generator_rank_json_by_topo_type(path, save=False)['result']
    for topo in topo_list:
        for _type in ['basic', 'user', 'domain']:
            t = result['%s-%s' % (_type, topo)]
            for k in t:
                if k not in real_result:
                    real_result[k] = [0, 0]
                real_result[k][0] += t[k][0]
                real_result[k][1] += t[k][1]

    return real_result


def make_external_result_by_rank_all(path):
    real_result = {}
    result = generator_external_rank_json_by_topo_type(path, save=False)['result']
    t = result['rank']
    for k in t:
        if k not in real_result:
            real_result[k] = [0, 0]
        real_result[k][0] += t[k][0]
        real_result[k][1] += t[k][1]

    return real_result


def draw_plot(_type, path, position, title):
    estimator = KMeans(n_clusters=n_clusters)  # 构造聚类器

    result = {}
    dir_path = path
    # USE RANK
    result = make_result_by_rank(dir_path, _type)

    # USE WEIGHT
    # make_result(dir_path, 'med_rank.json', _type, result)
    # make_result(dir_path, 'var_rank.json', _type, result)

    ax = fig.add_subplot(position)
    ax.set_title(title, fontsize=title_font_size)
    # fig.subplots_adjust(right=0.8)
    X = list(map(lambda x: [result[x][0], result[x][1], x], result))
    X.sort(key=lambda x: float(x[0]) + float(x[1]))
    # X1 = list(map(lambda x: [x[0], x[1]], result.values()))
    X = np.array(X)
    estimator.fit(X[:, :2])
    label_pred = estimator.labels_
    table_data = []
    table_index = -1
    old_class_index = {}
    a_map = {}
    print(label_pred)
    for index, data in enumerate(X):
        # print(label_pred[index], data[2])
        if label_pred[index] not in old_class_index:
            table_index += 1
            old_class_index[label_pred[index]] = table_index
        using_table_index = old_class_index[label_pred[index]]

        if len(table_data) == using_table_index:
            table_data.append(['Class%s' % (using_table_index + 1)])
        a = ax.scatter(float(data[0]), float(data[1]), marker=mark_list[using_table_index], facecolors=facecolors_list[using_table_index],
                       s=100, color="black")
        if mark_list[using_table_index] not in a_map:
            a_map[mark_list[using_table_index]] = a

        table_data[using_table_index].append(data[2])
    ax.legend(handles=list(a_map.values()), labels=list(map(lambda x: x[0], table_data)), bbox_to_anchor=(0.99, 0.02), loc='lower right',
              labelspacing=1.5,
              borderaxespad=0., prop={'size': 15, 'family': 'times'})
    max_len = max(map(lambda x: len(x), table_data))

    for index, i in enumerate(table_data):
        if len(i) < max_len:
            i += [''] * (max_len - len(i))

    table = ax.table(cellText=table_data, bbox=[0.0, -0.45, 1, 0.35], cellLoc='center')
    cells = table.properties()["celld"]
    table.auto_set_font_size(False)
    table.set_fontsize(table_font_size)
    # table.scale(1, 4)
    for i in range(0, n_clusters):
        cells[i, 0].set_fontsize(row0_font_size)
        # for j in range(0,max_len):
        #     cells[i,j].set_height(0.1)

    ax.add_table(table)

    ax.tick_params(axis='both', which='major', labelsize=label_font_size)


def draw_plot_all(_type, path, position, title):
    estimator = KMeans(n_clusters=n_clusters)  # 构造聚类器

    result = {}
    dir_path = path
    # USE RANK
    if sys.argv[2] == 'e':
        result = make_external_result_by_rank_all(dir_path)
    else:
        result = make_result_by_rank_all(dir_path)


    # USE WEIGHT
    # make_result(dir_path, 'med_rank.json', _type, result)
    # make_result(dir_path, 'var_rank.json', _type, result)

    ax = fig.add_subplot(position)
    ax.set_title(title, fontsize=title_font_size * 1.5)
    # fig.subplots_adjust(right=0.8)
    X = list(map(lambda x: [result[x][0], result[x][1], x], result))
    X.sort(key=lambda x: float(x[0]) + float(x[1]))
    # X1 = list(map(lambda x: [x[0], x[1]], result.values()))
    X = np.array(X)
    estimator.fit(X[:, :2])
    label_pred = estimator.labels_
    table_data = []
    table_index = -1
    old_class_index = {}
    a_map = {}
    # print(label_pred)
    for index, data in enumerate(X):
        # print(label_pred[index], data[2])
        if label_pred[index] not in old_class_index:
            table_index += 1
            old_class_index[label_pred[index]] = table_index
        using_table_index = old_class_index[label_pred[index]]

        if len(table_data) == using_table_index:
            table_data.append(['Class%s' % (using_table_index + 1)])
        a = ax.scatter(float(data[0]), float(data[1]), marker=mark_list[using_table_index], facecolors=facecolors_list[using_table_index],
                       s=100, color="black")
        if mark_list[using_table_index] not in a_map:
            a_map[mark_list[using_table_index]] = a

        table_data[using_table_index].append(data[2])
    ax.legend(handles=list(a_map.values()), labels=list(map(lambda x: x[0], table_data)), bbox_to_anchor=(0.99, 0.02), loc='lower right',
              labelspacing=1.5,
              borderaxespad=0., prop={'size': 15, 'family': 'times'})
    max_len = max(map(lambda x: len(x), table_data))

    for index, i in enumerate(table_data):
        if len(i) < max_len:
            i += [''] * (max_len - len(i))

    table = ax.table(cellText=table_data, bbox=[0.0, -0.45, 1, 0.35], cellLoc='center')
    cells = table.properties()["celld"]
    table.auto_set_font_size(False)
    table.set_fontsize(table_font_size)
    # table.scale(1, 4)
    for i in range(0, n_clusters):
        cells[i, 0].set_fontsize(row0_font_size)
        # for j in range(0,max_len):
        #     cells[i,j].set_height(0.1)

    ax.add_table(table)

    ax.tick_params(axis='both', which='major', labelsize=label_font_size)


if __name__ == '__main__':
    fig = plt.figure(figsize=(15, 10))  # 创建画布
    # fig = plt.figure(figsize=(30, 10))  # 创建画布
    # draw_plot('basic', os.path.join(sys.argv[1], 'output', 'public'), 131, 'clustering results(connectivity)')
    # print('------------')
    # draw_plot('user', os.path.join(sys.argv[1], 'output', 'public'), 132, 'clustering results(user)')
    # print('------------')
    # draw_plot('domain', os.path.join(sys.argv[1], 'output', 'public'), 133, 'clustering results(domain)')
    # print('------------')
    draw_plot_all('', os.path.join(sys.argv[1], 'output', 'public'), 111, 'clustering results of Inter-region survivability')
    plt.show()
