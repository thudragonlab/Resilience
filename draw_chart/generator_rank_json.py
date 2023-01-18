import os
import json
import sys

c1 = ['basic', 'user', 'domain']
c2 = ['asRank', 'problink', 'toposcope', 'toposcope_hidden']

# rank原始文件里面有，但是不显读的时候，写在skip_list里面
skip_list = []
# 用来区分优化前和优化后的结果
suffix = ''

__type_map = {
    'med': 0,
    'var': 1
}

SKIP_URB = False


def generator_type_map():
    result = {}
    for i in range(len(c1) * len(c2)):
        if c1[int(i / 4)] in skip_list or c2[int(i % 4)] in skip_list:
            continue

        result[i] = '%s-%s%s' % (c1[int(i / 4)], c2[int(i % 4)], suffix)

    return result


def generator_type_map_by_num(num):
    result = {}
    for i in range(len(c1) * len(c2)):
        if c1[int(i / 4)] in skip_list or c2[int(i % 4)] in skip_list:
            continue

        result[i] = '%s-%s.%s' % (c1[int(i / 4)], c2[int(i % 4)], num)

    return result


topo_type_map = generator_type_map()


# print(topo_type_map)


# def cal_rank_sorted(type_index, all_data):
#     sort_list = []
#     for cc in all_data:
#         if len(all_data[cc]) == 0:
#             continue
#         if len(all_data[cc]) <= type_index:
#             sort_list.append([cc, 10000])
#         else:
#             sort_list.append([cc, all_data[cc][type_index]])
#     sort_list.sort(key=lambda x: x[1])
#     return sort_list


def make_exteral_result(path, file_name, result, __type):
    real_path = os.path.join(path, file_name)

    with open(real_path) as f:
        all_data = json.load(f)
        sort_dict = {}
        print(all_data)
        for cc in all_data:
            for rank_value in all_data[cc]:
                if rank_value not in sort_dict:
                    sort_dict[rank_value] = []
                if rank_value not in result:
                    result[rank_value] = {}
                sort_dict[rank_value].append([cc, all_data[cc][rank_value]])

        for k in sort_dict:
            sort_dict[k].sort(key=lambda x: x[1])

            rank = 0
            min_v = -1

            for cl in sort_dict[k]:
                if cl[0] not in result[k].keys():
                    result[k][cl[0]] = [0, 0]
                if min_v < cl[1]:
                    rank += 1
                    min_v = cl[1]
                result[k][cl[0]][__type_map[__type]] = rank
            # print('cc = %s,rank = %s,weight = %s,type = %s' % (cl[0], rank, cl[1], __type))


def make_result(path, file_name, result, __type):
    real_path = os.path.join(path, file_name)

    with open(real_path) as f:
        all_data = json.load(f)
        sort_dict = {}
        for cc in all_data:
            for rank_value in all_data[cc]:
                if rank_value not in sort_dict:
                    sort_dict[rank_value] = []
                if rank_value not in result:
                    result[rank_value] = {}
                sort_dict[rank_value].append([cc, all_data[cc][rank_value]])

        for k in sort_dict:
            sort_dict[k].sort(key=lambda x: x[1])

            rank = 0
            min_v = -1

            for cl in sort_dict[k]:
                if cl[0] not in result[k].keys():
                    result[k][cl[0]] = [0, 0]
                if min_v < cl[1]:
                    rank += 1
                    min_v = cl[1]
                result[k][cl[0]][__type_map[__type]] = rank
            # print('cc = %s,rank = %s,weight = %s,type = %s' % (cl[0], rank, cl[1], __type))


def make_result_by_cc(path, file_name, result, __type):
    real_path = os.path.join(path, file_name)
    with open(real_path) as f:
        all_data = json.load(f)
    sort_dict = {}

    for cc in all_data:
        for rank_value in all_data[cc]:
            if rank_value not in sort_dict:
                sort_dict[rank_value] = []
            sort_dict[rank_value].append([cc, all_data[cc][rank_value]])

    for k in sort_dict:
        sort_dict[k].sort(key=lambda x: x[1])

        rank = 0
        min_v = -1

        for cl in sort_dict[k]:

            if cl[0] not in result:
                result[cl[0]] = {}
            if min_v < cl[1]:
                rank += 1
                min_v = cl[1]
            # if SKIP_URB:
            #     if cl[0] in ['US', 'RU', 'BR']:
            #         continue
            if k not in result[cl[0]]:
                result[cl[0]][k] = [0, 0]
            result[cl[0]][k][__type_map[__type]] = rank
        # print('cc = %s,rank = %s,weight = %s,type = %s' % (cl[0], rank, cl[1], __type))


def make_result_by_cc_only_cc(path, file_name, result, __type, cc3, index):
    # topo_type_map2 = generator_type_map_by_num(index)
    real_path = os.path.join(path, file_name)
    with open(real_path) as f:
        all_data = json.load(f)
        sort_dict = {}

        for cc in all_data:
            for rank_value in all_data[cc]:
                sort_dict_key = f'{rank_value}.{index}'
                if sort_dict_key not in sort_dict:
                    sort_dict[sort_dict_key] = []
                sort_dict[sort_dict_key].append([cc, all_data[cc][rank_value]])

        for k in sort_dict:
            sort_dict[k].sort(key=lambda x: x[1])
            # print(sort_dict[k],cc3,k)
            rank = 0
            min_v = -1

            for cl in sort_dict[k]:
                # if __type == 'var':
                #     print(cl,cc3)
                if cl[0] not in result and cl[0] == cc3:
                    result[cl[0]] = {}
                if min_v < cl[1]:
                    rank += 1
                    min_v = cl[1]
                # print(min_v,rank,cl)
                # if __type == 'var':
                #     print('min_v',min_v)

                if cl[0] != cc3:
                    continue
                # 'HU', 4.974242424242425
                if k not in result[cl[0]]:
                    result[cl[0]][k] = [0, 0]
                result[cl[0]][k][__type_map[__type]] = rank

            # print('cc = %s,rank = %s,weight = %s,type = %s' % (cl[0], rank, cl[1], __type))


def generator_rank_json_by_topo_type(path, file_name="rank_by_topo_type", save=True):
    result = {}
    make_result(path, 'med_rank.json', result, 'med')
    make_result(path, 'var_rank.json', result, 'var')
    dst_path = os.path.join(os.getcwd(), 'rank_files')
    os.makedirs(dst_path, exist_ok=True)
    if save:
        with open(os.path.join(dst_path, '%s%s.json' % (file_name, suffix)), 'w') as f:
            json.dump(result, f)
    return {'result': result,
            'name': '%s%s.json' % (file_name, suffix)}

def generator_external_rank_json_by_topo_type(path, file_name="rank_by_topo_type", save=True):
    result = {}
    make_exteral_result(path, 'med_rank.json', result, 'med')
    make_exteral_result(path, 'var_rank.json', result, 'var')
    dst_path = os.path.join(os.getcwd(), 'rank_files')
    os.makedirs(dst_path, exist_ok=True)
    if save:
        with open(os.path.join(dst_path, '%s%s.json' % (file_name, suffix)), 'w') as f:
            json.dump(result, f)
    return {'result': result,
            'name': '%s%s.json' % (file_name, suffix)}


def generator_rank_json_by_cc(path, file_name="rank_by_cc", save=True):
    result = {}
    make_result_by_cc(path, 'med_rank.json', result, 'med')
    make_result_by_cc(path, 'var_rank.json', result, 'var')
    dst_path = os.path.join(os.getcwd(),'rank_files')
    os.makedirs(dst_path,exist_ok=True)
    if save:
        with open(os.path.join(dst_path, '%s%s.json' % (file_name, suffix)), 'w') as f:
            json.dump(result, f)
    return {'result': result,
            'name': '%s%s.json' % (file_name, suffix)}


def generator_rank_json_by_cc_loop_cc(path, cc, file_name="rank_by_cc", save=True):
    # if cc in ['US', 'BR', 'RU']:
    #     return

    # if cc in ['IN', 'ID', 'HU', 'PT']:
    #     return

    opt_num_set = set()
    files = os.listdir(path)
    dst_path = os.path.join(os.getcwd(), 'rank_files')
    os.makedirs(dst_path, exist_ok=True)
    for i in files:
        opt_num = i.split('.')[-2]
        opt_num_set.add(opt_num)
    for i in list(opt_num_set):
        result = {}
        if not os.path.exists(os.path.join(path, 'med_rank.%s.%s.json' % (cc, i))):
            continue
        make_result_by_cc_only_cc(path, 'med_rank.%s.%s.json' % (cc, i), result, 'med', cc, i)
        make_result_by_cc_only_cc(path, 'var_rank.%s.%s.json' % (cc, i), result, 'var', cc, i)

        if save:
            with open(os.path.join(dst_path, '%s.%s.%s.json' % (file_name, cc, i)),
                      'w') as f:
                json.dump(result, f)


if __name__ == '__main__':

    '''
    运行 python3 generator_rank_json.py 运行时路径
    '''
    public_path2 = os.path.join(sys.argv[1], 'output', 'public')
    public_path = os.path.join(public_path2, 'optimize')
    generator_rank_json_by_cc(public_path2)

    cc_list = ["LV", "HU", "CL", "SK", "PT", "FI", "CO", "TR", "PH", "NG", "DK", "MX", "LU", "KH", "TW", "TH", "KR", "NO", "MY", "CN", "BG", "IR",
               "IE", "AR", "RO", "NZ", "CZ", "BD", "SE", "UA", "SG", "HK", "ES", "AT", "JP", "IN", "ID", "CH", "PL", "NL", "IT", "FR", "CA", "ZA",
               "AU", "GB", "DE", "US", "RU", "BR"]
    # cc_list = ['UA','AR','BD','RO','NL','CH','BG','CN','CZ','AT','TH','NZ','SG','CL','PH','NO','LV','MX','SK','FI','NG','CO']
    for cc2 in cc_list:
        generator_rank_json_by_cc_loop_cc(public_path, cc2)

    generator_rank_json_by_cc_loop_cc(public_path, 'LV')
    # generator_external_rank_json_by_topo_type('/Users/zdp/Desktop/resilience/test/pzd_test_external/output/public')
