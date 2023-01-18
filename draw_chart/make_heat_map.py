import json
import os
import sys


def cal(input_paths, value_index, output_path):
    result = {}
    for input_path in input_paths:
        with open(input_path, 'r') as f:
            all_data = json.load(f)
            for cc in all_data:
                # if cc in ['IN', 'ID', 'HU', 'PT']:
                #     continue
                if cc not in result:
                    result[cc] = {}
                for k in all_data[cc]:
                    result[cc][k] = all_data[cc][k][value_index]

    l = []
    for i in result:
        if i not in cc_list:
            l.append(i)

    for i in l:
        del result[i]
    with open(output_path, 'w') as ffff:
        json.dump(result, ffff)
    # for i in result:
    #     print(i, result[i])


if __name__ == '__main__':
    dst_path = os.path.join(os.getcwd(), 'rank_files')
    # cc_list  = ['UA','AR','BD','RO','NL','CH','BG','CN','CZ','AT','TH','NZ','SG','CL','PH','NO','LV','MX','SK','FI','NG','CO']
    cc_list = ["LV", "HU", "CL", "SK", "PT", "FI", "CO", "TR", "PH", "NG", "DK", "MX", "LU", "KH", "TW", "TH", "KR", "NO", "MY", "CN", "BG", "IR",
               "IE", "AR",
               "RO", "NZ", "CZ", "BD", "SE", "UA", "SG", "HK", "ES", "AT", "JP", "IN", "ID", "CH", "PL", "NL", "IT", "FR", "CA", "ZA", "AU", "GB",
               "DE", "US", "RU", "BR"]
    file_list = [os.path.join(dst_path, 'rank_by_cc.json')]
    opt_num_set = set()
    files = os.listdir(os.path.join(sys.argv[1], 'output', 'public', 'optimize'))
    for i in files:
        opt_num = i.split('.')[-2]
        opt_num_set.add(opt_num)
    for cc in cc_list:
        for i in list(opt_num_set):
            if not os.path.exists(os.path.join(dst_path, 'rank_by_cc.%s.%s.json' % (cc, i))):
                continue
            file_list.append(os.path.join(dst_path, 'rank_by_cc.%s.%s.json' % (cc, i)))

    cal(file_list, 1, 'var.json')
    cal(file_list, 0, 'med.json')
