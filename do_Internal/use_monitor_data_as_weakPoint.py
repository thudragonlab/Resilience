import os
import json


def make_weak_point(rtree_path, cc, dst_path):
    files_list = os.listdir(os.path.join(rtree_path, cc))
    sample_result = {}
    result = {}
    for file_name in files_list:
        # print(file_name)
        if 'addDel' in file_name:
            asn = file_name[9:-11]
            with open(os.path.join(rtree_path, cc, file_name)) as f:
                for line in f.readlines():
                    line = line.strip()
                    if '#' in line:
                        continue
                    if line == '|':
                        break
                    break_node, affected_node = line.split('|')
                    sample_result['%s|%s' % (break_node, asn)] = affected_node

    sorted_list = sorted(sample_result.items(), key=lambda x: len(x[1].split(' ')), reverse=True)
    for i in sorted_list:
        break_node, asn = i[0].split('|')
        if break_node not in result:
            result[break_node] = []
        for an in i[1].split(' '):
            result[break_node].append([asn, an])


    with open(os.path.join(dst_path, '%s.break_link.json' % cc), 'w') as f:
        json.dump(result, f)
    return result
