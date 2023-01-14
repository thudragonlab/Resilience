from random import randint
from copy import copy

old_index_list = []
def get_random_index(node2):
    if len(node2) == 0:
        return '-1'
    random_index = randint(0,len(node2) - 1)
    result = node2[random_index]
    node2.remove(result)
    # if random_index in old_index_list:
    #     return get_random_index(node2)
    # old_index_list.append(random_index)
    return result


def filter_rtree(nodeList,_asn_data_global):
    node2 = copy(nodeList)
    nodeList2 = [get_random_index(node2) for i in range(100)]

    return list(filter(lambda x:x != '-1',nodeList2))