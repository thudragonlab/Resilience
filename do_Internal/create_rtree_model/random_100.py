from random import randint


old_index_list = []
def get_random_index(nodeList):
    
    random_index = randint(0,len(nodeList) - 1)
    if random_index in old_index_list:
        return get_random_index(nodeList)
    old_index_list.append(random_index)
    return nodeList[random_index]


def filter_rtree(nodeList,_asn_data_global):
    nodeList2 = [get_random_index(nodeList) for i in range(100)]

    return nodeList2