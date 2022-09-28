asn_data_global = {}

def sort_by_cone(k):
        sk = str(k)
        if sk in asn_data_global:
            return asn_data_global[sk]
        else:
            return 0


def filter_rtree(nodeList,_asn_data_global):
    global asn_data_global
    asn_data_global = _asn_data_global
    nodeList.sort(key=sort_by_cone,reverse=True)
    return nodeList[0:50]