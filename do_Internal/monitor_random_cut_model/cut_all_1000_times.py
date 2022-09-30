def get_list_len(nodeList):
    if len(nodeList)<1000:
        return len(nodeList)
    else:
        return 1000

def get_top_rtree(file):
    return file