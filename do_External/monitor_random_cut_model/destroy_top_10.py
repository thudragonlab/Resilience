
def get_destroy_trees(tree_list):
    sorted_trees = sorted(tree_list, key=lambda x: x[1], reverse=True)
    return sorted_trees[:10]