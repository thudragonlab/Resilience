from do_External.create_routingtree_external import main as create_routingTree
from do_External.external_security import anova_sort, country_broadas_rank, extract_connect_list, groud_truth_based_anova, groud_truth_based_var, second_order
from do_External.topology_external import broad_as_routingtree, _monitor_remove_as, create_nonconnect
from do_Internal.data_analysis import as_rela_txt as transformToJSON
import os
import sys
import json
from other_script.make_asn_cone_file import make_asn_cone_file
from other_script.make_cc2as import make_cc2as

from other_script.my_types import *
model_map: Dict[str, str] = {
    'ALL': 'do_Internal.create_rtree_model.all_tree',
    'SORT_BY_CONE_TOP_10': 'do_Internal.create_rtree_model.sort_by_cone_top_10',
    'SORT_BY_CONE_TOP_50': 'do_Internal.create_rtree_model.sort_by_cone_top_50',
    'SORT_BY_CONE_TOP_100': 'do_Internal.create_rtree_model.sort_by_cone_top_100',
    'RANDOM_100': 'do_Internal.create_rtree_model.random_100',

    'DESTROY_All': 'do_External.monitor_random_cut_model.destroy_all',
    'DESTROY_TOP_10': 'do_External.monitor_random_cut_model.destroy_top_10',
    'DESTROY_TOP_6_BY_TREE_NODE': 'do_External.monitor_random_cut_model.destroy_top_6_by_tree_node',
    'DESTROY_TOP_10_BY_TREE_NODE': 'do_External.monitor_random_cut_model.destroy_top_10_by_tree_node',

    'CUT_FULL_TREE': 'do_External.monitor_random_cut_model.cut_full_tree',
    'CUT_NO_MORE_THAN_1000': 'do_External.monitor_random_cut_model.cut_no_more_than_1000_times',

    'CAL_RANK_SUM_RANK': 'do_Internal.cal_rank.cal_method_1',
    'CAL_RANK_SUM_RANK_AND_LINK_COUNT': 'do_Internal.cal_rank.cal_method_2',

    'RECORD_TREE_AS': 'do_Internal.monitor_random_cut_origin',
    'RECORD_COUNTRY_AS': 'do_Internal.monitor_random_cut',
}


def make_model_path(model_path: str) -> str:
    if model_path in model_map:
        return model_map[model_path]
    else:
        return '.'.join(['input', model_path])
    
if __name__ == '__main__':
    source_path = sys.argv[1]
    # print(len(os.listdir(os.path.join(source_path,'output/rtree'))))
    # print(len(os.listdir(os.path.join(source_path,'output/monitor'))))
    
    type_map = {
            "asRank": {
                'type': 'asRank',
                'txt_path': os.path.join(source_path, 'input/asRank.txt')
            },
            # "problink": {
            #     'type': 'problink',
            #     'txt_path': os.path.join(source_path, 'input/problink.txt')
            # },
            # "toposcope": {
            #     'type': 'toposcope',
            #     'txt_path': os.path.join(source_path, 'input/toposcope.txt')
            # },
            # "toposcope_hidden": {
            #     'type': 'toposcope_hidden',
            #     'txt_path': os.path.join(source_path, 'input/toposcope_hidden.txt')
            # }
        }
    dst_dir_path = os.path.join(source_path,'output')
    os.makedirs(dst_dir_path,exist_ok=True)
    with open(os.path.join(source_path,'input','config.json'),'r') as cf:
        config= json.load(cf)
        rtree_node_min_cone: int = config['min_cone']
        cut_route_tree_model: str = config['cut_route_tree_model']
        cut_route_tree_model_path = make_model_path(cut_route_tree_model)
        cc_list: List[COUNTRY_CODE] = config['cc_list']
        cut_node_depth: int = config['cut_node_depth']

    cc2as_path: CC2AS_PATH = make_cc2as(os.path.join(source_path, 'input'), dst_dir_path)
    cone_path = make_asn_cone_file(os.path.join(source_path, 'input'), dst_dir_path)
    with open(cone_path, 'r') as asn_f:
        asn_data: Dict[AS_CODE, int] = json.load(asn_f)
    as_rela_file = transformToJSON(dst_dir_path, type_map['asRank']['txt_path'], asn_data, rtree_node_min_cone)
    as_rela_file_txt = os.path.join(dst_dir_path,'as_rela_code.txt')
    v2_path = os.path.join(dst_dir_path,'cal_rtree_code_v2.json')
    
    
    bar = broad_as_routingtree(dst_dir_path,cc_list,as_rela_file)
    bar.cal_rtree_code()

    create_routingTree(dst_dir_path,v2_path,as_rela_file_txt)
    # # # # bar.remove_cc_internal_link(source_path)
    _monitor_remove_as(dst_dir_path,cut_node_depth,cut_route_tree_model_path)
    create_nonconnect(dst_dir_path)
    extract_connect_list(dst_dir_path,cut_node_depth)
    groud_truth_based_anova(dst_dir_path)
    groud_truth_based_var(dst_dir_path)
    
    # # # # groud_truth_based_anova 内部调用 anova_sort
    # # # ## anova_sort(os.path.join(source_path,'result','anova'), 'democracy')


    country_broadas_rank(os.path.join(dst_dir_path,'result/anova/sorted_country_democracy.json'),\
        os.path.join(dst_dir_path,'public'), \
            'static/AUR/', \
        os.path.join(dst_dir_path,'as-country-code.json'),'med_rank')
    
    country_broadas_rank(os.path.join(dst_dir_path,'result/var/sorted_country_democracy.json'),\
        os.path.join(dst_dir_path,'public'), \
            'static/AUR/', \
        os.path.join(dst_dir_path,'as-country-code.json'),'var_rank')

    # # # # # # TODO topology_external
    # # # # # # TODO create rtree