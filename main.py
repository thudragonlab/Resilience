import os
from typing import Dict, List
from do_Internal.create_routingtree import create_rtree as createRoutingTree
from do_Internal.data_analysis import as_rela_txt as transformToJSON
from do_Internal.monitor_random_cut import monitor_country_internal as monitorCountryInternal

from do_Internal.anova import do_extract_connect_list, do_groud_truth_based_anova, do_country_internal_rank, do_groud_truth_based_var

from do_Internal.find_optimize_link_use_existed_cut_data import find_optimize_link
from do_Internal.train_routing_tree import train_routing_tree

from make_as_importance_weight import make_as_importance as make_as_importance
from make_cc2as import make_cc2as as make_cc2as
from util import mkdir, set_timestamp_path
from make_asn_cone_file import make_asn_cone_file as make_asn_cone_file
import json
from datetime import datetime
import sys

DEBUG = True
ONLY_PATH = True

model_map: Dict[str, str] = {
    'ALL': 'do_Internal.create_rtree_model.all_tree',
    'SORT_BY_CONE_TOP_10': 'do_Internal.create_rtree_model.sort_by_cone_top_10',
    'SORT_BY_CONE_TOP_50': 'do_Internal.create_rtree_model.sort_by_cone_top_50',
    'SORT_BY_CONE_TOP_100': 'do_Internal.create_rtree_model.sort_by_cone_top_100',
    'RANDOM_100': 'do_Internal.create_rtree_model.random_100',

    'DESTROY_All': 'do_Internal.monitor_random_cut_model.destroy_all',
    'DESTROY_TOP_10': 'do_Internal.monitor_random_cut_model.destroy_top_10',

    'CUT_FULL_TREE': 'do_Internal.monitor_random_cut_model.cut_full_tree',
    'CUT_NO_MORE_THAN_1000': 'do_Internal.monitor_random_cut_model.cut_no_more_than_1000_times',
}

if __name__ == '__main__':

    topo_list = []
    start_time = datetime.now()
    debug_path = None

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
        root_path = config['root']
        sys.path.append(root_path)
        types = config['types']
        cc_list: List[str] = config['cc_list']

        build_rtree_model: str = config['build_route_tree_model']
        destroy_rtree_model: str = config['destroy_route_tree_model']
        cut_rtree_model: str = config['cut_route_tree_model']
        cut_node_depth: int = config['cut_node_depth']
        optimize_link_list: List[int] = config['optimize_link_list']

        set_timestamp_path(root_path)
    with open(os.path.join(root_path, 'time_stamp.txt'), 'a') as time_stamp:
        time_stamp.write('\n-------------------------%s-------------------------\n' % datetime.now())
        time_stamp.write('%s\n' % cc_list)
        time_stamp.write('%s\n' % root_path)
        time_stamp.write('load config.json %s \n' % (datetime.now() - start_time))
        time_stamp.flush()
        type_map = {
            "asRank": {
                'type': 'asRank',
                'txt_path': os.path.join(root_path, 'input/asRank.txt')
            },
            "problink": {
                'type': 'problink',
                'txt_path': os.path.join(root_path, 'input/problink.txt')
            },
            "toposcope": {
                'type': 'toposcope',
                'txt_path': os.path.join(root_path, 'input/toposcope.txt')
            },
            "toposcope_hidden": {
                'type': 'toposcope_hidden',
                'txt_path': os.path.join(root_path, 'input/toposcope_hidden.txt')
            }
        }

        dst_dir_path = os.path.join(root_path, 'output')
        if DEBUG:
            debug_path = os.path.join(dst_dir_path, 'debug')
            mkdir(debug_path)
        public_path = os.path.join(dst_dir_path, 'public')
        temp_path = os.path.join(dst_dir_path, 'temp')
        mkdir(dst_dir_path)
        mkdir(public_path)
        mkdir(temp_path)

        if ONLY_PATH:
            cc2as_path = os.path.join(dst_dir_path, 'cc2as')
            cone_path = os.path.join(dst_dir_path, 'asns.json')
            weight_data_path = os.path.join(root_path, 'output/weight_data')
        else:
            cc2as_path = make_cc2as(os.path.join(root_path, 'input'), dst_dir_path)
            cone_path = make_asn_cone_file(os.path.join(root_path, 'input'), dst_dir_path)
            weight_data_path = make_as_importance(root_path, cc_list)

        start_time = datetime.now()
        with open(cone_path, 'r') as asn_f:
            asn_data:Dict[str,int] = json.load(asn_f)
        time_stamp.write('load asns.json %s  \n' % (datetime.now() - start_time))

        def make_model_path(model_path: str) -> str:
            if model_path in model_map:
                return model_map[model_path]
            else:
                return '.'.join(['input', model_path])

        build_rtree_model_path = make_model_path(build_rtree_model)
        destroy_rtree_model_path = make_model_path(destroy_rtree_model)
        cut_rtree_model_path = make_model_path(cut_rtree_model)

        for _type in types:
            if not types[_type]:
                continue
            topo_list.append(_type)
            # time_stamp.write('------------------- %s start -------------------\n' % _type)
            # time_stamp.flush()
            # as_rela_file = transformToJSON(dst_dir_path, type_map[_type]['txt_path'])
            # if as_rela_file == '':
            #     raise Exception('PATH ERROR')
            # createRoutingTree(as_rela_file, dst_dir_path, _type, cc_list, asn_data, cc2as_path, build_rtree_model_path)
            # monitorCountryInternal(dst_dir_path, _type, asn_data, destroy_rtree_model_path, cut_rtree_model_path,
            #                        cut_node_depth)
            # #     # # # anova.py
            # do_extract_connect_list(dst_dir_path, _type, weight_data_path)
            # do_groud_truth_based_anova(dst_dir_path, _type, debug_path)
            # do_groud_truth_based_var(dst_dir_path, _type, debug_path)
            # find_optimize_link(type_map[_type]['txt_path'], os.path.join(dst_dir_path, _type), cone_path, cc_list,
            #                    weight_data_path)
            # time_stamp.write('------------------- %s end ------------------- \n\n' % _type)
            # time_stamp.flush()
        do_country_internal_rank(dst_dir_path, cc_list, topo_list, debug_path)
        # train_routing_tree(topo_list, cc_list, dst_dir_path, weight_data_path, optimize_link_list)
