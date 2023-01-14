import os
from do_Internal.sort_rank import set_gl_cal_rank_model
from other_script.my_types import *
from do_Internal.create_routingtree import create_rtree as createRoutingTree
from do_Internal.data_analysis import as_rela_txt as transformToJSON
# from do_Internal.monitor_random_cut import monitor_country_internal as monitorCountryInternal
from do_Internal.monitor_random_cut_origin import monitor_country_internal as monitorCountryInternal_local

from do_Internal.anova import do_extract_connect_list, do_groud_truth_based_anova, do_country_internal_rank, do_groud_truth_based_var

from do_Internal.find_optimize_link_origin import find_optimize_link
# from do_Internal.find_optimize_link_use_existed_cut_data import find_optimize_link
# from do_Internal.find_optimize_link import find_optimize_link
from do_Internal.train_routing_tree import train_routing_tree

from other_script.make_as_importance_weight import make_as_importance as make_as_importance
from other_script.make_cc2as import make_cc2as as make_cc2as
from other_script.util import mkdir, set_timestamp_path
from other_script.make_asn_cone_file import make_asn_cone_file as make_asn_cone_file
from importlib import import_module
import json
from datetime import datetime
import sys

ONLY_PATH = False

model_map: Dict[str, str] = {
    'ALL': 'do_Internal.create_rtree_model.all_tree',
    'SORT_BY_CONE_TOP_10': 'do_Internal.create_rtree_model.sort_by_cone_top_10',
    'SORT_BY_CONE_TOP_50': 'do_Internal.create_rtree_model.sort_by_cone_top_50',
    'SORT_BY_CONE_TOP_100': 'do_Internal.create_rtree_model.sort_by_cone_top_100',
    'RANDOM_100': 'do_Internal.create_rtree_model.random_100',

    'DESTROY_All': 'do_Internal.monitor_random_cut_model.destroy_all',
    'DESTROY_TOP_10': 'do_Internal.monitor_random_cut_model.destroy_top_10',
    'DESTROY_TOP_6_BY_TREE_NODE': 'do_Internal.monitor_random_cut_model.destroy_top_6_by_tree_node',
    'DESTROY_TOP_10_BY_TREE_NODE': 'do_Internal.monitor_random_cut_model.destroy_top_10_by_tree_node',

    'CUT_FULL_TREE': 'do_Internal.monitor_random_cut_model.cut_full_tree',
    'CUT_NO_MORE_THAN_1000': 'do_Internal.monitor_random_cut_model.cut_no_more_than_1000_times',

    'CAL_RANK_SUM_RANK': 'do_Internal.cal_rank.cal_method_1',
    'CAL_RANK_SUM_RANK_AND_LINK_COUNT': 'do_Internal.cal_rank.cal_method_2',

    'RECORD_TREE_AS': 'do_Internal.monitor_random_cut_origin',
    'RECORD_COUNTRY_AS': 'do_Internal.monitor_random_cut',
}

if __name__ == '__main__':

    topo_list = []
    start_time = datetime.now()
    # debug_path = None

    with open(os.path.join(sys.argv[1], 'input', 'config.json'), 'r') as config_file:
        config = json.load(config_file)
        root_path: ROOT_PATH = sys.argv[1]
        sys.path.append(root_path)
        types: Dict[TOPO_TPYE, bool] = config['types']
        cc_list: List[COUNTRY_CODE] = config['cc_list']
        rtree_node_min_cone: int = config['min_cone']
        data_aspect: List[ASPECT_TPYE] = config['data_dim']
        build_rtree_model: str = config['build_route_tree_model']
        destroy_rtree_model: str = config['destroy_route_tree_model']
        cut_rtree_model: str = config['cut_route_tree_model']
        cal_rank_model: str = config['cal_rank_model']
        monitor_model: str = config['monitor_model']
        cut_node_depth: int = config['cut_node_depth']
        opt_cc_list: List[COUNTRY_CODE] = config['opt_cc_list']
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

        dst_dir_path: OUTPUT_PATH = os.path.join(root_path, 'output')
        # if DEBUG:
        #     debug_path = os.path.join(dst_dir_path, 'debug')
        #     mkdir(debug_path)
        public_path = os.path.join(dst_dir_path, 'public')
        temp_path = os.path.join(dst_dir_path, 'temp')
        mkdir(dst_dir_path)
        mkdir(public_path)
        mkdir(temp_path)

        if ONLY_PATH:
            cc2as_path: CC2AS_PATH = os.path.join(dst_dir_path, 'cc2as')
            cone_path = os.path.join(dst_dir_path, 'asns.json')
            weight_data_path: WEIGHT_PATH = os.path.join(root_path, 'output/weight_data')
        else:
            cc2as_path: CC2AS_PATH = make_cc2as(os.path.join(root_path, 'input'), dst_dir_path)
            cone_path = make_asn_cone_file(os.path.join(root_path, 'input'), dst_dir_path)
            weight_data_path: WEIGHT_PATH = make_as_importance(root_path, cc_list)

        start_time = datetime.now()
        with open(cone_path, 'r') as asn_f:
            asn_data: Dict[AS_CODE, int] = json.load(asn_f)
        time_stamp.write('load asns.json %s  \n' % (datetime.now() - start_time))

        def make_model_path(model_path: str) -> str:
            '''
            model_path:自定义模块的路径

            return 真实的文件路径
            '''
            if model_path in model_map:
                return model_map[model_path]
            else:
                return '.'.join(['input', model_path])

        build_rtree_model_path = make_model_path(build_rtree_model)
        destroy_rtree_model_path = make_model_path(destroy_rtree_model)
        cut_rtree_model_path = make_model_path(cut_rtree_model)
        cal_rank_model_path = make_model_path(cal_rank_model)
        monitor_model_path = make_model_path(monitor_model)

        dynamic_model = import_module(monitor_model_path)
        monitorCountryInternal = dynamic_model.monitor_country_internal

        set_gl_cal_rank_model(cal_rank_model_path)
        for _type in types:
            if not types[_type]:
                continue
            topo_list.append(_type)
            time_stamp.write('------------------- %s start -------------------\n' % _type)
            time_stamp.flush()
            as_rela_file = transformToJSON(dst_dir_path, type_map[_type]['txt_path'], asn_data, rtree_node_min_cone)
            if as_rela_file == '':
                raise Exception('PATH ERROR')
            createRoutingTree(as_rela_file, dst_dir_path, _type, cc_list, asn_data, cc2as_path, build_rtree_model_path)

            if monitor_model == 'RECORD_TREE_AS':
                # 只算树上被影响的节点
                monitorCountryInternal(dst_dir_path, _type, asn_data, destroy_rtree_model_path, cut_rtree_model_path,
                                       cut_node_depth, cc_list)
            elif monitor_model == 'RECORD_COUNTRY_AS':
                #不在树中也算被影响的节点
                monitorCountryInternal(dst_dir_path, _type, asn_data, destroy_rtree_model_path, cut_rtree_model_path,
                                       cut_node_depth, cc_list, cc2as_path)
            else:
                monitorCountryInternal_local(dst_dir_path, _type, asn_data, destroy_rtree_model_path, cut_rtree_model_path,
                                             cut_node_depth, cc_list)
            # # # anova.py
            do_extract_connect_list(dst_dir_path, _type, weight_data_path, cut_node_depth)
            do_groud_truth_based_anova(dst_dir_path, _type, cc_list)
            do_groud_truth_based_var(dst_dir_path, _type, cc_list)
            find_optimize_link(type_map[_type]['txt_path'], dst_dir_path,_type, cone_path, opt_cc_list,
                               weight_data_path)
            ###### find_optimize_link(type_map[_type]['txt_path'], dst_dir_path,_type, cone_path, opt_cc_list,
            ######                    weight_data_path,optimize_link_list,cc2as_path,rtree_node_min_cone)
        do_country_internal_rank(dst_dir_path, cc_list, topo_list, data_aspect)
        train_routing_tree(topo_list, opt_cc_list, dst_dir_path, weight_data_path, optimize_link_list,data_aspect)