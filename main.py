import os
# from do_Internal.create_routingtree  import main as createRoutingTree
# from do_Internal.create_routingtree_only_10  import main as createRoutingTree
from do_Internal.create_routingtree_random_100  import main as createRoutingTree
from do_Internal.data_analysis  import as_rela_txt as transformToJSON
from do_Internal.monitor_random_cut import monitor_country_internal as monitorCountryInternal
# from do_Internal.monitor_random_cut_full_tree import monitor_country_internal as monitorCountryInternal
# from do_Internal.monitor_random_cut_full_tree_1000_times import monitor_country_internal as monitorCountryInternal

from do_Internal.anova import do_extract_connect_list,do_groud_truth_based_anova,do_country_internal_rank,do_groud_truth_based_var
from do_Internal.make_as_importance_weight import main as make_as_importance
from do_Internal.make_cc2as import make_cc2as as make_cc2as
import json
from datetime import datetime



if __name__ == '__main__':
    topo_list = []
    start_time = datetime.now()
    with  open('time_stamp.txt','a') as time_stamp:
        with open('config.json','r') as config_file:
            config = json.load(config_file)
            root_path = config['root']
            types = config['types']
            cc_list = config['cc_list']
        time_stamp.write('\n-------------------------%s-------------------------\n' % datetime.now())
        time_stamp.write('%s\n' % root_path)
        time_stamp.write('load config.json %s \n' % (datetime.now() - start_time))
        type_map = {
        "asRank":{
            'type':'asRank',
            'txt_path':os.path.join(root_path,'input/asRank.txt')
        },
        "problink":{
            'type':'problink',
            'txt_path':os.path.join(root_path,'input/problink.txt')
        },
        "toposcope":{
            'type':'toposcope',
            'txt_path':os.path.join(root_path,'input/toposcope.txt')
        },
        "toposcope_hidden":{
            'type':'toposcope_hidden',
            'txt_path':os.path.join(root_path,'input/toposcope_hidden.txt')
        }}
        
        

        dst_dir_path = os.path.join(root_path,'output')
        if not os.path.exists(dst_dir_path):
                os.mkdir(dst_dir_path)
        public_path = '%s/public'  % dst_dir_path
        if not os.path.exists(public_path):
            os.mkdir(public_path)
        cc2as_path = make_cc2as(os.path.join(root_path,'input'),dst_dir_path)
        print(cc2as_path)
        start_time = datetime.now()
        with open(os.path.join(root_path,'input','asns.json'),'r') as asn_f:
            asn_data = json.load(asn_f)
        time_stamp.write('load asns.json %s  \n' % (datetime.now() - start_time))
        start_time = datetime.now()
        weight_data_path = make_as_importance(root_path,cc_list)
        time_stamp.write('make_as_importance  %s  \n' % (datetime.now() - start_time))

        for _type in types:
            time_stamp.write('------------------- %s start -------------------\n' % _type)
            if not types[_type]:
                continue
            topo_list.append(_type)
            start_time = datetime.now()
            as_rela_file = transformToJSON(dst_dir_path,type_map[_type]['txt_path'])
            time_stamp.write('transformToJSON  %s  \n' % (datetime.now() - start_time))
            if as_rela_file == '':
                raise Exception('PATH ERROR')
            # print('!!!!')
            start_time = datetime.now()
            createRoutingTree(as_rela_file,dst_dir_path,_type,cc_list,asn_data,cc2as_path)
            time_stamp.write('createRoutingTree  %s  \n' % (datetime.now() - start_time))
            start_time = datetime.now()
            monitorCountryInternal(dst_dir_path,_type,asn_data)
            time_stamp.write('monitorCountryInternal  %s  \n' % (datetime.now() - start_time))
            # anova.py
            start_time = datetime.now()
            do_extract_connect_list(dst_dir_path,_type,weight_data_path)
            time_stamp.write('do_extract_connect_list  %s  \n' % (datetime.now() - start_time))
            start_time = datetime.now()
            do_groud_truth_based_anova(dst_dir_path,_type)
            time_stamp.write('do_groud_truth_based_anova  %s  \n' % (datetime.now() - start_time))
            start_time = datetime.now()
            do_groud_truth_based_var(dst_dir_path,_type)
            time_stamp.write('do_groud_truth_based_var  %s  \n' % (datetime.now() - start_time))
            time_stamp.write('------------------- %s end ------------------- \n' % _type)
        do_country_internal_rank(dst_dir_path,cc_list,topo_list)
        

