import json
import os
from other_script.my_types import *
from typing import Dict, List
import jsonlines
from other_script.util import mkdir, record_launch_time


@record_launch_time
def make_cc2as(input_path: str, output_path: OUTPUT_PATH) -> CC2AS_PATH:
    '''
    input_path:input文件夹路径
    input_path:output文件夹路径

    根据input中的asns.jsonl生成cc2as文件夹,用来存放as和国家的对应关系文件

    return as和国家对应关系文件夹路径
    '''
    result: Dict[cc:str, List[asn:str]] = {}
    cc2as_path: str = os.path.join(output_path, 'cc2as')
    mkdir(cc2as_path)

    with open(os.path.join(input_path, 'asns.jsonl'), 'r') as f:
        for i in jsonlines.Reader(f):
            cc = i['country']['iso']
            asn = i['asn']
            if not cc:
                continue
            if cc not in result:
                result[cc] = []
            result[cc].append(asn)
    for ccc in result:

        with open(os.path.join(cc2as_path, '%s.json' % ccc), 'w') as cf:
            json.dump(result[ccc], cf)
    return cc2as_path
