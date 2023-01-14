from typing import Dict
import jsonlines
import json
import os
from other_script.my_types import *
from other_script.util import record_launch_time


@record_launch_time
def make_asn_cone_file(input_path: str, output_path: str) -> str:
    '''
    input_path:input文件夹路径
    input_path:output文件夹路径

    根据input中的asns.jsonl生成as和cone的对应关系文件

    return as和cone关系文件路径
    '''
    result: Dict[AS_CODE, int] = {}
    with open(os.path.join(input_path, 'asns.jsonl'), 'r') as f:
        for i in jsonlines.Reader(f):
            cone = i['cone']['numberAsns']
            asn = i['asn']
            if asn not in result:
                result[asn] = cone

    with open(os.path.join(output_path, 'asns.json'), 'w') as ff:
        json.dump(result, ff)
    return os.path.join(output_path, 'asns.json')