import json
import os
from my_types import *
from typing import Dict, List
import jsonlines
from util import mkdir, record_launch_time


@record_launch_time
def make_cc2as(input_path: str, output_path: OUTPUT_PATH) -> CC2AS_PATH:
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
