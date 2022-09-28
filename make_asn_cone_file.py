import jsonlines
import json
import os


def make_asn_cone_file(input_path, output_path):
    result = {}
    with open(os.path.join(input_path, 'asns.jsonl'), 'r') as f:
        for i in jsonlines.Reader(f):
            cc = i['country']['iso']
            cone = i['cone']['numberAsns']
            rank = i['rank']
            asn = i['asn']
            if asn not in result:
                result[asn] = cone

    with open(os.path.join(output_path, 'asns.json'), 'w') as ff:
        json.dump(result, ff)
    return os.path.join(output_path, 'asns.json')