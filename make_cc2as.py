import json
import os
import jsonlines


def make_cc2as(input_path,output_path):
    result = {}
    cc2as_path = os.path.join(output_path,'cc2as')
    if not os.path.exists(cc2as_path):
        os.mkdir(cc2as_path)
    
    with open(os.path.join(input_path,'asns.jsonl'),'r') as f:
        for i in jsonlines.Reader(f):
            cc = i['country']['iso']
            asn = i['asn']
            if not cc:
                continue
            # print(cc)
            if cc not in result:
                result[cc] = []
            result[cc].append(asn)
    for ccc in result:
        # print(ccc,result[ccc])
        with open(os.path.join(cc2as_path,'%s.json' % ccc),'w') as cf:
            json.dump(result[ccc],cf)
    return cc2as_path

