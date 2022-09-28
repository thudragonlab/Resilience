import json
import os
from util import record_launch_time

@record_launch_time
def as_rela_txt(dsn_path,txt_path):
    '''
    处理AS关系数据
    {as:[peer, customor], ......}
    '''
    file = open(txt_path,'r')
    file_name = txt_path.split('/')[-1][:-4]
    # dsn_path = '/home/peizd01/for_dragon/new_data_pzd/'
    real_dst_path = ''
    result = {}
    try:
        for line in file:
            if line[0]!='#':
                line = line.strip().split('|')
                a = line[0]
                b = line[1]
                if line[-1]=='1': continue
                if a not in result.keys():
                    result[a] = [[], []]
                if line[-1]=='-1':
                    result[a][1].append(b)
                else:
                    result[a][0].append(b)
        real_dst_path = os.path.join(dsn_path,file_name+'.json')
        with open(real_dst_path,'w') as f:
            json.dump(result, f)
            print(real_dst_path)
            f.close()
    finally:
        file.close()
    return real_dst_path


# as_rela_txt()