def do_cal(res_cc,ans):
    try:
        return sum([res_cc[_as] * ans[_as] for _as in res_cc])/sum(ans.values())
    except Exception as e:
        print('Exception happen: ',e)
        return sum(res_cc.values())/len(res_cc)