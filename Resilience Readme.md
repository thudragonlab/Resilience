# 获取traceroute数据

## RIPE Atlas获取数据

```python
# 跑 get_traceroute.py
```


## IP地理定位

```python
# 跑 get_location.py
```
#### 函数：creatfile_4()
- 功能：ipv4数据中，把四个数据库的定位信息汇总并存储到文件
- 调用：creatfile_4('/data/lyj/shiyan/AS-RTT-v4-temp/*****')

#### 函数：change()
- 功能：将creatfile_4处理后的定位信息，按照投票标准得到最终的定位，存储文件
- 调用：change('/data/lyj/shiyan_database/ASInfov4_database/*****')

#### 函数：change_file_format()
- 功能：转换存储格式，用原来的json变为txt
- 调用：change_file_format('/data/lyj/shiyan_database/ASInfov4_new/*****')

# 数据来源
## AS关系
- asrank数据集 https://publicdata.caida.org/datasets/as-relationships/serial-1/
- problink数据集 https://yuchenjin.github.io/problink-as-relationships/ （github地址 https://github.com/YuchenJin/ProbLink）
- toposcope数据集 http://rma.cs.tsinghua.edu.cn/toposcope/download.html (包含toposcope-hidden)

## 资源权重
- 用户比例 https://stats.labs.apnic.net/aspop/
- Prefix_weight https://prefixtoplists.net.in.tum.de/20190801/
- 通过上面两个文件可以算出as_importance的权重（用户影响权重，域名影响权重）
- 域名重要性 论文：Prefix top lists: Gaining insights with prefixes from domain-based top lists on dns deployment

# 区域内部量化抗毁性代码运行步骤


## 准备数据

```public/cc2as``` 记录了AS与国家的对应关系

```prefix_weight/as_importance```记录了每个国家下AS的用户重要性和域名重要性



## 配置

```json
{
    "root":"/home/peizd01/for_dragon/pzd_test1",//根路径
    "types":{
        "asRank":true,
        "problink":false,
        "toposcope":false,
        "toposcope_hidden":false
    },//要跑的类型，设置为False会跳过
    "cc_list":["LV", "HU", "CL", "SK", "PT", "FI", "CO", "TR", "PH", "NG", "DK", "MX", "LU", "KH", "TW", "TH", "KR", "NO", "MY", "CN", "BG", "IR", "IE", "AR", "RO", "NZ", "CZ", "BD", "SE", "UA", "SG", "HK", "ES", "AT", "JP", "IN", "ID", "CH", "PL", "NL", "IT", "FR", "CA", "ZA", "AU", "GB", "DE", "US", "RU", "BR"]//要跑的国家列表
}
```

<span style="color:red">注：要跑的数据放在{root}/input路径下</span>

​	asRank的数据命名为**asRank.txt**

​	problink的数据命名为**problink.txt**

​	toposcope的数据命名为**toposcope.txt**

​	toposcope_hidden的数据命名为**toposcope_hidden.txt**

## 运行

运行 ```/home/peizd01/for_dragon/pzd_python/do_Internal/main.py```

输出路径```dst_path```为{root}/output



## main.py中详细步骤

```python
#把txt文件转成json，作为下一步的输入
as_rela_file = transformToJSON(txt_path)
#输出文件在{dst_path}/下与txt文件同名的json文件
```



### 生成每个国家的routingTree

```python
# 创建路由树
createRoutingTree(as_rela_file,dst_path,_type['type'])
'''
在{dst_path}下创建对应的asRank，problink...文件夹，在这个文件夹中创建存储路由树的rtree文件夹
{dst_path}/{type}/rtree/{country_code}/as_rel.txt记录了这个国家下所有AS的关系
具体表示如下，
52307|52308|0
52363|61470|-1

52307 与 52308 是 p2p关系
52363 与 61470 是 p2c关系

<peer-as>|<peer-as>|0
<provider-as>|<customer-as>|-1
'''

'''
{dst_path}/{type}/rtree/{country_code}/dcomplete{as_code}.npz 记录的是对应{as_code}的路由树
'''
```



### 随机模拟破坏，得到破坏结果

```python
# 模拟破话节点
monitorCountryInternal(dst_path,_type['type'])


'''
会生成{dst_path}/{type}/rtree/{country_code}/dcomplete{as_code}.addDel.txt 记录的是对{as_code}的破坏情况
具体结构如下
dcomplete27674.addDel.txt

262187|262187
27976|27976 52430

破坏以27674为根结点的路由树中的262187,会影响262187节点
破坏以27674为根结点的路由树中的27976,会影响27976，52430节点
'''

'''
会生成{dst_path}/{type}/rtree/{country_code}/dcomplete{as_code}.graph.json 是对应{dst_path}/{type}/rtree/{country_code}/dcomplete{as_code}.npz转化为图的形式
数据结构为
{
'as_code':[[前向点][后向点]],
}
'''
```



### 计算每个破坏结果的破坏影响

```python
# 跑这个生成的中间数据（每个AS的域名重要性，用户重要性，联通节点个数）
do_extract_connect_list(dst_path,_type['type'])
'''
路径为{dst_path}/{type}/result/count_num/{country_code}/dcomplete{as_code}.json 为破坏{coutry_code}下{as_code}对应的路由树的影响

其中的数据结构为
{
 "dcomplete{as_code}":{
 "asNum":路由树链接的节点数量,
 "connect":[
 [
	 [受影响的节点数量，用户维度重要性，域名维度重要性],
	 [受影响的节点数量，用户维度重要性，域名维度重要性]
 ],  破坏这颗树上的一个节点的影响
 [], 破坏这颗树上的两个节点的影响
 [], 破坏这颗树上的三个节点的影响
 [], 破坏这颗树上的四个节点的影响
 ]
 }
}
'''

```



### anova排序

```python
# 生成每个AS的中位数排名数据
do_groud_truth_based_anova(dst_path,_type['type'])
'''
路径为
{dst_path}/{type}/result/anova/sorted_country_basic.json
{dst_path}/{type}/result/anova/sorted_country_user.json
{dst_path}/{type}/result/anova/sorted_country_domain.json

数据结构：
[
[排名第一的AS列表],
[排名第二的AS列表],
[排名第三的AS列表],
[排名第四的AS列表],
...
]

注：这个是所有国家的AS一起进行的排名
'''
```



### 方差排名

```python
# 生成每个AS的方差排名数据
do_groud_truth_based_var(dst_path,_type['type'])
'''
路径为
{dst_path}/{type}/result/var/sorted_country_basic.json
{dst_path}/{type}/result/var/sorted_country_user.json
{dst_path}/{type}/result/var/sorted_country_domain.json

数据结构：
[
[排名第一的AS列表],
[排名第二的AS列表],
[排名第三的AS列表],
[排名第四的AS列表],
...
]

注：这个是所有国家的AS一起进行的排名
'''
```





### 生成国家排名

```python
# 生成国家排名
do_country_internal_rank(dst_path)
'''
生成文件路径为{dst_path}/public/new_rank.json 中位数排名
生成文件路径为{dst_path}/public/var_rank.json 方差排名

数据结构为:{
country_code:[
asRank连通性排名，
problink连通性排名，
toposcope连通性排名，
toposcope_hidden连通性排名，
asRank用户维度排名，
problink用户维度排名，
toposcope用户维度排名，
toposcope_hidden用户维度排名，
asRank域名维度排名，
problink域名维度排名，
toposcope域名维度排名，
toposcope_hidden域名维度排名，
]
}
'''


```



# 区域内部抗毁性优化运行步骤
计算优化链接
```python
# 跑find_optimize_link.py

# 在536行将文件路径换为最新数据的txt文件路径
# 572-575行，更换文件路径为新的rtree路径
# dsn_path 可根据需求自行更改
```
计算加入优化链接后的路由树、模拟破坏结果、anova排序结果、与原排名比较
```python
# 跑train_routing_tree.py
# 修改 prefix 修改为新的路径（459行）
# 跑完之后优化后的数据在 optimize_link/cost_0-benefit_basic-state_p2p-num_3/floyed 路径下
```



# 区域外部抗毁性代码运行步骤

### step1:

 跑

```python
bar = broad_as_routingtree(os.path.join(prefix, \
        'ccExternal/globalCountryLabel/add_hidden_link/as_rela_code.txt'))
broad_as_routingtree.cal_rtree_code()
```



会生成as_rela_code.txt， as-country-code.json， cal_rtree_code_v2.json



### step2

用生成的as_rela_code.txt作为路由树的**输入**，使用**之前的**路由树的代码，生成路由树。

**只跑边界AS的路由树**，cal_rtree_code_v2.json中是需要跑的边界AS。



```shell
# 跑 create_routingtree_external.py
# as_rela_file 用上面的as_rela_code.txt （510行）
# 367行读取文件用上面的cal_rtree_code_v2.json
```







### step3

跑broad_as_routingtree.remove_cc_internal_link



### step4

跑_monitor_remove_as



预处理，生成国家-AS的编码，以及连接关系

```python
# 跑topology_external.py
```

计算排名
```python
# 跑external_security.py
```


