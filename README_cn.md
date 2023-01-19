# 评估以及优化区域拓扑的抗毁性
区域网络受到各种安全攻击和威胁，有可能导致网络故障。本文借鉴统计学领域的定量排序思想，提出了一种对区域弹性的排序方法。

为了提高区域的抗毁性，我们提出了一种贪婪算法，通过在AS之间增加关键链路来优化区域的弹性。

我们选取了50个国家/地区的AS拓扑进行研究和排名，从连通性、用户和域影响角度评价拓扑的鲁棒性，将结果聚类得到典型的区域类型，并添加最优链路提高网络弹性。实验结果表明，通过建立少量新连接可以大大提高区域网络的弹性，验证了优化方法的有效性。

## 快速开始

1. 下载python依赖库

```shell
pip install --user -r requirements.txt
```

2. 运行
```shell
python3 main.py <folder>
```

<span style="color:red">注：运行目录`folder`下要有一个input文件夹，里面存放输入文件，具体见输入文件模版 </span>


## 代码结构
<table>
<tr><td>main.py</td><td>计算域内抗毁性的入口程序</td></tr>  
  <tr><td>main_external.py</td><td>计算域间抗毁性的入口程序</td></tr>  
  <tr><td>Internal_input_example</td><td>计算域内抗毁性输入文件模版</td></tr>  
  <tr><td>External_input_example</td><td>计算域间抗毁性输入文件模版</td></tr>  
  <tr><td>do_Internal</td><td>计算域内抗毁性程序源码</td></tr>  
  <tr><td>do_External</td><td>计算域间抗毁性程序源码</td></tr>  
  <tr><td>other_script</td><td>其他辅助代码</td></tr>  
  <tr><td>static</td><td>静态文件</td></tr>  
</table>



## 联系我们
欢迎通过邮件（liuyujia19@tsinghua.org.cn）联系我们。

