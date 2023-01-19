# Evaluating and Improving Regional Network Robustness from an AS TOPO Perspective

Currently, regional networks are subject to various security attacks and threats, which can cause the network to fail. 

We borrowed the quantitative ranking idea from the fields of statistics and proposed a ranking method for evaluating regional resilience.

To improve a region’s robustness, we proposed a greedy algorithm to optimize the resilience of regions by adding key links among AS. 

We selected the AS topology of 50 countries/regions for research and ranking, evaluating the topology robustness from connectivity, user, and domain influence perspectives, clustering  the  results and get typical region types, and adding  optimal links to improve the network resilience. Experimental results illustrate  that the resilience of regional networks can be greatly improved by establishing a few new connections, which demonstrates the effectiveness of the optimization method.

## Quickstart

1. install python dependencies

```shell
pip install --user -r requirements.txt
```

2. run

```shell
python3 main.py <folder>
```

<span style="color:red">Tips: `folder` should have an input folder where the `input` files are stored. See the Input File Template for details. </span>


## Structure

<table>
<tr><td>main.py</td><td>An entry program for calculating intra-region resistance </td></tr>  
  <tr><td>main_external.py</td><td>An entry program for calculating interzone inter-region resistance</td></tr>  
  <tr><td>Internal_input_example</td><td>An input file template for calculating the intra-region resistance</td></tr>  
  <tr><td>External_input_example</td><td>An input file template for calculating the inter-region resistance</td></tr>  
  <tr><td>do_Internal</td><td>The source code of the intra-region resistance</td></tr>  
  <tr><td>do_External</td><td>The source code of the inter-region resistance</td></tr>  
  <tr><td>other_script</td><td>Auxiliary code</td></tr>  
  <tr><td>static</td><td>Static file</td></tr>  
</table>

## Contact

You can contact us at liuyujia19.tsinghua.org.cn.

