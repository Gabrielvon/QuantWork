


# Logs:
20180518 **Major Updates**
1. 对详尽版本进行重大更新，debug后使其保持与压缩版本处理方法一致；
2. 对两个功能版本都添加了以下功能：
    * 按quantiles分段的分组方式；
    * 在循环计算因子的部分，加入了因子缺失值处理的功能，用户有两个选项可用，填0或整行删除；

# Condensed（压缩）

## Functions：
1. 将原始数据压缩储存，对压缩数据进行处理，节省空间。

## Versions
Most updated versions
* Local: v32
* Notebook: v40


# Elaborated（详尽）

## Functions：
1. 直接对原始数据进行储存和处理，虽然准确性高，但资源消耗极大。

## Versions
Most updated version1:
* Local: v22
* Notebook: v20