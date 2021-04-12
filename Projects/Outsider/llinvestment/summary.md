# 实习记录 



# 实习项目

## 基于ML方法形成策略

- BENCHMARK: HS300最近一个月的日收益
- 策略构思
	- 数据种类：量价数据（包括盘口和逐笔订单）
	- 数据频率：日内数据日间化
	- 交易频率：日间，最高是每日调仓
	- 策略方式：先选股后择时
- 自动化逻辑
	- 选股
		- 基于择时适合的场景选取因子，通过因子找出让择时策略更优的标的
	- 择时
		- 基于资金流动选取因子，通过因子抓住标的的价格趋势
	- 因子组合
		- DL
	- 其他
		- 所有因子基于逻辑挖掘（3~5个）
		- 至少一个因子使用到时序关系
		- 使用
- 工作顺序
	- 择时
	- 择时因子组合（若时间充裕）
	- 选股（若时间充裕）
	- 选股因子组合（若时间充裕）


## 实际工作总结
- 构造因子
	- 数据集: orderbook, dealbook
	- 基于逻辑（市场动作，研报）
- 因子检测
	- ic，以时序上为主
		- IC(绝对值大小)
		- IC显著性(t test)
		- ICIR
	- IC稳定性(unit test)
		- adfuller
	- 单因子收益（基于历史值滚动设定开平仓阈值）
- 因子组合
	- 方式选择
		- lasso
		- rf
		- xgboost
		- auto-keras(dl)
		- tpot(fe + dl)
		- deep-forest
		- 因子正交化后基于icir值作为权重(还未完成)
	- 建模记录
		- rf非常容易过拟合，需要给与高一点的惩罚
		- 初试中auto-keras效果相对较好


## 研究过程中遇到的问题

数据相关
- 20210224-sz002024数据异常(这天涨停)。盘口数据为0。
	- 收益率填充为0

系统相关
- 画面卡顿
- 左侧屏幕经常黑屏无法恢复，重启或重插显示器
- 进程开太多，CPU使用率过高，机器自动重启
- 网络问题
	- 无法使用pip安装```WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1125)'))': /simple/xgboost/
```

瓶颈
- 内存太小
	- 牺牲IO速度
- 数据“脏”
	- 数据集约定需要探索 --> 耗费时间
		- 如通过数据集找出异常股票，涨跌停，停牌等。
	- 有些数据有ohlc和昨收，有些数据没有


## 进一步
- 因子精细化处理
	- 标准化：归一化，中心化，去极值等
- 模型
	- 自定义loss function


## 未来策略发展
- 因子挖掘
	- 方法
		- 逻辑
		- GP（客观有效的评价指标有待商榷）
	- 加入其他数据种类
		- 如市值，行业，资金流动性等
- 因子优化
	- 中性化
	- 对每个因子进行精细化调整，控制分布，值域等
	- 指定具备调参因素的因子，通过GA找出最优参数
	- 是否需要加速
- 因子组合
	- 去相关性（正交）
	- ML/DL


## 团队建议与优化地方
- 建个数据库，可以用parquet等储存方式。
- tick数据没有对齐


## Q&A
Q: 中低频因子的ic值一般多大
A: 4%左右

Q: 会用些什么评价指标
A: 绝对收益+最大回撤+IC

Q: 选股中评价因子分层收益的指标
A: 一般很难通过一个维度去定义量化指标进行评价，也是通过多个维度，最直观就是看图。

Q: 对于ic显著负的因子会被认为有效吗？
A: ？

Q: 相关性多高的因子，认为同一类因子？应该分因子大类吗？
A: ？


## Others

pip镜像
- references
	- https://www.cnblogs.com/zhangruoxu/p/6370107.html
- 国内常用镜像
	- 阿里云 http://mirrors.aliyun.com/pypi/simple/
	- 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
	- 豆瓣(douban) http://pypi.douban.com/simple/
	- 清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
	- 中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/
- shell 
```shell
# pip修改默认镜像
pip config --user set global.index-url http://pypi.douban.com/simple/
pip config --user set global.trusted-host pypi.douban.com
pip config --user set global.index-url http://mirrors.aliyun.com/pypi/simple/
pip config --user set global.trusted-host mirrors.aliyun.com
# pip指定镜像
pip install tpot -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install tpot -U -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

```
