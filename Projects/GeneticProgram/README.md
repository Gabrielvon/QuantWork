# GP策略软件说明

## 文件结构
```
├─v5
│  ├─dataset
│  │  ├─data
│  │  ├─ga_results
│  │  │  ├─rankIC
│  │  │  └─sharpe
│  │  └─live
│  │      ├─data
│  │      │  ├─industry
│  │      │  └─stock
│  │      ├─executions
│  │      └─ga_results
│  │          ├─rankIC
│  │          │   └─raw
│  │          └─sharpe
│  │              └─raw
│  ├─dependencies
│  ├─backtest.py
│  ├─factor_mining_mp.py
│  ├─factor_update_fast.py
│  ├─factor_update_full.py
│  ├─order_generator.py
│  ├─order_scan_by_csv.py
│  ├─requirements.txt
│  ├─save_data_to_local.py
```

- dataset
	- data: 行情数据文件，按类型和日期分类
	- ga_results: GP挖掘因子后生成，按评价指标分类
	- live: 用于实盘交易
		- data: 行情数据文件，按类型和日期分类
		- executions: 
		- ga_results: GP挖掘因子后生成，按评价指标分类
- dependencies: 软件依赖
- backtest.py: 回测脚本
- factor_mining_mp.py: GP因子挖掘，保存结果到`dataset/ga_results`中。
- factor_update_fast.py: 基于每周新挖掘的因子对因子库进行迭代更新，保存结果到`dataset/live/ga_results`中。
- factor_update_full.py: 对因子库进行整体更新，保存结果到`dataset/ga_results`中。
- order_generator.py: 生成第二天交易信号，保存结果到`dataset/live/executions`中。
- order_scan_by_csv.py: 通过掘金进行交易
- save_data_to_local.py: 从聚宽下载行情数据并保存到本地，保存结果到`dataset/data`中


注意
- 以rankIC为例，dataset/live/ga_results/rankIC中，子目录中的数据文件为每次更新后的因子，子目录raw文件夹中保存的是每周挖掘的新因子，未加入因子库进行更新



## 使用说明

### 安装依赖

```
pip install -r dependencies/requirements.txt
pip install dependencies/TA_Lib-0.4.19-cp38-cp38-win_amd64.whl
```

### 数据源

数据接口使用聚宽的，需要聚宽的相关权限。

### 交易接口

交易执行使用掘金扫单功能，需要掘金的相关权限。

### 脚本运行顺序

回测情景
1. 下载数据 - save_data_to_local.py
2. 因子挖掘 - factor_mining_mp.py
3. 因子更新 - factor_update_full.py
4. 策略回测 - backtest.py

实盘情景
1. 因子更新 - factor_update_fast.py
2. 信号生成 - order_generator.py
3. 交易执行 - order_scan_by_csv.py

