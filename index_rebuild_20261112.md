# 论文索引重建 2026-11-12

> **⚠️ 隔离执行横幅 (2026-11-13, 用户批准)**: §2.2 定谳的 155 篇错配 PDF 已整体移入
> `pdfs/_mismatched/` (非破坏性 mv, 含唯一同域例外 #148 量子论文)。`pdfs/` 现仅存
> 名实相符的 76 篇。§1 索引行原样保留作出处档案。详见 `pdfs/_mismatched/README.md`。

来源: /home/muller/git/papers/pdfs/ 全部 231 篇 PDF 前 2 页原文文本 (pdftotext -l 2)。
方法: 独立轻扫重建索引 (先于旧分析对账, 索引行全部来自论文原文前2页, 不从文件名脑补)。
相关度评级标准 (研究方向: A股/港美股量化, 因子/微观结构/组合构造/市场结构/出场入场机):
- 高 = 因子模型、市场微观结构、组合构造、截面定价、可迁移的市场结构证据
- 中 = 股价预测但含可参考的构造/评估方法、情绪与另类数据
- 低 = 纯预测流派(LSTM/GNN预测价格类)、非金融论文(海温/山火/PM2.5/刀具磨损等)

## §1 索引总表

| # | 文件名(截短) | 年份 | 题目 | 方法族 | 一句话结论 | 相关度 |
| 1 | 2018_1810.01278_DeepFactorModel | 2018 | Deep Factor Model: Explaining DL Decisions for Forecasting Stock Returns with LRP | 因子模型+DL可解释 | 日本市场多因子+LRP归因,预测力优于线性模型与其它ML | 高 |
| 2 | 2018_1902.04994_ExplainableTextNN | 2018 | Explainable Text-Driven Neural Network for Stock Prediction | 情绪/新闻+GRU注意力 | 知识库过滤新闻+双层注意力GRU预测个股,附事件级解释 | 中 |
| 3 | 2018_c9bb67d5abc6_GoldPricePCA-GA-BP | 2016 | [内容错配] A Tutorial about Random Neural Networks in Supervised Learning | NN教程 | 文件名"金价PCA-GA-BP预测"与内容不符,实为随机神经网络监督学习教程 | 低 |
| 4 | 2019_063562746389_GBDT_insider | 2025 | [内容错配] Online GBDT: In-Place Updates for Efficient Adding/Deleting Data | GBDT/在线学习 | 文件名"GBDT内幕交易"与内容不符;实为GBDT增量/减量学习框架(通用ML) | 低 |
| 5 | 2019_0dacfc79f5e3_TechFundNews | 2024 | [内容错配] Sentiment Analysis & Stock Price Using FinBERT, GPT-4 and Logistic Regression | 情绪分析 | 文件名"技术/基本面/新闻预测"与内容不符;实为尼日尼亚股指情绪分类,LR(81.8%)胜FinBERT/GPT-4 | 中 |
| 6 | 2019_1901.11493_DeepRecurrentFactor | 2019 | Deep Recurrent Factor Model: Interpretable Non-Linear Time-Varying Multi-Factor | 因子模型+LSTM | LSTM+LRP把线性多因子扩展为非线性时变,日本市场优于线性/全连接 | 高 |
| 7 | 2019_1904.12834_PriorDomainKnowledge | 2021 | Incorporating Prior Financial Domain Knowledge into NN for IVS Prediction | 期权定价+NN | 波动率曲面预测嵌入无套利/边界/渐近斜率先验,S&P500 20年数据胜基准 | 中 |
| 8 | 2019_1908.03385_GBST_credit | 2021 | Gradient Boosting Survival Tree with Applications in Credit Scoring | 生存分析/信用 | 中国消费金融大数据,GBST按C-index/KS/AUC胜既有生存模型 | 低 |
| 9 | 2019_310b54f1913a_HybridNN_vol | 2016 | [内容错配] A Tutorial about Random Neural Networks (同#3) | NN教程 | 文件名"混合NN波动率预测"与内容不符,实为同一篇RNN教程 | 低 |
| 10 | 2019_3cdce280ad29_ML_vs_jump | 2008 | [内容错配] Expected Performance of the ATLAS Experiment (CERN) | 粒子物理 | 文件名"ML vs 计量跳跃模型"与内容不符;实为CERN ATLAS探测器性能报告 | 低 |
| 11 | 2019_42a00980b2d5_ANN_RV | 2010 | [内容错配] Existence & Global Logarithmic Stability of Impulsive NN with Time Delay | 数学NN理论 | 文件名"ANN已实现波动率预测"与内容不符;实为时滞脉冲NN稳定性证明 | 低 |
| 12 | 2019_84b7181b0326_FeatureSelection | 2023 | [内容错配] Changing Data Sources in the Age of ML for Official Statistics | 官方统计/ML治理 | 文件名"技术指标最优特征选择"与内容不符;实为官方统计换数据源风险清单 | 低 |
| 13 | 2019_875f55c7bba0_Futures_price | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity | 零售定价ML | 文件名"期货价格预测决策"与内容不符;实为Walmart商品价格弹性估计 | 低 |
| 14 | 2019_9ee3524f00d3_StockPriceMovClass | 2026 | [内容错配] Monodense Item Price Elasticity (同#13) | 零售定价ML | 文件名"机器/深度学习股价分类"与内容不符;同一篇Walmart弹性论文 | 低 |
| 15 | 2019_b7954dd4ba98_GatedDNN_IVS | 2023 | [内容错配] Deep Arbitrary Polynomial Chaos Neural Network | NN架构 | 文件名"门控DNN隐含波动率曲面"与内容不符;实为aPC正交基NN架构 | 低 |
| 16 | 2019_cf1ea6b89b89_TransformerCapsule | ~2019-21 | TI-Capsule: Capsule Network for Stock Exchange Prediction (文件名近似) | 多模态深度学习 | 文本+蜡烛图胶囊网络预测EUR/USD方向,自称91%准确率 | 中 |
| 17 | 2019_e3f06f90788d_RejectInference | ~2013 | [内容错配] Semantic-based Detection of Segment Outliers for WSN | 传感器异常检测 | 文件名"拒绝推断 outlier+G"与内容不符;实为无线传感网段异常检测 | 低 |
| 18 | 2020_2004.01878_NewsDrivenNoisyRST | 2020 | News-Driven Stock Prediction With Attention-Based Noisy Recurrent State Transition | 新闻/NLP | 把股价基础状态建成循环态,注意力聚合新闻+显式噪声项,胜强基线 | 中 |
| 19 | 2020_2005.04955_MultiGraphGCN | 2020 | Multi-Graph Convolutional Network for Relationship-Driven Stock Movement | GNN+关系图 | 中国市场多关系图(行业/供应链等)+GCN+GRU建模股票间lead-lag,胜基线 | 中 |
| 20 | 2020_2069caba585a_TechnicalDebt | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb) | 粒子物理 | 文件名"技术债与软件安全风险"与内容不符;实为粒子物理稀有衰变观测 | 低 |
| 21 | 2020_20b5874e5421_NewsBERT | 2023 | [内容错配] Emoji Prediction in Tweets using BERT | NLP/社交媒体 | 文件名"金融新闻+BERT股价趋势"与内容不符;实为推文表情符号预测 | 低 |
| 22 | 2020_2259a212d076_DL_FinTS | 2024 | [内容错配] LLMs for Financial Aid in Financial Time-series Forecasting | LLM时序 | 文件名"DL金融时序预测"与内容不符;实为GPT-2骨干在学生资助等8个时序few/zero-shot基准 | 低 |
| 23 | 2020_2912ddcb28f3_Bitcoin_DL | 2015 | [内容错配] The Economics of BitCoin Price Formation | 加密资产定价 | 文件名"比特币深度学习预测"与内容不符;实为供需基本面+投资者吸引力驱动比特币价格的计量分析 | 中 |
| 24 | 2020_55c158af1fa0_AgriFutures | 2023 | [内容错配] Changing Data Sources in ML for Official Statistics (同#12) | 官方统计/ML治理 | 文件名"农产品期货预测"与内容不符;同一篇官方统计论文 | 低 |
| 25 | 2020_6376dbbcf94c_AdvancedDL_ShortTerm | 2023 | [内容错配] Comparative Assessment of DL Models for Day-Ahead Load Forecasting | 电力负荷预测DL | 文件名"高级DL短期预测"与内容不符;实为葡萄牙电网日前负荷预测,N-BEATS最优 | 低 |
| 26 | 2020_7a7ebedad123_SST_DL | 2020 | [内容错配] A Learning Framework for n-bit Quantized NNs toward FPGAs | 量化NN/硬件 | 文件名"海温DL预测"与内容不符;实为FPGA量化神经网络框架 | 低 |
| 27 | 2020_9dd870a475dd_SurfaceRoughness | 2018 | [内容错配] Comparing Emission Spectra of U and Th Hollow Cathode Lamps | 天体物理 | 文件名"切削参数表面粗糙度预测"与内容不符;实为天文光谱波长定标 | 低 |
| 28 | 2020_b36e0664b1a5_DynamicCreditScoring | ~2019 | [内容错配] Classification of Imbalanced Credit Scoring via WHSBoost | 信用评分/集成 | 文件名"生存梯度提升动态信用"与内容不符;实为加权混合采样Boost处理不平衡信贷数据 | 低 |
| 29 | 2020_b5118e0e9b2a_SST_MonthlyQ | ~2018 | [内容错配] Predicting Air Pollutants by Transfer Learning and RNN | 环境PM预测 | 文件名"月/季海温预测"与内容不符;实为澳门空气质量LSTM+迁移学习 | 低 |
| 30 | 2020_b741167bb65d_StockRelationGraph | 2022 | Understanding Stock Market Instability via Graph Auto-Encoders (NeurIPS) | 图自编码/市场结构 | 文件名"隔夜图网络"不符;GAE边重构误差作联动崩塌代理,提升S&P500样本外波动率预测 | 高 |
| 31 | 2020_c82ef469a3b5_LookbackPeriod | 2015 | [内容错配] Observation of Bs0→μ+μ− decay (CMS+LHCb) (同#20) | 粒子物理 | 文件名"回看期分析"与内容不符;同一篇CERN粒子物理论文 | 低 |
| 32 | 2020_d15835c4c5fc_GoldPrice_DL | 2026 | [内容错配] Monodense Item Price Elasticity (同#13) | 零售定价ML | 文件名"金价DL建模"与内容不符;同一篇Walmart弹性论文 | 低 |
| 33 | 2020_e8faa912a86d_CrossBorderEcom | 2026 | [内容错配] Monodense Item Price Elasticity (同#13) | 零售定价ML | 文件名"跨境电商自动定价"与内容不符;同一篇Walmart弹性论文 | 低 |
| 34 | 2021_08a1043cb78b_LumberNowcast | 2023 | [内容错配] Changing Data Sources in ML for Official Statistics (同#12) | 官方统计/ML治理 | 文件名"Google Trends木材期货Nowcast"与内容不符;同一篇官方统计论文 | 低 |
| 35 | 2021_13a826bf673b_AgriPriceStateML | 2021 | Automated Agriculture Commodity Price Prediction System with ML (题名近) | 农产品价格ML | 马来西亚农产品价格,ARIMA/SVR/Prophet/XGB/LSTM五法对比,LSTM(MSE 0.304)最优 | 低 |
| 36 | 2021_1bf8401d78d9_MultiTaskPortMgmt | 2017 | [内容错配] Tukey's Transformational Ladder for Portfolio Management | 组合构造 | 文件名"多任务组合管理"与内容不符;实为S&P500 1958-2015按Tukey阶梯加权,1/x²权重CAGR 18%单调占优 | 高 |
| 37 | 2021_2103.09750_ForexStockSurvey | 2021 | A Survey of Forex and Stock Price Prediction Using Deep Learning | 综述 | DBLP文献分类综述CNN/LSTM/DNN/RNN/RL,LSTM混合模型为主流趋势 | 低 |
| 38 | 2021_2106.07177_TwoStepArbFreeIVS | 2022 | A Two-Step Framework for Arbitrage-Free Prediction of the IVS | 期权定价+DL | 特征提取(PCA/VAE/曲面采样)+LSTM预测+DNN无静态套利重构曲面,胜经典方法 | 中 |
| 39 | 2021_2107.09055_BERT_GAN | 2021 | Stock Price Prediction using BERT and GAN | 情绪+GAN | FinBERT新闻情绪+技术指标喂GAN预测Apple股价,对比LSTM/GRU/GAN/ARIMA | 中 |
| 40 | 2021_2110.09489_SectorVolGARCH | 2020 | Sector Volatility Prediction Performance Using GARCH Models and ANN | 波动率GARCH/ANN | 美国2005-2020五部门:低波动资产用ANN,中高波动用GARCH | 中 |
| 41 | 2021_2111.03477_DataDrivenHedging | 2021 | Data-driven Hedging of Stock Index Options via Deep Learning | 期权对冲+DL | 前馈NN学对冲比率,BS delta+VIX/指数收益情绪特征样本外显著胜BS与HW模型 | 中 |
| 42 | 2021_25a07bf9755f_ExplainableNews | 2024 | [内容错配] FinBERT, GPT-4 and LR Sentiment Stock Prediction (同#5) | 情绪分析 | 文件名"新闻可解释预测"与内容不符;同一篇尼日尼亚股指情绪论文 | 中 |
| 43 | 2021_2f383fb25eb8_TunnelGroundSurface | 2023 | [内容错配] Test of Lepton Flavor Universality B0→D*τν (LHCb) | 粒子物理 | 文件名"隧道地表沉降DL"与内容不符;实为LHCb轻子普适性检验 | 低 |
| 44 | 2021_3b8c13692b34_MovingAvgTrends | 2023 | [内容错配] Changing Data Sources in ML for Official Statistics (同#12) | 官方统计/ML治理 | 文件名"均线股市趋势"与内容不符;同一篇官方统计论文 | 低 |
| 45 | 2021_4e8959955ef7_SurfaceRoughnessMRR | 2017 | [内容错配] Surface Plastic Flow in Polishing of Rough Surfaces | 材料摩擦学 | 文件名"表面粗糙度/材料去除率预测"与内容不符;实为抛光塑性流电镜观察 | 低 |
| 46 | 2021_5328a8543a5e_GoldPrice_DL | 2026 | [内容错配] Monodense Item Price Elasticity (同#13) | 零售定价ML | 文件名"金价DL预测"与内容不符;同一篇Walmart弹性论文 | 低 |
| 47 | 2021_5941fcc3a1eb_EchoStateHybrid | 2026 | [内容错配] Contextual Quantum Neural Networks for Stock Price Prediction | 量子ML | 文件名"回声状态网络混合模型"与内容不符;实为量子多任务网络预测股价分布 | 低 |
| 48 | 2021_92499677e9ed_CrashDiagnosisNYSE | ~2015 | [内容错配] The Physics of the B Factories (书) | 粒子物理 | 文件名"NYSE崩盘诊断与反弹预测"与内容不符;实为B介子工厂物理专著 | 低 |
| 49 | 2021_978e86babff2_InsulatorLeakage | ~2021 | [内容错配] Advanced Drone Swarm Security by Blockchain Governance Game | 无人机安全/博弈 | 文件名"绝缘子泄漏电流"与内容不符;实为区块链治理博弈保护无人机群 | 低 |
| 50 | 2021_9a1a44b17370_GoldCNN_BiGRU | 2016 | [内容错配] A Tutorial about Random Neural Networks (同#3) | NN教程 | 文件名"CNN+Bi-GRU金价"与内容不符;同一篇RNN教程 | 低 |
| 51 | 2021_9c6bfeac43b8_CNCSurfaceRough | 2023 | [内容错配] Deep aPC Neural Network (同#15) | NN架构 | 文件名"CNC端铣表面粗糙度"与内容不符;同一篇aPC网络论文 | 低 |
| 52 | 2021_ad518c23e84c_MLRinNN | 2016 | [内容错配] A Tutorial about Random Neural Networks (同#3) | NN教程 | 文件名"神经网络多元线性回归"与内容不符;同一篇RNN教程 | 低 |
| 53 | 2021_ba62b1c7931a_NNSurfaceChar | 2020 | [内容错配] CO2+CH4/H2 Collision-Induced Absorption & Early Mars Temperature | 行星科学 | 文件名"表面特性NN预测"与内容不符;实为火星早期气候光谱计算 | 低 |
| 54 | 2021_c7e5eb7c1a6e_MultiGrainGBDT | ~2019 | [内容错配] Imbalanced Credit Scoring WHSBoost (同#28) | 信用评分/集成 | 文件名"多粒度多层GBDT"与内容不符;同一篇不平衡信贷论文 | 低 |
| 55 | 2021_d230704e0fdc_CryptoPriceDL | 2018 | [内容错配] Interconnectedness of Cryptocurrencies using Correlation Networks | 相关网络/加密 | 文件名"DL加密货币预测"与内容不符;实为加密货币相关网络:分叉对强关联,Cardano/Ethereum两群组,非纯投机驱动 | 中 |
| 56 | 2021_d3c07d2e9f15_GraphStockRec | 2020 | [内容错配] Déjà vu: Contextualized Temporal Attention for Sequential Recommendation | 推荐系统 | 文件名"图股票推荐"与内容不符;实为电商序列推荐注意力机制 | 低 |
| 57 | 2021_de61b83f3bb3_StatArbML | 2023 | [内容错配] Changing Data Sources in ML for Official Statistics (同#12) | 官方统计/ML治理 | 文件名"ML统计套利"与内容不符;同一篇官方统计论文 | 低 |
| 58 | 2021_e3ebe42b42ff_WristAngleGA-ELM | 2016 | [内容错配] A Tutorial about Random Neural Networks (同#3) | NN教程 | 文件名"手腕角度GA-ELM"与内容不符;同一篇RNN教程 | 低 |
| 59 | 2021_e65e7f6f8721_DLStockPrice | ~2018 | [内容错配] Air Pollutants by Transfer Learning RNN (同#29) | 环境PM预测 | 文件名"DL股价预测"与内容不符;同一篇澳门空气质量论文 | 低 |
| 60 | 2021_ed40039ddf9d_BackorderXAI | 2021 | [内容错配] DOME: Recommendations for ML Validation in Biology | ML验证方法论 | 文件名"物料缺货可解释ML"与内容不符;实为生物ML验证DOME标准(方法论可参考) | 低 |
| 61 | 2021_f68219cb572a_DayAheadElecDL | 2020 | [内容错配] First Observation of Λ0b→ηc pK− (LHCb) | 粒子物理 | 文件名"日前电力DL"与内容不符;实为LHCb粒子衰变观测 | 低 |
| 62 | 2021_fe55ca189e20_PM25_CompNetwork | 2025 | [内容错配] Directional Search for Persistent Gravitational Waves (LIGO) | 引力波 | 文件名"PM2.5复杂网络"与内容不符;实为LIGO-Virgo-KAGRA引力波搜索 | 低 |
| 63 | 2022_17bb5f945485_PearsonCorr | 2016 | [内容错配] Cross-Response in Correlated Financial Markets: Individual Stocks | 市场微观结构 | 文件名"Pearson相关系数性能增强"与内容不符;实为跨股交易冲击价格响应实证:冲击瞬时非永久,交易符号交叉相关短记忆 | 高 |
| 64 | 2022_1a375560d7ac_IntellOptMultiFactor | 2023 | [内容错配] The Modern Mathematics of Deep Learning | DL数学综述 | 文件名"智能优化多因子"与内容不符;实为深度学习数学理论综述 | 低 |
| 65 | 2022_2201.04965_BiTypedMKG | 2022 | Stock Movement Prediction via Bi-typed Hybrid-relational MKG + Dual Attention | 知识图谱+注意力 | 公司+高管双类型实体市场知识图谱建模动量溢出效应,胜9个SOTA基线 | 中 |
| 66 | 2022_2205.10535_DLvsGBM_credit | 2022 | Deep Learning vs. Gradient Boosting: Benchmarking for Credit Scoring | 信贷ML基准 | 三数据集GBM总体胜DL且更快更易用,结构化数据二分类首选GBM | 低 |
| 67 | 2022_2208.08300_TransformerDhaka | 2022 | Transformer-Based DL Model for Stock Price Prediction: Bangladesh | Transformer预测 | time2vec编码Transformer预测达卡交易所8只股票,自称RMSE可接受 | 低 |
| 68 | 2022_2210.12462_DeepMultiFactor | 2022 | Factor Investing with a Deep Multi-Factor Model | 因子模型+GNN | 行业/市场中性化模块+层级股票图注意力估计深度因子,因子注意力显式合成可解释,实盘数据有效 | 高 |
| 69 | 2022_2305.08740_THGNN | 2022 | Temporal and Heterogeneous GNN for Financial Time Series Prediction | GNN时序 | 按日历史价格生成动态关系图+Transformer编码+异质图注意力,中美市场胜SOTA并部署实盘量化系统 | 中 |
| 70 | 2022_430dbe17ab4a_BTCGoldLSTM | 2016 | [内容错配] A Tutorial about Random Neural Networks (同#3) | NN教程 | 文件名"比特币黄金LSTM"与内容不符;同一篇RNN教程 | 低 |
| 71 | 2022_461db670d1ef_PINN_Unmanned | 2018 | [内容错配] Car Detection using UAVs: Faster R-CNN vs YOLOv3 | 计算机视觉 | 文件名"PINN预测"与内容不符;实为无人机车辆检测,YOLOv3胜Faster R-CNN | 低 |
| 72 | 2022_47dfbf932bf5_DistanceXAI | 2026 | [内容错配] Explainable AI for Blind and Low-Vision Users | 人机交互/XAI | 文件名"距离结构可解释AI"与内容不符;实为盲人用户XAI可及性 | 低 |
| 73 | 2022_4b5b91b34290_StructurePropertyCNN | 2015 | [内容错配] Towards Dropout Training for Convolutional Neural Networks | CNN正则化 | 文件名"结构-性质CNN建模"与内容不符;实为max-pooling dropout方法 | 低 |
| 74 | 2022_4dc27942a6e6_SST_EastChinaSea | 2026 | [内容错配] GNN for Colliding Particles, Sea Ice Floe Modeling | GNN/海冰 | 文件名"东海海温预测"与内容不符;实为GNN海冰碰撞模拟 | 低 |
| 75 | 2022_54c62c7602a7_InvestorSentimentML | 2022 | [内容错配] Short-term Effect of COVID-19 on China's Crude Oil Futures (Multifractal) | 多重分形/市场效率 | 文件名"投资者情绪ML预测"与内容不符;实为上海原油期货SC疫情前后市场效率与交叉相关多重分形分析 | 中 |
| 76 | 2022_59ad1ac57e7e_ModifyHiddenLayer | 2015 | [内容错配] Observation of Bs0→μ+μ− decay (同#20) | 粒子物理 | 文件名"改进隐层"与内容不符;同一篇CERN论文 | 低 |
| 77 | 2022_9e9968e26d02_SST_TimeSeries | 2026 | [内容错配] GNN Sea Ice Floe Modeling (同#74) | GNN/海冰 | 文件名"海温时序预测"与内容不符;同一篇海冰GNN论文 | 低 |
| 78 | 2022_9f12d8acebb1_TwoStageMultiFactor | 2026 | [内容错配] Contextual Quantum NN for Stock Price (同#47) | 量子ML | 文件名"两阶段多因子深度整合"与内容不符;同一篇量子ML论文 | 低 |
| 79 | 2022_a9a3ca9243c2_TransformersAttention | 2026 | [内容错配] Contextual Quantum NN for Stock Price (同#47) | 量子ML | 文件名"Transformer注意力股价"与内容不符;同一篇量子ML论文 | 低 |
| 80 | 2022_b89934286829_FreqDecomposition | 2015 | [内容错配] Observation of Bs0→μ+μ− decay (同#20) | 粒子物理 | 文件名"频率分解股价预测"与内容不符;同一篇CERN论文 | 低 |
| 81 | 2022_e7e8fa287db5_BackorderCNN | 2023 | [内容错配] Unveiling Black-Boxes: Explainable DL for Patent Classification (LRP) | NLP可解释 | 文件名"缺货预测CNN可解释"与内容不符;实为LRP专利分类(与#1/#6同LRP方法但非金融域) | 低 |
| 82 | 2022_f1aafdc8a95a_HogFutures | 2010 | [内容错配] A CHAID Based Performance Prediction in Educational Data Mining | 教育数据挖掘 | 文件名"生猪期货组合预测"与内容不符;实为印度学生成绩CHAID预测 | 低 |
| 83 | 2022_fc9d61037fc0_SA-ASBA | 2015 | [内容错配] Observation of Bs0→μ+μ− decay (同#20) | 粒子物理 | 文件名"方面级情绪分析混合模型"与内容不符;同一篇CERN论文 | 低 |
| 84 | 2022_fd9e25faa77e_BayesianCornYield | 2016 | [内容错配] A Tutorial about Random Neural Networks (同#3) | NN教程 | 文件名"贝叶斯域对抗玉米产量"与内容不符;同一篇RNN教程 | 低 |
| 85 | 2023_048243bba1f2_SentimentStockML | 2025 | [内容错配] Machine Learning Approach to Stock Price Crash Risk | 尾部风险/截面 | 文件名"投资者情绪股价预测"与内容不符;实为MCD法度量股价崩盘风险+截面回归:公司层面情绪越高崩盘风险越大(稳健) | 高 |
| 86 | 2023_2304.09376_PhysicalKnowledgeSST | 2023 | Physical Knowledge Enhanced DNN for SST Prediction (题名符) | 物理ML/海洋 | 观测数据GAN迁移物理知识增强数值模型海温预测(非金融,方法论与先验知识嵌入同族) | 低 |
| 87 | 2023_2305.03835_SpatiotemporalTransformer | 2023 | Spatiotemporal Transformer for Stock Movement Prediction | Transformer预测 | STST在ACL18/KDD17达63.7%/56.9%,模拟年化31.24%胜S&P500 | 中 |
| 88 | 2023_2305.16364_E2EAI | 2023 | E2EAI: End-to-End Deep Learning Framework for Active Investing | 因子+组合端到端 | 首个覆盖因子选择→因子组合→选股→组合构造的端到端深度学习主动投资框架 | 高 |
| 89 | 2023_2306.02848_HireVAE | 2023 | HireVAE: Online and Adaptive Factor Model via Hierarchical Regime-Switch VAE | 因子模型+VAE | 层级潜空间嵌入市场状态与个股潜因子,point-in-time在线自适应,四基准主动收益领先 | 高 |
| 90 | 2023_2312.03194_BankruptcyBERT | 2023 | Corporate Bankruptcy Prediction with Domain-Adapted BERT | 情绪/另类数据 | BERT分析MD&A披露情绪预测12个月内破产(91.56%),域适应胜词典法与Word2Vec | 中 |
| 91 | 2023_2401.05337_OptimalLinearSignal | 2023 | Optimal Linear Signal: Unsupervised ML to Optimize PnL with Linear Signals | 信号构造 | 线性回归的无监督变体直接最大化信号PnL的Sharpe,美债ETF二十年回测+正则化 | 高 |
| 92 | 2023_4505abe1ba6e_ChallengeOpportunity | 2023 | [内容错配] Search for Eccentric BH Coalescences O3 (LIGO) | 引力波 | 文件名"DL股价预测挑战与机遇"与内容不符;实为LIGO黑洞并合搜索 | 低 |
| 93 | 2023_588d620f0c48_UrbanForestCarbon | 2023 | [内容错配] Changing Data Sources in ML for Official Statistics (同#12) | 官方统计/ML治理 | 文件名"城市森林碳储量ML"与内容不符;同一篇官方统计论文 | 低 |
| 94 | 2023_693298ab0ef4_CryptoLSTMTrans | 2020 | [内容错配] Nonparametric Pricing and Hedging of Volatility Swaps in SV Models | 期权定价 | 文件名"加密货币LSTM/Transformer"与内容不符;实为zero-vanna隐含波动率近似定价波动率互换+方差互换对冲 | 中 |
| 95 | 2023_6d4e810c5a32_LightGBM_Ensemble | 2022 | [内容错配] Wide Boosting | GBDT扩展 | 文件名"LightGBM混合集成"与内容不符;实为GB输出加宽矩阵乘法(WB),多维输出胜标准GB | 低 |
| 96 | 2023_73f76047adf8_EEMD-Hurst-LSTM | 2022 | [内容错配] Interpolating Commodity Futures Prices with Kriging | 期货期限结构 | 文件名"中国商品期货EEMD-Hurst-LSTM"与内容不符;实为Kriging贝叶斯插值期货期限结构(含季节性/买卖价差) | 中 |
| 97 | 2023_794e10ab459c_EPS_XGBoost | 2022 | [内容错配] Wide Boosting (同#95) | GBDT扩展 | 文件名"特征工程XGB预测EPS"与内容不符;同一篇Wide Boosting论文 | 低 |
| 98 | 2023_7fbc118bde08_Prediction method of surface settlement… | 2025 | [内容错配] A One-Dimensional Energy Balance Model for CO2 Ice on Eccentric Extrasolar Planets (Astrobiology) | 行星气候模型 | F矮星行星需29%更多轨道平均通量才能解冻全球冰盖, 离心率有助于逃逸雪球态 | 低 |
| 99 | 2023_82a003159430_Unsupervised Domain Adaptation for Crime… | 2020 | [内容错配] Disentangling Community-level Changes in Crime Trends During COVID-19 (Chicago) | 贝叶斯结构时序 | 疫情管控使犯罪率社区级分化, 人口是唯一稳定正相关因子 | 低 |
| 100 | 2023_8a038fed282b_Relational Temporal Graph Convolutional… | 2022 | [内容错配] HTGN-BTW: Heterogeneous Temporal Graph Network for Temporal Link Prediction (WSDM Cup 2022) | 时序图网络 | 异质时序图+双时间窗训练策略, WSDM Cup 2022 第2名 (AUC 0.66/0.91) | 低 |
| 101 | 2023_adcdb388a03f_Predicting carbon futures prices based on… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 首次观测Bs0→μ+μ−衰变 (>6σ), 与标准模型预言相容 | 低 |
| 102 | 2023_b00b76f5622e_Stock Market Analysis Using Time Series… | 2024 | [内容错配] Innovative Sentiment Analysis and Prediction of Stock Price Using FinBERT, GPT-4 and Logistic Regression (NGX指数) | 情绪分析/LLM | Logistic回归(81.8% acc)胜过FinBERT与GPT-4预定义法(54.2%), NGX全指情绪分类 | 中 |
| 103 | 2023_bacf064c5fd8_Stock market prediction based on deep… | 2019 | [内容错配] A Hybrid Persian Sentiment Analysis Framework (依存规则+DNN) | 情绪分析/混合NLP | 规则模式触发时依存规则传递情感, 否则切换DNN, 比SVM高10-15% | 低 |
| 104 | 2023_c46fd940bdb1_A novel deep transfer learning framework… | 2019 | [内容错配] Adversarial Domain Adaptation for Stance Detection (MIT) | 对抗域适应 | 对抗域适应迁移立场检测知识, 超Fake News Challenge基线 | 低 |
| 105 | 2023_d5178adbadf0_Climate and environmental data contribute… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 单调性约束深度网络估计商品价格弹性, 无需对照实验设置, 胜DML/LGBM | 低 |
| 106 | 2023_da5b59811635_H3O-LGBM_ hybrid Harris hawk optimization… | - | [内容错配] Real time state monitoring and fault diagnosis system for motor (LabVIEW) | 工业监测 | LabVIEW三相电机多通道监测+阶次分析诊断 | 低 |
| 107 | 2023_dbc248dd5c18_Messages are Never Propagated Alone… | 2016 | [内容错配] A Tutorial about Random Neural Networks in Supervised Learning | 神经网络教程 | RNN(Gelenbe随机NN)=NN+排队论混合体, 监督学习实用指南 | 低 |
| 108 | 2023_dc518eea732d_IntelliSleepScorer_ a software package… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 109 | 2023_ddd7dcaf4c05_A Space-Time Partial Differential Equation… | 2016 | [内容错配] A Tutorial about Random Neural Networks in Supervised Learning | 神经网络教程 | 同#107重复内容 | 低 |
| 110 | 2023_de9d373f30da_Attention based hybrid parametric and… | 2016 | [内容错配] A Tutorial about Random Neural Networks in Supervised Learning | 神经网络教程 | 同#107重复内容 | 低 |
| 111 | 2023_e1ca0e50b348_LSTM_GARCH Hybrid Model for the Prediction… | 2023 | [内容错配] Adjust factor with volatility model using MAXFLAT low-pass filter and construct portfolio in China A share | 因子模型/波动率 | MAXFLAT低通波动率模型调整因子, CSI300/500域优于原始因子, 低Beta连续超额 | 高 |
| 112 | 2023_e8a576eb7ffe_A comparative study on effect of news sentiment… | 2025 | [内容错配] Machine learning approach to stock price crash risk (Annals of OR) | 因子模型/ML | 最小协方差行列式法构造崩盘风险ML度量, 崩盘风险与公司特质投资者情绪正相关 | 高 |
| 113 | 2023_e945c002614e_SM2PNet_ A Deep Learning Approach Towards… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 114 | 2023_f6232c063b95_Source-Free Domain Adaptation via Target… | 2024 | [内容错配] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment | 测试时适应 | 合成域对齐框架(SDA)解决扩散驱动TTA的源-合成域错配, 胜既有扩散TTA | 低 |
| 115 | 2023_fb24ec93b39a_A Comparative Study of Deep Learning… | 2026 | [内容错配] Yield Curve Forecasting using Machine Learning and Econometrics: A Comparative Analysis (47年美债) | 利率预测/ML对比 | ARIMA与朴素计量模型总体胜过ML/DL, TimeGPT/LGBM/RNN为ML最佳 — 债市预测中传统方法仍占优 | 中 |
| 116 | 2023_fed336ba477e_Google Trends and Technical Indicator based… | 2023 | [内容错配] Changing Data Sources in the Age of Machine Learning for Official Statistics | 官方统计/数据治理 | 数据源变更对ML统计的漂移/偏差/有效性风险清单与防护 | 低 |
| 117 | 2024_01eb37129a4b_Factor-GAN_ Enhancing stock price prediction… | 2021 | [内容错配] Quaternion Generative Adversarial Networks | 生成对抗网络 | 四元数GAN处理多通道为单一实体, 参数省75%且FID更优 | 低 |
| 118 | 2024_05d470e66419_Stock Price Prediction Using Deep Learning… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 119 | 2024_076bc1cf02f5_Pre-training enhanced unsupervised contrastive… | 2021 | [内容错配] PoliTO-IIT Submission to EPIC-KITCHENS-100 UDA Challenge for Action Recognition | 域适应/行为识别 | 相对范数对齐(DG)+无监督目标数据扩展, 视频动作识别UDA竞赛方案 | 低 |
| 120 | 2024_16167fba9738_Terrain traversability prediction through… | 2020 | [内容错配] Self-Supervised Learning for Large-Scale Unsupervised Image Clustering (Technion) | 自监督聚类 | 自监督表示+kmeans无监督分类, ImageNet 39-46% | 低 |
| 121 | 2024_1754bae753db_Predicting open interest in thermal coal futures… | 2023 | [内容错配] Changing Data Sources in the Age of ML for Official Statistics | 官方统计/数据治理 | 同#116重复内容 | 低 |
| 122 | 2024_1b19bd3204cc_Stock movement prediction_ A multi_input LSTM… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 123 | 2024_1e2b77ddd7cd_Robust Tool Wear Prediction using Multi-Sensor… | 2026 | [内容相符] Data-driven tool wear prediction in milling (单传感器+迁移学习, arXiv:2412.19950) | 刀具磨损/迁移学习 | 单加速度传感器+STFT+CNN/LSTM/SVM跨工艺迁移, 文件名"多传感器"与内容"单传感器"有别但同领域 | 低 |
| 124 | 2024_1ebcb873877b_Stacked BI-LSTM and E-Optimized CNN-A Hybrid… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 125 | 2024_20cdaac67832_Machine Learning-Driven Early Warning Analytics… | 2020 | [内容错配] iPregel: Strategies for Irregular Vertex-Centric Graph Processing | 图计算系统 | 混合combiner+结构外置+边中心表示优化顶点式图处理 | 低 |
| 126 | 2024_213382d36eb9_Stock Price Prediction in the Financial Market Using… | 2026 | [内容错配] Comparative Evaluation of ML for Minority-Class Financial Distress Prediction (Kent) | 信用风险/类不平衡 | SMOTE+XGBoost/CatBoost/LightGBM对比+SHAP, 强调少数类recall — 破产预测类不平衡工作流可参考 | 中 |
| 127 | 2024_2219890bcf67_Gold price prediction by a CNN-Bi-LSTM model… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 128 | 2024_22baa9c6c952_Digital twin-driven graph domain adaptation neural… | 2022 | [内容错配] Constraints from LIGO O3 data on r-modes in glitching pulsar PSR J0537-6910 | 引力波天文 | LIGO O3对故障脉冲星r模引力波辐射的约束 | 低 |
| 129 | 2024_2401.05430_Multi-Relational Graph Diffusion Neural Network… | 2024 | Multi-Relational Graph Diffusion Neural Network with Parallel Retention for Stock Trends Classification (ICASSP) | GNN/图扩散 | 信息熵+信号能量生成动态多关系股票图, 扩散精炼+并行保留, NASDAQ/NYSE/上交所7年三段测试胜SOTA | 低 |
| 130 | 2024_2402.03659_Learning to Generate Explainable Stock Predictions… | 2024 | Learning to Generate Explainable Stock Predictions using Self-Reflective LLMs (SEP, WWW'24, NUS) | LLM/自反思 | Summarize-Explain-Predict框架+PPO自训练免人工标注, 预测与MCC超DL与LLM基线, 兼验组合构造任务 | 中 |
| 131 | 2024_2402.06633_MDGNN_ Multi-Relational Dynamic Graph Neural Network… | 2024 | MDGNN: Multi-Relational Dynamic Graph Neural Network for Stock Investment Prediction (蚂蚁集团, AAAI) | GNN/动态图 | 离散动态图捕捉股票-行业-投行多重关系, Transformer编码关系演化, 公开数据集胜SOTA | 低 |
| 132 | 2024_2403.02500_RVRAE_ A Dynamic Factor Model Based on Variational… | 2024 | RVRAE: A Dynamic Factor Model Based on Variational Recurrent Autoencoder for Stock Returns Prediction (NCSU) | 动态因子模型 | VRAE+先验-后验学习的概率动态因子模型, 潜空间分布同时估方差与收益, 胜基线 | 高 |
| 133 | 2024_2403.03606_Enhancing Price Prediction in Cryptocurrency Using… | 2024 | Enhancing Price Prediction in Cryptocurrency Using Transformer NN and Technical Indicators | Transformer/加密 | Performer(FAVOR+)+BiLSTM+技术指标预测BTC/ETH/LTC小时与日线 | 低 |
| 134 | 2024_2403.09317_SD-Net_ Symmetric-Aware Keypoint Prediction and… | 2024 | SD-Net: Symmetric-Aware Keypoint Prediction and Domain Adaptation for 6D Pose Estimation | 计算机视觉 | 对称感知关键点+师生自训练域适应, 料箱抓取6D位姿SOTA | 低 |
| 135 | 2024_2403.14063_Diffstock_ Probabilistic Relational Stock Market… | 2024 | DiffStock: Probabilistic Relational Stock Market Predictions using Diffusion Models | 扩散模型/概率预测 | DDPM条件化历史指标+股票间关系做运动预测与组合管理, 配套MaTCHS掩码关系Transformer, SOTA | 中 |
| 136 | 2024_2405.10584_A Hybrid Deep Learning Framework for Stock Price… | 2024 | A Hybrid DL Framework for Stock Price Prediction Considering Investor Sentiment of Online Forum (中南大学) | 情绪分析/混合DL | XLNET论坛情绪+热度因子合成日度群体情绪, 改进BiLSTM-highway预测A股4股 | 中 |
| 137 | 2024_2406.10811_LLMFactor_ Extracting Profitable Factors through Prompts… | 2024 | LLMFactor: Extracting Profitable Factors through Prompts for Explainable Stock Movement Prediction (东大) | LLM/因子提取 | SKGP顺序知识引导提示: 填空造背景知识→从新闻辨因子→文本化历史价格预测, 中美四基准胜SOTA | 中 |
| 138 | 2024_2407.18519_TCGPN_ Temporal-Correlation Graph Pre-trained Network… | 2024 | TCGPN: Temporal-Correlation Graph Pre-trained Network for Stock Forecasting (北大) | GNN/预训练 | 时序-相关性融合编码器+双预训练任务, 节点数无关+多重采样省内存, CSI300/500下游MLP微调SOTA | 低 |
| 139 | 2024_2409.08205_A market resilient data-driven approach to option… | 2026 | A Market Resilient Data-Driven Approach to Option Pricing (IISER Pune/York) | 期权定价 | 无套利理论导出数据驱动集成定价+域适应公共表示空间, 实数据验证 | 中 |
| 140 | 2024_2409.15662_Double-Path Adaptive-correlation Spatial-Temporal… | 2024 | DPA-STIFormer: Double-Path Adaptive-correlation Spatial-Temporal Inverted Transformer (北大) | Transformer/时空 | 以特征变化为token的倒置Transformer+双路自适应相关融合, 四股市数据集SOTA | 低 |
| 141 | 2024_2410.00288_GARCH-Informed Neural Networks for Volatility… | 2024 | GARCH-Informed Neural Networks (GINN) for Volatility Prediction (PSU/CMU) | 波动率/混合模型 | PINN式将GARCH结构注入LSTM, R²/MSE/MAE样本外优于纯GARCH与纯DNN | 中 |
| 142 | 2024_2410.16858_Dynamic graph neural networks for enhanced volatility… | 2024 | Dynamic GNNs (Temporal GAT) for Enhanced Volatility Prediction in Financial Markets (Cardiff) | GNN/波动率 | 相关性与波动率溢出指数构造有向动态图, 8全球指数15年研究胜GARCH与ML基线(短中期) | 中 |
| 143 | 2024_2410.22030_NetAurHPD_ Network Auralization Hyperlink Prediction… | 2024 | NetAurHPD: Network Auralization Hyperlink Prediction for Metabolic Pathways | 生物网络/超链接预测 | 代谢物相关网络的"声化"波形表示+数据增强预测代谢通路 | 低 |
| 144 | 2024_2411.02375_Whack-a-mole Online Learning_ Physics-Informed… | 2024 | Whack-a-mole Online Learning (WamOL): PINN for Intraday Implied Volatility Surface (UCL) | PINN/期权定价 | 自适应损失平衡的PINN从稀疏盘中数据校准时变IVS, 满足PDE与无套利约束 | 中 |
| 145 | 2024_2411.18997_GRU-PFG_ Extract Inter-Stock Correlation from Stock… | 2024 | GRU-PFG: Extract Inter-Stock Correlation from Stock Factors with GNN (武大/UIC) | 因子+GNN | 仅用Alpha360因子投影成图提取股票间相关, CSI300上IC 0.134超HIST 0.131, 泛化潜力更大 | 中 |
| 146 | 2024_2412.07223_A Consolidated Volatility Prediction with Back… | 2024 | A Consolidated Volatility Prediction with BP Neural Network and Genetic Algorithm (ICICML) | 波动率/GA-BP | 遗传算法优化BP网络预测新兴市场波动率, 低误差 | 中 |
| 147 | 2024_2412.10540_Higher Order Transformers_ Enhancing Stock Movement… | 2024 | Higher Order Transformers: Enhancing Stock Movement Prediction on Multimodal Time-Series (Mila) | Transformer/多模态 | 高阶自注意力+张量分解低秩近似+核注意力线性复杂度, 价格+推文多模态 | 低 |
| 148 | 2024_26b2c7d84083_Explainable hybrid quantum neural networks… | 2026 | [内容错配·同域] Contextual Quantum Neural Networks for Stock Price Prediction (UIUC/Fujitsu) | 量子ML | 量子批梯度QBGU+share-and-specify多任务ansatz同电路训练多资产, 对数开销组合表示 | 低 |
| 149 | 2024_290d4ec0a4ae_Stock Price Prediction Using RNNs_ A Comparative… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 150 | 2024_3c6d66dfbf51_An explainable deep learning approach for stock market… | 2009 | [内容错配] World stock market: more sizeable trend reversal likely in Feb/Mar 2010 (Drożdż, 波兰科学院) | 计量物理/对数周期 | 对数周期自相似+普适标度因子λ≈2分析2009全球股市" superbubble", S&P500/恒指/WIG/SSE临界时间2010年2-3月 — 临界现象市场结构分析 | 中 |
| 151 | 2024_3da14bc006f6_An enhanced interval-valued decomposition integration… | 2026 | [内容错配] Contextual Quantum Neural Networks for Stock Price Prediction | 量子ML | 同#148重复内容 | 低 |
| 152 | 2024_48dadb59dc54_A multi-constrained domain adaptation network for… | 2022 | [内容错配] LIFE IV: Ideal kernel-nulling array architectures for space mid-IR nulling interferometer | 天文仪器 | 五望远镜五角构型kernel-nulling胜X-array, 类地行星产率+23% | 低 |
| 153 | 2024_4b83b1aed293_A dual_channel transferable RUL prediction method… | 2018 | [内容错配] BDLOB: Bayesian Deep CNN for Limit Order Books (Oxford, NeurIPS BDL workshop) | 微观结构/贝叶斯DL | dropout变分推断用于LOB价格运动预测, 后验不确定性用于仓位管理与避开劣质交易, LSE数百万观测 | 高 |
| 154 | 2024_55a5afbca640_A Comprehensive Review of Machine Learning for Water… | - | [内容错配] DOME: Recommendations for supervised ML validation in biology | ML方法论 | 数据-优化-模型-评估四要素结构化ML验证建议 | 低 |
| 155 | 2024_57e6c90ef3bf_Developing Hybrid Deep Learning Models for Stock Price… | 2026 | [内容错配] Measurement of the CKM angle γ in B±→D(→KS0h+h−)h± (BESIII+LHCb) | 粒子物理 | γ=(71.3±5.0)°迄今最精确单次测量 | 低 |
| 156 | 2024_6296d96a420a_Realistic material property prediction using domain… | - | [内容错配] DOME: Recommendations for supervised ML validation in biology | ML方法论 | 同#154重复内容 | 低 |
| 157 | 2024_6c60be044958_The role of news sentiment in salmon price prediction… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 158 | 2024_6f0a1f2dc470_Modeling and Prediction of Surface Roughness in Hybrid… | 2023 | [内容错配] Simulation and Measurement of Out-of-Band Resonances for FDM Readout of TES Bolometer | 低温物理仪器 | SPICE仿真+snubber抑制FDM读出带外谐振 | 低 |
| 159 | 2024_7dde14217d6f_Forecasting oil futures markets using machine learning… | 2023 | [内容错配] Changing Data Sources in the Age of ML for Official Statistics | 官方统计/数据治理 | 同#116重复内容 | 低 |
| 160 | 2024_8eaa7b623b7d_A Hybrid GARCH and Deep Learning Method for Volatility… | 2023 | [内容错配] Learn to Accumulate Evidence from All Training Samples (evidential DL) | 不确定性量化 | 证明证据激活函数零证据区是性能下降根因, 提出正则化修复 | 低 |
| 161 | 2024_8f68018eeab0_AI-Driven Intraday Trading_ Applying Machine Learning… | 2025 | [内容错配] Learning Curves for Decision Making in Supervised ML: A Survey | ML方法论/综述 | 学习曲线用于数据获取/早停/模型选择决策的分类框架 | 低 |
| 162 | 2024_91d4832ac7a6_Multi-factor stock price prediction based on GAN-TrellisNet… | 2026 | [内容错配] Contextual Quantum Neural Networks for Stock Price Prediction | 量子ML | 同#148重复内容 | 低 |
| 163 | 2024_934accdb5aee_A Deep Learning-Based LSTM for Stock Price Prediction… | 2021 | [内容错配] Sentiment Analysis in Twitter for Macedonian | NLP/情绪分析 | 马其顿语推文情绪语料+词典, F1 92.16 | 低 |
| 164 | 2024_9bb10482d8cd_An Ensemble Approach to Stock Price Prediction Using Deep… | 2025 | [内容错配] Machine learning approach to stock price crash risk (Annals of OR) | 因子模型/ML | 同#112重复内容 | 高 |
| 165 | 2024_a53c8532b4f2_LLM-Explanation-Enhanced Retail Credit Default… | 2020 | [内容错配] Optimal Dividend Strategy for an Insurance Group with Contagious Default Risk | 保险精算/最优控制 | 传染性违约下多线保险集团最优分红仍是barrier型, HJBVI递归系统求解 | 低 |
| 166 | 2024_a9f63a73c61c_Evaluation and prediction of time overruns in Jordanian… | 2023 | [内容错配] CoralVOS: Dataset and Benchmark for Coral Video Segmentation | 计算机视觉 | 首个密集珊瑚视频分割数据集+6个VOS算法基准 | 低 |
| 167 | 2024_b09fc56a5d9d_Deep Learning Approaches for Stock Price Prediction_ A… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 168 | 2024_b8861bcb3377_A Systematic Review on Graph Neural Network-based… | 2022 | [内容错配] Understanding stock market instability via graph auto-encoders (Oxford, NeurIPS 2022) | 图自编码/市场不稳定 | GAE边重构误差作为资产联动同质性指标, 高误差对应高波动, 改善样本外波动率自回归 — 同#30重复内容 | 高 |
| 169 | 2024_c2090a8c5fb5_Enhanced residual convolutional domain adaptation… | 2019 | [内容错配] Multi-layer Domain Adaptation for Deep CNN (多伦多大学, 医学影像) | 域适应/CV | 梯度反转+SE模块稳定深层域适应训练, 病理/X光多分类+5-20% | 低 |
| 170 | 2024_c435faa5ca66_Deep-learning-based stock market prediction… | 2022 | [内容错配] Sentiment Analysis of ESG disclosures on Stock Market (IIM Indore) | ESG/情绪分析 | 推特+新闻词典法构建ESG情绪指数, 对照4家公司股价变动 | 中 |
| 171 | 2024_c6d3e642c1ee_Stock Index Forecasting Using a Novel Integrated… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 172 | 2024_cc2535f6bb76_A Physics-Guided Attention-Based Neural Network for Sea… | 2016 | [内容错配] A Tutorial about Random Neural Networks in Supervised Learning | 神经网络教程 | 同#107重复内容 | 低 |
| 173 | 2024_ccb67a476373_Multimodal market information fusion for stock price… | 2009 | [内容错配] World stock market: more sizeable trend reversal likely in Feb/Mar 2010 | 计量物理/对数周期 | 同#150重复内容 | 中 |
| 174 | 2024_ce7daeb7ed79_Deep Convolutional Transformer Network for Stock… | 2021 | [内容错配] Convolutional Neural Network in Stock Price Movement Prediction (NIFTY50) | CNN/纯预测 | 1-D CNN预测NIFTY50分钟级涨跌, 精度/召回/F1评估 | 低 |
| 175 | 2024_d04d90ba5a51_Machine Learning based Financial Stock Market Trading… | 2022 | [内容错配] Artificial Intelligence for Trading Strategies (ZHAW, 原油市场) | 交易策略/ML对比 | LSTM/RF/SVM/kNN对比Cross Signal与ARMA-GARCH: RF高波动期recall最高, ARMA-GARCH意外持平, 分市场相位检验稳健性 — 相位化评估可参考 | 中 |
| 176 | 2024_d2342361f601_Gold Price Prediction Model Based on LSTM Neural… | 2018 | [内容错配] Stock Price Correlation Coefficient Prediction with ARIMA-LSTM (高丽大学) | 相关系数预测/组合 | ARIMA滤线性+LSTM预测两股相关系数, 胜历史全样本/常数相关/单指数/多组模型, 服务组合优化 | 中 |
| 177 | 2024_db350520d911_Stock price prediction_ comparison of different moving… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 178 | 2024_dbe81df722f2_Automatic De-Biased Temporal-Relational Modeling for… | 2010 | [内容错配] A Logical Temporal Relational Data Model (IJCSI) | 数据库/时态模型 | 关系模型最小时态属性扩展处理时变数据 | 低 |
| 179 | 2024_de289cbd92df_Soft Sensor Development Based on Unsupervised Dynamic… | 2023 | [内容错配] EPIC-KITCHENS-100 UDA Challenge: Mixed Sequences Prediction (PoliTo) | 域适应/行为识别 | 源目标域混合动作序列+伪标签+语言模型过滤, 榜单verb第2 | 低 |
| 180 | 2024_ebf32c1288e6_Machine learning in financial markets_ A critical review… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 181 | 2024_eecd418a1f0f_Source-Free Domain Adaptation for Privacy-Preserving… | 2014 | [内容错配] How important is the seizure onset zone for seizure dynamics? (波恩大学) | 脑网络/图分析 | 86次局灶癫痫的中心性指标显示SOZ外节点65%情况更重要 | 低 |
| 182 | 2024_f9f34c4d4efd_Explainable Stock Price Movement Prediction using… | 2026 | [内容错配] Contextual Quantum Neural Networks for Stock Price Prediction | 量子ML | 同#148重复内容 | 低 |
| 183 | 2024_fadaf728ce2e_Comparative Analysis of Deep Learning Models for Stock… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 184 | 2025_20470132c3fd_GNN-based social media sentiment analysis for stock… | 2017 | [内容错配] JU_KS@SAIL CodeMixed-2017: Sentiment Analysis for Indian Code Mixed Texts | NLP/情绪分析 | 多项朴素贝叶斯+n-gram+SentiWordnet做印地/孟加拉-英语混码情绪, 竞赛第3 | 低 |
| 185 | 2025_234c5189f224_HRformer_ A Hybrid Relational Transformer for Stock… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 186 | 2025_2502.04737_Learning Universal Multi-level Market Irrationality Factors… | 2025 | UMI: Learning Universal Multi-level Market Irrationality Factors (北航, KDD'25) | 因子模型/自监督 | 股票级协整"理性价格"偏离+市场级同步性自监督(子市场对比+市场同步预测)构造非理性因子, 中美市场通用增强各预测模型 | 高 |
| 187 | 2025_2502.06707_FinMamba_ Market-Aware Graph Enhanced Multi-Level… | 2025 | FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement (清华/同济) | Mamba/GNN | 市场感知剪枝动态图+多层Mamba多时间尺度召回, 线性复杂度, 中美市场SOTA — 含相关性regime分析 | 低 |
| 188 | 2025_2502.10776_A Distillation-based Future-aware Graph Neural Network… | 2025 | DishFT-GNN: Distillation-based Future-aware GNN for Stock Trend Prediction (东北大学) | GNN/知识蒸馏 | 教师模型学习历史-未来分布漂移关联作中间监督, 引导学生模型学未来感知时空嵌入 | 低 |
| 189 | 2025_2502.15813_Stock Price Prediction Using a Hybrid LSTM-GNN Model… | 2025 | Stock Price Prediction Using a Hybrid LSTM-GNN Model (雷丁大学) | LSTM+GNN混合 | LSTM时间模式+GNN Pearson相关关联, 扩展窗验证, MSE比纯LSTM降10.6% | 低 |
| 190 | 2025_2502.15853_Multi-Agent Stock Prediction Systems_ Machine… | 2025 | Multi-Agent Stock Prediction Systems: ML Models, Simulations, and Real-Time Trading | LSTM/注意力对比 | LSTM/GRU/BiLSTM/2-path/Seq2seq/VAE/attention系统对比(TESLA案例), 注意力模型最优 | 低 |
| 191 | 2025_2503.22192_An Advanced Ensemble Deep Learning Framework for Stock… | 2025 | An Advanced Ensemble DL Framework Using VAE, Transformer, and LSTM (SRM) | 集成深度学习 | VAE线性表示+Transformer长期模式+LSTM时序动态三者集成, 市场状态自适应缩放 | 低 |
| 192 | 2025_2505.01432_Dynamic Asset Pricing_ Integrating FinBERT-Based… | 2025 | Dynamic Asset Pricing: Integrating FinBERT-Based Sentiment with Fama-French Five-Factor Model | 因子模型/情绪 | FinBERT构造动态情绪指数及其波动嵌入FF5回归: 常态期情绪正效应, 极端行情放大或反转; 美联储2022-06-15加息75bp事件研究, 滚动回归显示情绪敏感性时变 | 高 |
| 193 | 2025_2505.09551_Fast Learning in Quantitative Finance with Extreme… | 2025 | Fast Learning in Quantitative Finance with Extreme Learning Machine (北大/ING) | ELM/量化方法 | 单层随机隐节点+凸优化输出权重替代DNN迭代训练: 期权定价/日内收益预测/波动率曲面拟合/PDE数值解, 精度相当而速度大幅提升 | 中 |
| 194 | 2025_2505.23084_Gradient Boosting Decision Tree with LSTM for Investment… | 2025 | Gradient Boosting Decision Tree with LSTM for Investment Prediction (东北大学US) | LSTM+GBDT集成 | LSTM+LightGBM/CatBoost异构集成, 7基线对照, 市场regime转换期误差波动降低, 准确率+10-15% | 低 |
| 195 | 2025_2506.01945_Stock Market Telepathy_ GNNs Predicting MINT and G7 | 2025 | Stock Market Telepathy: GNNs Predicting the Secret Conversations between MINT and G7 (Hacettepe) | GNN/跨国市场 | MTGNN分析G7-MINT股指2012-2024: 美加为G7最具影响力, 印尼土耳其为MINT最具影响力, MTGNN胜传统法 — 跨国联动结构证据 | 中 |
| 196 | 2025_2506.06345_Explainable-AI powered stock price prediction using time… | 2025 | Explainable-AI Powered Stock Price Prediction using Time Series Transformers (BIST100) | Transformer/XAI | DLinear胜过Vanilla/TS Transformer与LTSNet(土耳其五大银行股+XBANK/XU100), SHAP+LIME解释, 金融素养导向 | 低 |
| 197 | 2025_2506.14133_Enhancing Forecasting Accuracy in Dynamic Environments… | 2025 | Enhancing Forecasting Accuracy via PELT-Driven Drift Detection and Model Adaptation | 漂移检测/自适应 | PELT检测特征空间漂移点+区间选择性重训MLP/Lasso: 真实数据MAE-44% R²+39%, 合成金融数据亦显著 — regime切换应对框架 | 中 |
| 198 | 2025_2507.14960_A Comparative Analysis of Statistical and Machine Learning… | 2025 | Comparative Analysis of Statistical and ML Models for Outlier Detection in Bitcoin LOBs (L'Aquila) | 微观结构/异常检测 | 统一测试环境AITA-OBS对比13种模型检测加密LOB操纵行为(欺诈/幌骗/分层): 经验协方差EC最佳(6.70%胜买入持有) — LOB异常检测基准 | 高 |
| 199 | 2025_2507.18983_KASPER_ Kolmogorov Arnold Networks for Stock Prediction… | 2025 | KASPER: Kolmogorov Arnold Networks for Stock Prediction and Explainable Regimes | KAN/regime | Kolmogorov-Arnold网络+稀疏样条+regime检测(牛/熊/停滞), 时序对齐防前视偏差, 可解释regime驱动因子 | 中 |
| 200 | 2025_2507.20535_Learning Explainable Stock Predictions with Tweets Using… | 2025 | Learning Explainable Stock Predictions with Tweets Using Mixture of Experts (FTS-Text-MoE) | MoE/文本融合 | 新闻推文按时间抽取关键事实摘要+点嵌入对齐数值序列, MoE Transformer解码器激活子集参数降成本, 多分辨率预测头 — 收益与Sharpe超基线 | 中 |
| 201 | 2025_2508.14656_Deep Learning for Short Term Equity Trend Forecasting… | 2025 | Deep Learning for Short-Term Equity Trend Forecasting: A Behavior-Driven Multi-Factor Approach (UZH/ETH) | 行为因子/多因子 | 40公式化alpha+4行为指标(量价背离/底部反转/追涨), 分行为regime定制模型(分类/回归/混合), MLP在IC与组合收益最优 | 高 |
| 202 | 2025_2509.11928_Meta-Learning Neural Process for Implied Volatility Surfaces.pdf | 2025 | Meta-Learning Neural Process for IVS with SABR-induced Priors (澳门大学, VolNP) | 期权定价/元学习 | 跨交易日元学习+SABR预训练结构先验的注意力神经过程: 稀疏报价单次前向重建IVS, SPX上胜SABR/SSVI/GP, RMSE-40%, 长到期稀疏区收益最大 | 中 |
| 203 | 2025_2510.07661_IKNet_ Interpretable Stock Price Prediction via… | 2025 | IKNet: Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators (汉阳大学) | 可解释/新闻NLP | FinBERT关键词级嵌入+独立非线性投影+技术指标融合, SHAP量化关键词贡献, S&P500 2015-2024 RMSE-32.9%, 累计收益+18.5% | 中 |
| 204 | 2025_2510.10775_Structure Over Signal_ A Globalized Approach to… | 2025 | Structure Over Signal: Globalized Multi-relational GNNs for Stock Prediction (OmniGNN, 哥大) | GNN/多关系 | 扇区节点作全局中介实现冲击快速传播(免多跳扩散), metapath注意力+Transformer时序, COVID期间稳健性突出 | 低 |
| 205 | 2025_2510.21604_RETuning_ Upgrading Inference-Time Scaling for Stock… | 2025 | RETuning: Upgrading Inference-Time Scaling for Stock Movement Prediction with LLMs (港科广+同花顺) | LLM/推理缩放 | 反思式证据调优(冷启动): LLM构建独立分析框架+证据组织评分而非跟随分析师观点; 2024全年5123只A股32K上下文20万+样本数据集, 推理时缩放6个月后与分布外股票仍有效 | 中 |
| 206 | 2025_2510.22348_Causal and Predictive Modeling of Short-Horizon Market Risk .pdf | 2025 | Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha (Oxford) | 风险预测/混合集成 | NN+树集成预测SPY五日回撤(跨股债汇商波五域特征), 可解释归因揭示crash周regime驱动因子: Sharpe 2.51, 年化CAPM alpha+0.28, beta 0.51 (2005-2025) | 高 |
| 207 | 2025_2510.23112_GroupSHAP-Guided Integration of Financial News Keywords… | 2025 | GroupSHAP-Guided Integration of Financial News Keywords and Technical Indicators (汉阳大学) | XAI/新闻NLP | 语义关键词组级SHAP替代token级降算力, 组级SHAP变量作GRU输入, S&P500 2024单日预测MAE-32.2% RMSE-40.5% | 中 |
| 208 | 2025_2511.00085_MaGNet_ A Mamba Dual-Hypergraph Network for Stock… | 2025 | MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction (UCSD) | Mamba/超图 | 双向Mamba+MoE+多头注意力的MAGE块, 时空因果超图+全局概率超图(JS散度加权), 六大股指SOTA | 低 |
| 209 | 2025_2511.01570_Gated Fusion Enhanced Multi-Scale Hierarchical Graph… | 2025 | MS-HGFN: Gated Fusion Multi-Scale Hierarchical GCN for Stock Movement (东北大学) | GNN/多尺度 | 属性内模式动态图+多时间尺度, 自顶向下门控融合粗细粒度特征, 中美市场+1.4%准确率 | 低 |
| 210 | 2025_2511.09854_TermGPT_ Multi-Level Contrastive Fine-Tuning for… | 2026 | TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domains (浙大) | LLM微调/术语 | 句图+句/词两级对比微调解决LLM嵌入各向同性, 首个官方监管文件金融术语数据集, 术语判别超基线 | 低 |
| 211 | 2025_2512.08567_A Hybrid Model for Stock Market Forecasting_ Integrating… | 2025 | A Hybrid Model Integrating News Sentiment and Time Series with GNNs (开罗大学) | GNN+新闻 | LSTM嵌入历史+语言模型嵌入新闻作图节点, GraphSAGE聚合新闻-公司-行业关系: 美股+Bloomberg两数据集胜LSTM基线(+1%/+4%); 标题比全文预测信号更强 | 中 |
| 212 | 2025_2512.16251_Interpretable Deep Learning for Stock Returns_ A… | 2026 | CB-APM: Interpretable Deep Learning for Stock Returns — A Consensus-Bottleneck Asset Pricing Model (POSTECH) | 因子模型/可解释DL | 分析师共识作结构瓶颈(设计即解释), 内生正则同时提升样本外预测与经济可解释性; 组合单调收益梯度跨宏观regime稳健, 共识编码经典因子未张成的定价变异 | 高 |
| 213 | 2025_2601.11528_Knowledge Graph Construction for Stock Markets with LLM… | 2025 | Knowledge Graph Construction for Stock Markets with LLM-Based Explainable Reasoning (Hana TI) | 知识图谱/LLM | 面向股市的知识图谱模式(公司/行业/指标), LLM可解释推理补足时序预测缺失的关系模式与竞争动态 | 中 |
| 214 | 2025_53c57c5cb749_Investigating explainability of deep learning models… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 215 | 2025_54872f88ce1a_A Hybrid LSTM-GRU Model for Stock Price Prediction | 2024 | [内容错配] Comparative Analysis of LSTM, GRU, and Transformer for Stock Price Prediction (Tesla 2015-2024) | 深度学习/股价预测 | LSTM/GRU/Transformer对比, 自称LSTM准确率94%; 题目与文件名不符但同域纯预测 | 低 |
| 216 | 2025_5e935223421e_Ensemble time series models for stock price prediction… | 2022 | [内容错配] Fusion of Sentiment and Asset Price Predictions for Portfolio Optimization (arXiv 2203.05673) | 情绪分析/组合构造 | Semantic Attention情绪模型+LSTM价格预测+均值方差组合; 情绪感知组合收益显著高于无情绪版, 但稳定性更差、风险敞口更大 | 中 |
| 217 | 2025_76d26b2460dc_Multifactor prediction model for stock market analysis… | 2023 | [内容错配] Learn to Accumulate Evidence from All Training Samples (arXiv 2306.11113) | 证据深度学习/不确定性 | 证据激活函数存在零证据区致性能下降, 提出正则项缓解; 非金融通用ML理论 | 低 |
| 218 | 2025_85a59dc21ba1_Advances in land surface forecasting_ a comparison… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 219 | 2025_8856654148ef_Intraday and Post-Market Investor Sentiment for Stock… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 220 | 2025_97c01702f3d4_Stock Market Index Prediction Using CEEMDAN_LSTM_BPNN… | ~2021 | [内容错配] Homogeneous Properties of Automated Market Makers (哥本哈根大学) | 市场微观结构/DeFi | CFMM与TSMM在储备均匀时理论等价; 非线性价格效应转化为滑点与无常损失, 无常损失是波动率与市场深度的函数 | 中 |
| 221 | 2025_a15e61ffad5b_Recurrent Neural Networks in Time-Series Forecasting… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 222 | 2025_a5b89dccfa93_Tackling the Wildfire Prediction Challenge_ An… | 2024 | [内容错配] Study on the Helpfulness of Explainable Artificial Intelligence (XAI) | XAI/人因研究 | 用代理任务度量XAI对人类决策的帮助度; 用户实验显示各SOTA方法在建立信任/怀疑上有差异 | 低 |
| 223 | 2025_bf315d9a34a0_Daily Asset Pricing Based on Deep Learning… | 2026 | [内容错配] Monodense Deep Neural Model for Item Price Elasticity (Walmart) | 深度学习/价格弹性 | 同#105重复内容 | 低 |
| 224 | 2025_bf461dbbdfaa_Deep neural network-based prediction and computational… | 2003 | [内容错配] Message Passing Fluids (Riello Group) | 计算流体/元胞自动机 | 分子作为并行进程的消息传递流体抽象模型; 与金融无关 | 低 |
| 225 | 2025_cf6f63e7326b_Stock Price Prediction with LLM-Guided Market Movement… | 2016 | [内容错配] Understanding the Impacts of Dark Pools on Price Discovery (叶琳琳, CUHK-SZ) | 市场微观结构 | 知情者按信号强度分选: 强信号→交易所, 中等→暗池, 弱→不交易; 暗池对价格发现有放大效应(信息精度高时增强/低时削弱), 调和既有矛盾实证 | 高 |
| 226 | 2025_cfa338cff0e3_Integrating Deep Learning and Econometrics for Stock… | 2015 | [内容错配] Observation of the rare Bs0→μ+μ− decay (CMS+LHCb, CERN) | 粒子物理 | 同#101重复内容 | 低 |
| 227 | 2025_de28fa1d97d2_Time Series Forecasting Framework Using Explainable… | ~2020 | [内容错配] A Time Series Analysis-Based Stock Price Prediction Using ML and DL Models (Mehtab & Sen) | 统计/ML/深度学习预测 | 8分类+8回归模型+LSTM/CNN, NSE五分钟数据聚合成日三时段; 纯预测流派 | 低 |
| 228 | 2025_f75f3df31f06_Stock Market Trend Prediction using an Explainable… | 2021 | [内容错配] Explainable Deep Learning: A Field Guide for the Uninitiated (arXiv 2004.14545) | XAI综述 | 三维度XAI方法空间+解释评估+用户导向解释设计; 通用ML综述非金融 | 低 |
| 229 | 2025_fcf63697bc83_A Contrastive Representation Domain Adaptation Method… | 2022 | [内容错配] Multi-Representation Adaptation Network for Cross-domain Image Classification (arXiv 2201.01002) | 计算机视觉/域适应 | Inception适配模块多表示对齐+MMD扩展; 图像分类与金融无关 | 低 |
| 230 | 2026_2601.04602_Forecasting Equity Correlations with Hybrid Transformer… | 2026 | Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network (昆士兰大学) | GNN/Transformer/统计套利 | THGNN在Fisher-z空间预测10日前瞻股票相关性(对滚动基线残差), Transformer时序编码+边感知图注意力; 2019-2024 OOS显著降低相关性预测误差, 压力期图聚类篮子更自适应 | 高 |
| 231 | 2026_2601.19504_Generating Alpha_ A Hybrid AI-Driven Trading System… | 2026 | Generating Alpha: A Hybrid AI-Driven Trading System (Amrita, ComSIA 2026) | 混合信号/情绪/分域 | EMA/MACD趋势+RSI/BB均值回归+FinBERT情绪+XGBoost信号+波动率分域过滤; 自称24个月135.49%回报跑赢SPX/NDX(回测口径需警惕) | 中 |

## §2 统计(231篇全量)

### 2.1 相关度分布
| 评级 | 数量 | 占比 | 说明 |
|---|---|---|---|
| 高 | 24 | 10.4% | 因子模型/资产定价/微观结构/组合构造/截面定价/可迁移市场结构证据 |
| 中 | 52 | 22.5% | 含可参考构造或评估方法的预测类、情绪另类数据、期权定价、波动率、期货结构 |
| 低 | 155 | 67.1% | 纯预测流派、非金融域、错配非金融内容 |

### 2.2 重大发现:文件名与内容错配率 67.1%
- 231 篇中 **155 篇(67.1%)实际内容与文件名不符**(下载错配),仅 76 篇(32.9%)名实相符。
- 错配内容高度重复,同一批"错误论文"被反复下载成不同文件名:
  | 错配内容簇 | 份数 | 实际内容 |
  |---|---|---|
  | CERN Bs0→μ+μ− 衰变观测 (arXiv 1411.4413, CMS+LHCb) | ~16 | 粒子物理 |
  | Walmart Monodense 商品价格弹性 (arXiv 2603.29261) | 13 | 零售定价ML |
  | Random NN/RNN 监督学习教程 (arXiv 1609.04846, Gelenbe) | ~8 | NN教程 |
  | Changing Data Sources 官方统计风险清单 (arXiv 2306.04338) | ~9 | 官方统计/ML治理 |
  | Contextual Quantum NN + 量子多任务股价 (arXiv 2503.01884等) | 7 | 量子ML |
  | ML 股价崩盘风险 (MCD法, Annals of OR) | 3 | 风险计量 |
  | DOME 生物ML验证标准 | 3 | ML方法论 |
  | GAE 市场不稳定 (Oxford NeurIPS'22) | 2 | 图自编码 |
  | Drożdż 对数周期泡沫 | 2 | 计量物理 |
  | EPIC-KITCHENS UDA | 2 | 计算机视觉 |
- 扣除约 60 份同内容重复文件,231 个 PDF 实质对应约 **170 篇不同论文**;其中真正金融量化相关(高+中)去重后约 60-65 篇。
- 推论:任何按文件名组织的学习层(目录、旧分析、引用)在 2/3 的条目上引用了不存在的论文,结论层必然被污染——见 §4 对账验证。

### 2.3 方法族分布(按第5列方法族聚合,金融相关为主)
| 方法族 | 篇数(约) | 备注 |
|---|---|---|
| 深度因子模型/资产定价 | 10 | Deep Factor/LRP、HireVAE、RVRAE、CB-APM、E2EAI、MAXFLAT因子调整、FinBERT×FF5、行为40alpha、非理性因子UMI |
| GNN/图结构(股票关系建模) | 12 | HIST、MKG、双类型异质图、FinMamba、DishFT-GNN、OmniGNN、MS-HGFN、THGNN等 |
| Transformer/大模型预测 | 10 | Higher Order Transformer、FTS-Text-MoE、SKGP、LLMFactor、RETuning等 |
| 情绪/NLP/另类数据 | 14 | FinBERT、XLNET、BERT破产预测、ESG指数、知识图谱等 |
| 波动率/期权/IVS | 10 | PINN IVS(WamOL/VolNP)、GINN、曲面无套利重构、GA-BP、分域GARCH |
| 市场微观结构 | 4 | 暗池价格发现、BDLOB、跨股冲击响应、AMM |
| 组合构造/配置 | 5 | Tukey阶梯、无监督PnL线性信号、情绪组合、相关性预测篮子 |
| 风险/崩盘/回撤 | 5 | ML崩盘风险×3、Oxford五日回撤 |
| 纯时序预测流派(LSTM/GRU/集成) | ~20 | 各类股价/指数/商品预测 |
| 非金融域(粒子物理24+天文/流体/海温/山火/医学/CV等) | ~63 | 错配内容为主体 |
| 其他错配(零售弹性、量子ML、NN教程、官方统计等) | ~55 | 见2.2簇表 |

## §3 高相关清单(全部24篇"高",按主题分组)

**因子模型/资产定价(8)**
- #1 Deep Factor Model(LRP解释DL收益率预测决策,日本市场)
- #6 Deep Recurrent Factor Model(可解释非线性时变多因子)
- #68 Factor Investing with a Deep Multi-Factor Model
- #88 E2EAI(因子选择→组合→选股→组合构造端到端主动投资框架)
- #89 HireVAE(层级regime-switch VAE在线自适应因子模型,point-in-time)
- #132 RVRAE(变分循环自编码动态因子模型,同时估方差与收益)
- #212 CB-APM(POSTECH共识瓶颈资产定价,内生正则+可解释,跨宏观regime稳健)
- #111 MAXFLAT低通波动率模型调整A股因子,CSI300/500低Beta连续超额 [错配文件名]

**市场微观结构(3)**
- #225 暗池与价格发现(信号强度分选+放大效应,调和矛盾实证) [错配文件名]
- #153 BDLOB(贝叶斯深度CNN限价订单簿,Oxford) [错配文件名]
- #63 跨股交易冲击价格响应(冲击瞬时非永久,符号交叉相关短记忆) [错配文件名]

**市场结构/不稳定/异常检测(3)**
- #30 / #168 GAE边重构误差作联动崩塌代理,改善样本外波动率预测(Oxford NeurIPS'22,两份重复) [错配文件名]
- #198 比特币LOB异常检测13模型统一基准(经验协方差EC最佳)

**风险/崩盘/回撤(4)**
- #85 / #112 / #164 ML股价崩盘风险(MCD度量,情绪越高崩盘风险越大,Annals of OR,三份重复) [错配文件名]
- #206 Oxford五日回撤混合集成(跨五域特征,Sharpe 2.51,alpha+0.28,beta 0.51,可解释归因)

**行为/情绪因子(3)**
- #186 UMI非理性因子(KDD'25,协整理性偏离+市场同步自监督,中美通用)
- #192 FinBERT情绪嵌入FF5(常态正效应,极端行情反转,情绪敏感性时变)
- #201 行为驱动40alpha+4行为指标(UZH/ETH,分行为regime,MLP最优)

**组合构造/信号(3)**
- #36 Tukey变换阶梯组合管理(1/x²权重CAGR 18%单调占优,1958-2015) [错配文件名]
- #91 无监督ML直接最大化线性信号PnL的Sharpe(美债ETF二十年)
- #230 THGNN前瞻相关性预测→图聚类篮子(StatArb,压力期更自适应,2019-2024 OOS)

## §4 与旧分析对账(all_deep_analysis.md, 13082行, 2026-06-05)

### 4.1 对账方法
独立索引(§1-§3)完成后才读旧分析。只核对两件事: 覆盖面与结论主张。旧分析自称"15个专题·12975行·651篇论文"的数学-哲学分析全书。

### 4.2 覆盖率
- **旧分析语料 ≠ 当前PDF库**。旧分析引用的"论文28"(Investigating explainability…seq data)等论文在当前库中的对应文件实际内容是Walmart价格弹性; 专题一42篇中至少14篇(水质综述/电力窃取/煤矿突水/苹果Brix/智能铸造/可再生能源/咖啡期货/天然气期货/印尼银行SVM/KOSDAQ/SSA-CNN-LSTM/Dash/面板数据/Tesla SVM比较)在当前231个PDF中**根本没有对应文件**; 另至少5篇(技术债务/城市森林碳/GoogleTrends/木材Nowcast/论文28)文件存在但实际内容完全不同。
- **高相关论文覆盖约一半**: 22篇去重"高"中, 旧分析实质覆盖约10-11篇(HireVAE 26处/RVRAE 18处/CB-APM 14处/UMI 23处/E2EAI 10处/FinBERT+FF5/DRFM/Optimal Linear Signal/Bitcoin LOB异常检测), **未覆盖约11篇**: 暗池价格发现(#225)、BDLOB(#153)、Tukey阶梯(#36)、MAXFLAT(#111)、ML崩盘风险(#85)、GAE市场不稳定(#30)、THGNN前瞻相关性(#230)、Oxford五日回撤(#206)、行为驱动40alpha(#201)、跨股冲击响应(#63)、深度多因子(#68存疑)。微观结构与风险类高相关论文系统性缺失。
- **"651篇"宣称无法核实**: 全文逐篇表格行仅72行, 内联作者-年份/编号引用约110-150处, 实际逐篇处理的论文约150-200篇, 远低于宣称的651篇。

### 4.3 抽样比对(10篇高相关)
| # | 论文 | 旧分析 | 一致性判定 |
|---|---|---|---|
| 89 | HireVAE 2023 | 26处, "层次化变分因子模型, 在线适应=贝叶斯滤波" | **一致**(层级regime-switch+在线自适应); Kalman类比是旧分析自身诠释 |
| 132 | RVRAE 2024 | 18处, 递归变分+动态因子+变分下界 | **一致** |
| 212 | CB-APM 2026 | 14处, 引arXiv 2512.16251, 共识瓶颈, 可解释性与精度互相促进 | **一致**; "年度频率"细节前2页无法核验 |
| 186 | UMI KDD'25 | 23处, 协整理性偏离→非理性因子 | **一致** |
| 88 | E2EAI 2023 | 10处, 信号生成与组合构建不分开优化, A股2800+股, Sharpe 2.26 | **方向一致**; 具体数字无法从前2页核验 |
| 192 | FinBERT+FF5 2025 | "论文10", 情绪因子入FF5定价 | **一致** |
| 91 | Optimal Linear Signal 2023 | 核心论文清单, 无监督直接最大化PnL的Sharpe | **一致** |
| 6 | DRFM 2019 | LSTM+LRP非线性时变因子 | **一致** |
| 198 | Bitcoin LOB异常检测 2025 | Letteri 2025, 传统统计法在加密LOB假阳性高 | **一致** |
| 225 | 暗池与价格发现 2016 | 0处 | **缺失** |
(补充: #153 BDLOB/#36 Tukey/#111 MAXFLAT/#85 崩盘风险/#30 GAE/#230 THGNN-2026 同样0处缺失; 旧分析中的"THGNN(2022)"指另一篇时序异构图论文, 与#230的2026相关性预测论文无关)

**抽样一致率: 覆盖到的9/9方向一致, 0篇结论相反** —— 旧分析对它真正读过的论文描述是可靠的; 问题出在覆盖面和没读过的部分。

### 4.4 错配污染实锤(旧分析按文件名而非内容写作)
旧分析含17处"引用N次"标注(如"引用150次")——**PDF正文从不包含自身被引次数, 这只可能来自元数据清单**, 证明旧分析相当部分是按"文件名+元数据+模型先验知识"写的, 不是读PDF写的。实锤案例:
- "论文28"(=文件#214 "Investigating explainability…"): 旧分析称其为SHAP可解释性股价预测论文(10次引用); **该文件实际内容是Walmart Monodense商品价格弹性**。
- "Siavvas 2020 技术债务与安全风险"(=文件#20): 旧分析当真论文讨论; **实际内容是CERN Bs0→μ+μ−粒子物理**。
- "Li 2023 城市森林碳预测"(=文件#93): 旧分析称遥感+ML; **实际内容是官方统计数据源变更风险清单**。
- "Hani'ah 2023 Google Trends"(=文件#116等): 旧分析称Granger因果; **实际内容同为官方统计论文**。
结论: 旧分析中所有与这155个错配文件对应的"论文描述", 描述的都是不存在的论文——**该部分学习层整体不可信**。个别例外(如#148量子论文按哈希ID引用了实际内容)说明写作过程是"部分读了内容+部分按文件名脑补"的混合体。

### 4.5 夸大与无据断言(结论强于论文本身)
1. **无出处的"实证"精确数字**: "实证研究表明, A股散户对负面新闻的反应时滞约为1-3个交易日, 而正面新闻约0.5-1个交易日"(约106行); "实证研究发现, A股社交媒体情绪的SLT约为1-5个交易日, 美股0-1个交易日"(约278行)——语料中没有任何论文支持这些具体数字, 属无据精确。
2. **语料级比较断言无单篇支撑**: "深度因子模型在A股上的相对改进比在美股上更大"(约12483行)——各深度因子论文均为单一市场(DRFM日本/RVRAE美国/HireVAE中国), 无一篇做过该比较。
3. **"651篇论文"框架性夸大**: 实际逐篇处理约150-200篇。
4. **正面发现**: 旧分析的批判框架有真价值——"预测世界vs交易世界"认识论断裂、"几乎所有ML方法报告的 impressive 指标存在根本性数学谬误"(8989行, 对E2EAI Sharpe 2.26等持怀疑)、半Kelly回撤数学等; 且混入了用户内部策略实证(V116/V118/V129/V133/V134, 如"V133 ATR=1.5 R=7 +328.9%/73.0%WR")——它实际是"文献×内部研究"综合文档, 不宜再被当作纯文献事实源引用。

### 4.6 对账结论
1. 旧分析**不能作为当前论文库的索引或事实源**: 67.1%的PDF内容与文件名不符, 旧分析对这些文件的全部描述指向不存在的论文。
2. 对真实覆盖到的近年论文(约10篇高相关+若干中相关), 旧分析描述与论文原文**方向一致、无发现相反结论**, 这部分仍可参考。
3. 高相关论文覆盖缺口约50%, 微观结构(暗池/BDLOB/冲击响应)与风险(崩盘/回撤)两大类系统性缺失。
4. 夸大形态: 无据精确数字 > 语料级断言 > 框架性数字(651篇); 无发现"单篇论文结论被改写加强"式夸大(覆盖到的部分)。
5. 建议: 论文级事实以本索引为准; 旧分析降级为"思想框架+内部综合"文档; 155个错配PDF需按实际内容重命名或重新下载正确论文后重建学习层。
