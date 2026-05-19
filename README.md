# 机器学习任务引擎合集

基于Python的机器学习算法实践项目，包含5个完整的机器学习任务引擎。

## 📋 项目简介

本项目包含5个机器学习核心任务，涵盖关联规则、聚类、分类等主流算法。

## 📂 项目结构

```
ML/
├── 任务引擎一_关联规则挖掘_UCI_Online_Retail/
│   ├── apriori_analysis.py      # Apriori关联规则算法
│   └── online_retail_II.csv     # 数据集
│
├── 任务引擎二_K-means聚类_Kaggle_Mall_Customer_Segmentation/
│   ├── kmeans_analysis.py       # K-means聚类算法
│   └── Mall_Customers.csv       # 数据集
│
├── 任务引擎三_KNN分类_Kaggle_Telco_Customer_Churn/
│   ├── knn_analysis.py          # KNN分类算法
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 任务引擎四_决策树分类_Kaggle_Heart_Disease_UCI/
│   ├── decision_tree_analysis.py # 决策树分类算法
│   └── heart_disease_uci.csv
│
├── 任务引擎五_逻辑回归_Kaggle_Stroke_Prediction/
│   ├── logistic_regression_analysis.py  # 逻辑回归算法
│   └── healthcare-dataset-stroke-data.csv
│
├── .gitignore
└── README.md
```

## 🧠 任务概览

| 任务 | 算法 | 数据集 | 核心主题 |
|------|------|--------|----------|
| 任务一 | Apriori | Online Retail II | 商品关联规则挖掘 |
| 任务二 | K-means | Mall Customers | 客户细分聚类 |
| 任务三 | KNN | Telco Customer Churn | 客户流失预测 |
| 任务四 | 决策树 | Heart Disease UCI | 心脏病风险预测 |
| 任务五 | 逻辑回归 | Stroke Prediction | 中风风险预测 |

## 🚀 运行方式

每个任务文件夹下包含完整的Python代码，使用相对路径读取数据文件。

**前置要求**：
- Python 3.8+
- pandas
- numpy

**运行任意任务**：
```bash
cd 任务引擎X_xxx
python xxx_analysis.py
```

## 📚 核心知识点

### 关联规则
- 支持度 (Support): 项集出现的频率
- 置信度 (Confidence): 购买A后购买B的概率
- 提升度 (Lift): 规则的实际价值

### K-means聚类
- 肘部法则确定最优K值
- 轮廓系数评估聚类效果
- 数据标准化

### KNN分类
- 欧氏距离度量
- K值选择
- 数据标准化

### 决策树
- 基尼指数
- 信息增益
- 决策树剪枝

### 逻辑回归
- Sigmoid函数
- 梯度下降优化
- ROC曲线与AUC

## 📊 数据来源

- [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) - UCI Machine Learning Repository
- [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- [Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## 📝 课程信息

- **课程**: 机器学习算法入门与编程实现（基于Python）
- **数据来源**: Kaggle / UCI Machine Learning Repository

## 📄 License

MIT License
