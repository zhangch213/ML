# 机器学习课程 — Kaggle数据集任务引擎合集

> **课程**: 机器学习算法入门与编程实现（基于Python）  
> **数据来源**: Kaggle / UCI Machine Learning Repository  
> **制作日期**: 2026-05-12

---

## 任务引擎一：关联规则挖掘 — UCI Online Retail 购物篮分析

### 【任务名称】
基于Apriori算法的在线零售商品关联规则挖掘

### 【数据来源】
- **原始数据集**: UCI Machine Learning Repository — Online Retail Dataset
- **Kaggle链接**: https://www.kaggle.com/datasets/arnavs19/wine-quality-uci-machine-learning-repository
- **数据规模**: 原数据集541,909条交易记录，来自英国在线零售商（2010-2011年）
- **模拟规模**: 2,000条交易记录（保持原始数据结构）

### 【对应章节】
第3章 3.1 关联规则挖掘（Apriori算法）

### 【任务描述】
某英国在线零售商希望通过分析顾客的购物篮数据，发现商品之间的购买关联规则，从而优化商品推荐策略和捆绑销售方案。使用Apriori算法对交易数据进行分析，找出频繁项集和强关联规则。

### 【数据集字段】
| 字段 | 说明 |
|------|------|
| InvoiceNo | 发票编号 |
| Description | 商品描述（多个商品用\|分隔） |
| Quantity | 购买数量 |
| UnitPrice | 单价 |
| CustomerID | 客户编号 |
| Country | 国家 |

### 【核心知识点】
- **支持度（Support）**: P(A∪B)，A和B同时出现的概率
- **置信度（Confidence）**: P(B|A) = P(A∪B)/P(A)
- **提升度（Lift）**: Confidence/P(B)，衡量规则的实际价值
- **Apriori原理**: 频繁项集的任何子集也一定是频繁的

### 【实践代码】

```python
import pandas as pd
from itertools import combinations
from collections import Counter

# 加载数据
df = pd.read_csv('kaggle_task1_online_retail.csv')
transactions = [desc.split('|') for desc in df['Description']]

# Apriori算法实现
def apriori(transactions, min_support=0.05, min_confidence=0.5):
    all_items = sorted(list(set([item for t in transactions for item in t])))
    n = len(transactions)

    # 频繁1-项集
    item_counts = Counter()
    for t in transactions:
        for item in t:
            item_counts[item] += 1
    freq1 = {frozenset([item]): count/n for item, count in item_counts.items() if count/n >= min_support}

    # 频繁2-项集
    pairs = list(combinations(all_items, 2))
    pair_counts = Counter()
    for t in transactions:
        t_set = set(t)
        for pair in pairs:
            if set(pair).issubset(t_set):
                pair_counts[frozenset(pair)] += 1
    freq2 = {pair: count/n for pair, count in pair_counts.items() if count/n >= min_support}

    # 生成关联规则
    rules = []
    for itemset, support in freq2.items():
        items = list(itemset)
        for i in range(2):
            ant, cons = items[i], items[1-i]
            conf = support / freq1.get(frozenset([ant]), 1)
            if conf >= min_confidence:
                lift = conf / freq1.get(frozenset([cons]), 1)
                rules.append((ant, cons, support, conf, lift))
    return freq1, freq2, sorted(rules, key=lambda x: x[3], reverse=True)

# 运行算法
freq1, freq2, rules = apriori(transactions, min_support=0.05, min_confidence=0.5)

# 输出结果
print("频繁1-项集:")
for itemset, support in sorted(freq1.items(), key=lambda x: x[1], reverse=True):
    print(f"  {list(itemset)[0]}: 支持度={support:.3f}")

print("\n强关联规则:")
for ant, cons, sup, conf, lift in rules[:10]:
    print(f"  {ant} → {cons}: 支持度={sup:.3f}, 置信度={conf:.3f}, 提升度={lift:.3f}")
```

### 【思考问题】
1. 为什么"HEART + CAKESTAND → HEART_DECO"会有较高的提升度？这反映了什么购物行为？
2. 如果提升度小于1，说明什么？这样的规则是否有商业价值？
3. 在实际零售场景中，如何利用发现的关联规则进行商品摆放优化？

### 【拓展任务】
尝试调整最小支持度和最小置信度，观察规则数量的变化。当支持度从5%提高到10%时，规则数量如何变化？

---

## 任务引擎二：K-means聚类 — Kaggle Mall Customer Segmentation

### 【任务名称】
基于K-means算法的商场客户细分分析

### 【数据来源】
- **原始数据集**: Kaggle — Mall Customer Segmentation Data
- **Kaggle链接**: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- **数据规模**: 200条真实客户记录
- **模拟规模**: 500条客户记录（保持原始特征结构）

### 【对应章节】
第4章 4.3 K-means和K中心点聚类

### 【任务描述】
某商场希望通过分析客户的年龄、年收入和消费评分，将客户划分为不同的价值群体，以便实施精准营销策略。使用K-means算法进行客户细分，并通过肘部法则确定最佳聚类数。

### 【数据集字段】
| 字段 | 说明 |
|------|------|
| CustomerID | 客户编号 |
| Gender | 性别 |
| Age | 年龄 |
| Annual Income (k$) | 年收入（千美元） |
| Spending Score (1-100) | 消费评分（1-100） |

### 【核心知识点】
- **K-means算法**: 指定K个中心点，迭代分配样本到最近中心
- **簇内平方和（SSE）**: 评估聚类效果的指标
- **肘部法则**: 通过SSE曲线拐点确定最佳K值
- **数据标准化**: 消除不同量纲特征的影响

### 【实践代码】

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('kaggle_task2_mall_customers.csv')

# 特征选择
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 肘部法则确定最佳K值
inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 绘制肘部法则图
plt.plot(range(2, 10), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.grid(True)
plt.show()

# 选择K=4进行聚类
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# 分析各簇特征
print("各簇客户数量:")
print(df['Cluster'].value_counts().sort_index())

print("\n各簇特征均值:")
print(df.groupby('Cluster')[features].mean().round(2))

# 可视化
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                label=f'Cluster {i+1}', alpha=0.6)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation (K-means)')
plt.legend()
plt.show()
```

### 【思考问题】
1. 为什么K-means算法对初始中心点敏感？如何缓解？
2. 数据标准化对聚类结果有什么影响？如果不标准化会怎样？
3. 根据聚类结果，商场应该如何对不同客户群体制定营销策略？

### 【拓展任务】
尝试使用K-medoids算法对同一数据进行聚类，比较两种算法的结果差异。

---

## 任务引擎三：KNN分类 — Kaggle Telco Customer Churn

### 【任务名称】
基于KNN算法的电信客户流失预测

### 【数据来源】
- **原始数据集**: Kaggle — Telco Customer Churn
- **Kaggle链接**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **数据规模**: 原数据集7,043条客户记录
- **模拟规模**: 500条客户记录（保持原始21个特征）

### 【对应章节】
第5章 5.1 分类和KNN算法

### 【任务描述】
某电信公司希望根据客户的合同信息、服务使用情况和个人信息，预测客户是否会流失（取消服务）。使用KNN算法构建分类模型，帮助公司提前识别高风险客户并采取挽留措施。

### 【数据集字段】
| 字段 | 说明 |
|------|------|
| customerID | 客户编号 |
| gender | 性别 |
| SeniorCitizen | 是否为老年人 |
| Partner | 是否有伴侣 |
| Dependents | 是否有家属 |
| tenure | 在网时长（月） |
| PhoneService | 是否使用电话服务 |
| MultipleLines | 是否多线路 |
| InternetService | 互联网服务类型 |
| OnlineSecurity | 在线安全服务 |
| OnlineBackup | 在线备份服务 |
| DeviceProtection | 设备保护 |
| TechSupport | 技术支持 |
| StreamingTV | 流媒体电视 |
| StreamingMovies | 流媒体电影 |
| Contract | 合同类型 |
| PaperlessBilling | 是否无纸账单 |
| PaymentMethod | 支付方式 |
| MonthlyCharges | 月费 |
| TotalCharges | 总费用 |
| Churn | 是否流失（目标变量） |

### 【核心知识点】
- **KNN算法**: 通过计算距离选择K个最近邻进行投票
- **距离度量**: 欧氏距离、曼哈顿距离
- **K值选择**: 通过交叉验证选择最佳K值
- **数据标准化**: 消除量纲影响
- **类别不平衡**: 流失客户通常远少于留存客户

### 【实践代码】

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('kaggle_task3_telco_churn.csv')

# 编码分类变量
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'Churn']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 选择特征
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
            'InternetService', 'OnlineSecurity', 'TechSupport', 'Churn']
X = df[features[:-1]].values
y = df['Churn'].values

# 处理缺失值
import numpy as np
X = np.nan_to_num(X, nan=0)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 选择最佳K值
k_values = [3, 5, 7, 9, 11, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"K={k}: Accuracy={acc:.4f}")

# 使用最佳K值训练
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
```

### 【思考问题】
1. 为什么电信客户流失预测中类别不平衡问题特别严重？如何处理？
2. KNN算法的时间复杂度是多少？对于大规模数据集有什么优化方法？
3. 除了KNN，还有哪些算法适合客户流失预测？

### 【拓展任务】
尝试使用SMOTE过采样技术处理类别不平衡问题，观察对模型性能的影响。

---

## 任务引擎四：决策树分类 — Kaggle Heart Disease UCI

### 【任务名称】
基于决策树的心脏病风险预测

### 【数据来源】
- **原始数据集**: UCI Machine Learning Repository — Heart Disease Dataset
- **Kaggle链接**: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- **数据规模**: 原数据集303条患者记录
- **模拟规模**: 500条患者记录（保持原始13个特征）

### 【对应章节】
第5章 5.4 决策树分类算法

### 【任务描述】
某医院希望根据患者的临床指标（年龄、血压、胆固醇等），预测其是否患有心脏病。使用决策树算法构建分类模型，并提取可解释的决策规则，辅助医生进行诊断。

### 【数据集字段】
| 字段 | 说明 |
|------|------|
| age | 年龄 |
| sex | 性别（1=男，0=女） |
| cp | 胸痛类型（0-3） |
| trestbps | 静息血压（mmHg） |
| chol | 血清胆固醇（mg/dl） |
| fbs | 空腹血糖>120mg/dl（1=是，0=否） |
| restecg | 静息心电图结果（0-2） |
| thalach | 最大心率 |
| exang | 运动诱发心绞痛（1=是，0=否） |
| oldpeak | ST段压低 |
| slope | 峰值运动ST段斜率（0-2） |
| target | 是否患心脏病（1=是，0=否） |

### 【核心知识点】
- **信息增益**: 选择信息增益最大的特征划分数据（ID3）
- **基尼指数**: 衡量数据不纯度（CART）
- **决策树剪枝**: 防止过拟合
- **特征重要性**: 评估各特征对分类的贡献

### 【实践代码】

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('kaggle_task4_heart_disease.csv')

X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 训练决策树（限制深度防止过拟合）
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, 
                            min_samples_split=10, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
      target_names=['No Disease', 'Disease']))

# 特征重要性
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature Importance:")
for i in indices[:5]:
    print(f"  {df.columns[i]}: {importances[i]:.4f}")

# 可视化决策树
plt.figure(figsize=(20, 12))
plot_tree(dt, feature_names=df.columns[:-1], 
          class_names=['No Disease', 'Disease'],
          filled=True, rounded=True, fontsize=10)
plt.title('Heart Disease Decision Tree')
plt.show()
```

### 【思考问题】
1. 在医疗诊断场景中，决策树相比黑盒模型（如神经网络）有什么优势？
2. 为什么需要对决策树进行剪枝？max_depth参数如何影响模型性能？
3. 如果某个特征（如性别）的基尼指数为0，说明什么？

### 【拓展任务】
尝试使用随机森林算法对同一数据进行分类，比较单棵决策树和随机森林的准确率差异。

---

## 任务引擎五：逻辑回归 — Kaggle Stroke Prediction

### 【任务名称】
基于逻辑回归的中风风险预测模型

### 【数据来源】
- **原始数据集**: Kaggle — Stroke Prediction Dataset
- **Kaggle链接**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **数据规模**: 原数据集5,110条患者记录
- **模拟规模**: 500条患者记录（保持原始11个特征）

### 【对应章节】
第6章 回归与逻辑回归

### 【任务描述】
某医疗机构希望根据患者的健康指标（年龄、BMI、血糖水平、吸烟状况等），预测其发生中风的风险概率。使用逻辑回归构建二分类模型，并解释各特征对中风风险的影响。

### 【数据集字段】
| 字段 | 说明 |
|------|------|
| id | 患者编号 |
| gender | 性别 |
| age | 年龄 |
| hypertension | 是否有高血压（0/1） |
| heart_disease | 是否有心脏病（0/1） |
| ever_married | 是否已婚 |
| work_type | 工作类型 |
| Residence_type | 居住类型 |
| avg_glucose_level | 平均血糖水平 |
| bmi | 体重指数 |
| smoking_status | 吸烟状况 |
| stroke | 是否中风（目标变量） |

### 【核心知识点】
- **Sigmoid函数**: g(z) = 1/(1+e^(-z))，将线性输出映射到[0,1]
- **对数几率（Logit）**: ln(P/(1-P)) = w^T x + b
- **极大似然估计**: 逻辑回归的参数估计方法
- **ROC曲线和AUC**: 评估二分类模型性能
- **优势比（OR）**: exp(系数)，表示特征对风险的影响倍数

### 【实践代码】

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report

df = pd.read_csv('kaggle_task5_stroke_prediction.csv')

# 编码分类变量
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 处理BMI缺失值
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

X = df.drop(['id', 'stroke'], axis=1).values
y = df['stroke'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Stroke Prediction)')
plt.legend()
plt.show()

print(f"AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))

# 特征系数解释
coef = lr.coef_[0]
feature_names = df.drop(['id', 'stroke'], axis=1).columns
print("\nFeature Impact (Odds Ratio):")
for i, name in enumerate(feature_names):
    or_value = np.exp(coef[i])
    direction = "increases" if coef[i] > 0 else "decreases"
    print(f"  {name}: OR={or_value:.3f} ({direction} risk)")
```

### 【思考问题】
1. 逻辑回归的系数如何解释为"优势比"（Odds Ratio）？OR>2代表什么？
2. 在医疗场景中，为什么AUC比准确率更适合评估模型？
3. 当数据类别极度不平衡时（中风患者很少），逻辑回归会出现什么问题？

### 【拓展任务】
尝试调整分类阈值（默认0.5），观察精确率和召回率的变化。在医疗诊断中，应该更关注哪个指标？

---

## 附录：文件清单

| 文件名 | 说明 | 大小 |
|--------|------|------|
| kaggle_task1_online_retail.csv | UCI Online Retail购物数据 | 116.5 KB |
| kaggle_task2_mall_customers.csv | Kaggle Mall Customer数据 | 9.2 KB |
| kaggle_task3_telco_churn.csv | Kaggle Telco Churn数据 | 60.6 KB |
| kaggle_task4_heart_disease.csv | UCI Heart Disease数据 | 16.1 KB |
| kaggle_task5_stroke_prediction.csv | Kaggle Stroke数据 | 29.7 KB |
| kaggle_task1_apriori.png | 关联规则可视化 | 193.4 KB |
| kaggle_task2_kmeans.png | K-means聚类可视化 | 473.9 KB |
| kaggle_task3_knn_churn.png | KNN分类可视化 | 178.6 KB |
| kaggle_task4_decision_tree.png | 决策树可视化 | 424.4 KB |
| kaggle_task5_logistic_regression.png | 逻辑回归可视化 | 190.9 KB |

## 课程章节对应表

| 任务引擎 | Kaggle数据集 | 对应章节 | 核心算法 | 应用场景 |
|----------|-------------|----------|----------|----------|
| 任务一 | UCI Online Retail | 第3章 3.1 | Apriori关联规则 | 零售商品推荐 |
| 任务二 | Mall Customer Segmentation | 第4章 4.3 | K-means聚类 | 客户细分 |
| 任务三 | Telco Customer Churn | 第5章 5.1 | KNN分类 | 客户流失预测 |
| 任务四 | Heart Disease UCI | 第5章 5.4 | 决策树分类 | 心脏病诊断 |
| 任务五 | Stroke Prediction | 第6章 | 逻辑回归 | 中风风险预测 |
