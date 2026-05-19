import pandas as pd
import numpy as np
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("任务三：KNN分类 — 电信客户流失预测")
print("="*60)

csv_path = os.path.join(script_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = pd.read_csv(csv_path)
print(f"\n数据集规模: {len(df)} 条记录")
print(f"数据列: {list(df.columns)}")

print("\n" + "-"*40)
print("目标变量分布")
print("-"*40)
print(df['Churn'].value_counts())
print(f"流失率: {(df['Churn']=='Yes').mean()*100:.2f}%")

customer_ids = df['customerID']
df = df.drop(['customerID'], axis=1)

print("\n" + "-"*40)
print("数据预处理")
print("-"*40)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')
print(f"类别特征: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = {val: i for i, val in enumerate(df[col].unique())}
    df[col] = df[col].map(le)
    label_encoders[col] = le

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df.drop('Churn', axis=1).values.astype(float)
y = df['Churn'].values

print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")

def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

X_train_scaled, X_test_scaled = standardize(X_train, X_test)

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            most_common = Counter(k_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

print("\n" + "-"*40)
print("KNN分类器训练与评估")
print("-"*40)

k_values = [3, 5, 7, 9, 11]
best_k = 5
best_accuracy = 0

for k in k_values:
    knn = KNNClassifier(k=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = np.mean(y_pred == y_test)
    print(f"  K={k}: 准确率={acc*100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print(f"\n最优K值: K={best_k}")

knn_final = KNNClassifier(k=best_k)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

print("\n" + "-"*40)
print(f"最终模型评估 (K={best_k})")
print("-"*40)
print(f"\n准确率: {np.mean(y_pred == y_test)*100:.2f}%")

cm = np.zeros((2, 2), dtype=int)
for i in range(len(y_test)):
    cm[int(y_test[i]), int(y_pred[i])] += 1

print("\n混淆矩阵:")
print(f"  预测→    Not Churn  Churn")
print(f"  实际↓")
print(f"  Not Churn   {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"  Churn       {cm[1,0]:4d}    {cm[1,1]:4d}")

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n分类报告:")
print(f"              无流失    流失")
print(f"  精确率(_precision):  {tp/(tp+fp) if (tp+fp)>0 else 0:.2f}      {tn/(tn+fn) if (tn+fn)>0 else 0:.2f}")
print(f"  召回率(recall):     {tp/(tp+fn) if (tp+fn)>0 else 0:.2f}      {tn/(tn+fp) if (tn+fp)>0 else 0:.2f}")
print(f"  F1分数:             {f1:.2f}")

print("\n" + "="*60)
print("KNN分类完成!")
print("="*60)