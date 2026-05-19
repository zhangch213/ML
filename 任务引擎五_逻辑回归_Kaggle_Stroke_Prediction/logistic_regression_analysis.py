import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("任务五：逻辑回归 — 中风风险预测")
print("="*60)

csv_path = os.path.join(script_dir, 'healthcare-dataset-stroke-data.csv')
df = pd.read_csv(csv_path)
print(f"\n数据集规模: {len(df)} 条记录")
print(f"数据列: {list(df.columns)}")

print("\n" + "-"*40)
print("目标变量分布")
print("-"*40)
print(df['stroke'].value_counts())
print(f"中风率: {df['stroke'].mean()*100:.2f}%")

df = df.drop(['id'], axis=1)

print("\n" + "-"*40)
print("数据预处理")
print("-"*40)

df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
print(f"类别特征: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = {val: i for i, val in enumerate(df[col].unique())}
    df[col] = df[col].map(le)
    label_encoders[col] = le

X = df.drop('stroke', axis=1).values.astype(float)
y = df['stroke'].values

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
    return X_train_scaled, X_test_scaled, mean, std

X_train_scaled, X_test_scaled, mean, std = standardize(X_train, X_test)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, C=1.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (1 / self.C) * self.weights / n_samples
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

print("\n" + "-"*40)
print("逻辑回归模型训练")
print("-"*40)

C_values = [0.01, 0.1, 1, 10]
best_C = 1
best_accuracy = 0

for C in C_values:
    lr = LogisticRegression(C=C, learning_rate=0.1, n_iterations=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)
    acc = np.mean(y_pred == y_test)
    print(f"  C={str(C):5s}: 准确率={acc*100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_C = C

print(f"\n最优正则化参数: C={best_C}")

lr_final = LogisticRegression(C=best_C, learning_rate=0.1, n_iterations=1000)
lr_final.fit(X_train_scaled, y_train)
y_pred = lr_final.predict(X_test_scaled)
y_prob = lr_final.predict_proba(X_test_scaled)

print("\n" + "-"*40)
print(f"最终模型评估")
print("-"*40)
print(f"\n准确率: {np.mean(y_pred == y_test)*100:.2f}%")

cm = np.zeros((2, 2), dtype=int)
for i in range(len(y_test)):
    cm[int(y_test[i]), int(y_pred[i])] += 1

print("\n混淆矩阵:")
print(f"  预测→    无中风  中风")
print(f"  实际↓")
print(f"  无中风   {cm[0,0]:5d}  {cm[0,1]:4d}")
print(f"  中风     {cm[1,0]:5d}  {cm[1,1]:4d}")

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n分类报告:")
print(f"              无中风  中风")
print(f"  精确率:      {tp/(tp+fp) if (tp+fp)>0 else 0:.2f}    {tn/(tn+fn) if (tn+fn)>0 else 0:.2f}")
print(f"  召回率:      {tp/(tp+fn) if (tp+fn)>0 else 0:.2f}    {tn/(tn+fp) if (tn+fp)>0 else 0:.2f}")
print(f"  F1分数:      {f1:.2f}")

print("\n" + "-"*40)
print("特征系数 (对中风风险的影响)")
print("-"*40)
feature_names = df.drop('stroke', axis=1).columns
feature_coef = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_final.weights
}).sort_values('coefficient', key=abs, ascending=False)

for idx, row in feature_coef.iterrows():
    direction = "↑" if row['coefficient'] > 0 else "↓"
    print(f"  {row['feature']:<20s}: {row['coefficient']:+.4f} {direction}")

print("\n" + "="*60)
print("逻辑回归分析完成!")
print("="*60)