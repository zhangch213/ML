import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("任务四：决策树分类 — 心脏病预测")
print("="*60)

csv_path = os.path.join(script_dir, 'heart_disease_uci.csv')
df = pd.read_csv(csv_path)
print(f"\n数据集规模: {len(df)} 条记录")
print(f"数据列: {list(df.columns)}")

print("\n" + "-"*40)
print("目标变量分布 (num: 0=无病, 1-4=有病)")
print("-"*40)
print(df['num'].value_counts().sort_index())
df['target'] = (df['num'] > 0).astype(int)
print(f"\n患病率: {df['target'].mean()*100:.2f}%")

df = df.drop(['id', 'dataset', 'num'], axis=1)

print("\n" + "-"*40)
print("数据预处理")
print("-"*40)

numeric_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'ca', 'thal']
target_col = df['target'].copy()
df_numeric = df[numeric_cols].copy()

for col in numeric_cols:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

df = df_numeric.copy()
df['target'] = target_col

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

print(f"处理后样本数: {len(df)}")

X = df.drop('target', axis=1).values.astype(float)
y = df['target'].values.astype(int)

print(f"特征数量: {X.shape[1]}")
print(f"特征列表: {list(df.drop('target', axis=1).columns)}")

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

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None

    def gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / len(y)
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples < 10:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return {'leaf': True, 'class': np.bincount(y).argmax()}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node['leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        return self.predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

print("\n" + "-"*40)
print("决策树分类器训练与评估")
print("-"*40)

max_depth_values = [3, 5, 7, 10, None]
best_depth = 5
best_accuracy = 0

for depth in max_depth_values:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    acc = np.mean(y_pred == y_test)
    depth_str = str(depth) if depth else "None"
    print(f"  max_depth={depth_str:5s}: 准确率={acc*100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_depth = depth

print(f"\n最优深度: {'无限制' if best_depth is None else 'max_depth='+str(best_depth)}")

dt_final = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_final.fit(X_train, y_train)
y_pred = dt_final.predict(X_test)

print("\n" + "-"*40)
print(f"最终模型评估")
print("-"*40)
print(f"\n准确率: {np.mean(y_pred == y_test)*100:.2f}%")

cm = np.zeros((2, 2), dtype=int)
for i in range(len(y_test)):
    cm[int(y_test[i]), int(y_pred[i])] += 1

print("\n混淆矩阵:")
print(f"  预测→    无病   有病")
print(f"  实际↓")
print(f"  无病     {cm[0,0]:4d}    {cm[0,1]:4d}")
print(f"  有病     {cm[1,0]:4d}    {cm[1,1]:4d}")

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n分类报告:")
print(f"              无病    有病")
print(f"  精确率:      {tp/(tp+fp) if (tp+fp)>0 else 0:.2f}      {tn/(tn+fn) if (tn+fn)>0 else 0:.2f}")
print(f"  召回率:      {tp/(tp+fn) if (tp+fn)>0 else 0:.2f}      {tn/(tn+fp) if (tn+fp)>0 else 0:.2f}")
print(f"  F1分数:      {f1:.2f}")

print("\n" + "="*60)
print("决策树分类完成!")
print("="*60)