import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("任务二：K-means聚类 — 商场客户细分分析")
print("="*60)

csv_path = os.path.join(script_dir, 'Mall_Customers.csv')
df = pd.read_csv(csv_path)
print(f"\n数据集规模: {len(df)} 条记录")
print(f"数据列: {list(df.columns)}")

print("\n" + "-"*40)
print("数据基本统计")
print("-"*40)
print(df.describe())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values.astype(float)

def kmeans_simple(X, k, max_iter=100, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        labels = np.argmin(distances, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

def silhouette_score_simple(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    score = 0
    for i in range(n_samples):
        cluster_i = labels[i]
        a_i = np.mean([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for j in range(n_samples) if labels[j] == cluster_i and j != i]) if np.sum((labels == cluster_i) & (np.arange(n_samples) != i)) > 0 else 0
        b_i = min([np.mean([np.sqrt(np.sum((X[i] - X[j]) ** 2)) for j in range(n_samples) if labels[j] == cluster_j]) for cluster_j in unique_labels if cluster_j != cluster_i]) if len(unique_labels) > 1 else 0
        if a_i + b_i > 0:
            score += (b_i - a_i) / (max(a_i, b_i))
    return score / n_samples

print("\n" + "-"*40)
print("Elbow方法确定最优K值")
print("-"*40)
inertias = []
K_range = range(1, 11)
for k in K_range:
    labels, centroids = kmeans_simple(X, k)
    inertia = sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(k)])
    inertias.append(inertia)
    print(f"  K={k}: Inertia={inertia:.2f}")

print("\n" + "-"*40)
print("轮廓系数评估")
print("-"*40)
sil_scores = {}
for k in range(2, 11):
    labels, centroids = kmeans_simple(X, k)
    sil_scores[k] = silhouette_score_simple(X, labels)
    print(f"  K={k}: 轮廓系数={sil_scores[k]:.4f}")

optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\n最优聚类数: K={optimal_k} (轮廓系数={sil_scores[optimal_k]:.4f})")

print("\n" + "-"*40)
print(f"K-means聚类结果 (K={optimal_k})")
print("-"*40)
df['Cluster'], centers = kmeans_simple(X, optimal_k)

print("\n各聚类中心:")
for i, center in enumerate(centers):
    print(f"  聚类{i}: 收入={center[0]:.2f}k$, 消费评分={center[1]:.2f}")

print("\n各聚类客户数量和特征:")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\n  聚类{cluster} ({len(cluster_data)}人):")
    print(f"    年龄: 均值={cluster_data['Age'].mean():.1f}, 范围=[{cluster_data['Age'].min()}-{cluster_data['Age'].max()}]")
    print(f"    年收入: 均值={cluster_data['Annual Income (k$)'].mean():.1f}k$")
    print(f"    消费评分: 均值={cluster_data['Spending Score (1-100)'].mean():.1f}")
    gender_dist = cluster_data['Gender'].value_counts(normalize=True)
    print(f"    性别: 女性={gender_dist.get('Female', 0)*100:.1f}%, 男性={gender_dist.get('Male', 0)*100:.1f}%")

print("\n" + "-"*40)
print("聚类特征总结")
print("-"*40)
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Count'})
cluster_summary = cluster_summary.round(2)
print(cluster_summary.to_string())

print("\n" + "="*60)
print("聚类分析完成!")
print("="*60)