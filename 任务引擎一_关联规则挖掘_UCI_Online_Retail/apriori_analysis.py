import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import warnings
import os
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("任务一：关联规则挖掘 — UCI Online Retail")
print("="*60)

csv_path = os.path.join(script_dir, 'online_retail_II.csv')
df = pd.read_csv(csv_path, encoding='latin1')
print(f"\n原始数据集规模: {len(df)} 条记录")
print(f"数据列: {list(df.columns)}")

print("\n" + "-"*40)
print("数据预处理")
print("-"*40)

df = df.dropna(subset=['Invoice', 'Description'])
df = df[~df['Invoice'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]

print(f"清洗后记录数: {len(df)}")

top_items = df['Description'].value_counts().head(30).index.tolist()
df = df[df['Description'].isin(top_items)]

print(f"选取Top 30热门商品后记录数: {len(df)}")
print(f"商品种类: {df['Description'].nunique()}")
print(f"交易次数: {df['Invoice'].nunique()}")

print("\n" + "-"*40)
print("构建事务数据集")
print("-"*40)

basket = df.groupby('Invoice')['Description'].apply(lambda x: set(x.astype(str))).reset_index()
transactions = basket['Description'].tolist()
print(f"事务数量: {len(transactions)}")

print("\n" + "-"*40)
print("Apriori算法实现")
print("-"*40)

def get_support(itemset, transactions):
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def apriori(transactions, min_support=0.02, max_length=3):
    items = set()
    for t in transactions:
        items.update(t)
    items = sorted(list(items))
    
    frequent_itemsets = []
    
    k = 1
    current_itemsets = [frozenset([item]) for item in items]
    while current_itemsets:
        print(f"  候选项数量 (k={k}): {len(current_itemsets)}")
        
        frequent_k = []
        for itemset in current_itemsets:
            support = get_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets.append((itemset, support))
                frequent_k.append(itemset)
        
        print(f"  频繁项数量 (k={k}): {len(frequent_k)}")
        
        if k >= max_length:
            break
            
        current_itemsets = []
        for i in range(len(frequent_k)):
            for j in range(i+1, len(frequent_k)):
                union = frequent_k[i] | frequent_k[j]
                if len(union) == k + 1:
                    all_subsets = True
                    for item in union:
                        if union - {item} not in frequent_k:
                            all_subsets = False
                            break
                    if all_subsets and union not in current_itemsets:
                        current_itemsets.append(union)
        
        k += 1
    
    return frequent_itemsets

min_support = 0.02
print(f"最小支持度: {min_support*100:.1f}%")
frequent_itemsets = apriori(transactions, min_support=min_support, max_length=3)

print(f"\n发现频繁项集总数: {len(frequent_itemsets)}")

print("\n" + "-"*40)
print("频繁项集列表 (支持度降序)")
print("-"*40)

frequent_itemsets.sort(key=lambda x: -x[1])
for itemset, support in frequent_itemsets[:20]:
    items_str = ', '.join(sorted(list(itemset)))
    print(f"  支持度 {support*100:5.2f}%: {items_str}")

print("\n" + "-"*40)
print("关联规则挖掘")
print("-"*40)

def generate_rules(frequent_itemsets, transactions, min_confidence=0.5):
    rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue
        
        for item in itemset:
            consequent = frozenset([item])
            antecedent = itemset - consequent
            antecedent_support = get_support(antecedent, transactions)
            
            if antecedent_support > 0:
                confidence = support / antecedent_support
                if confidence >= min_confidence:
                    lift = confidence / get_support(consequent, transactions) if get_support(consequent, transactions) > 0 else 0
                    rules.append((antecedent, consequent, support, confidence, lift))
    
    return rules

min_confidence = 0.5
rules = generate_rules(frequent_itemsets, transactions, min_confidence=min_confidence)
rules.sort(key=lambda x: -x[3])

print(f"最小置信度: {min_confidence*100:.0f}%")
print(f"发现关联规则数量: {len(rules)}")

print("\n" + "-"*40)
print("Top 15 关联规则 (按置信度排序)")
print("-"*40)
print(f"{'规则':<50} {'支持度':>8} {'置信度':>8} {'提升度':>8}")
print("-"*74)

for antecedent, consequent, support, confidence, lift in rules[:15]:
    ant_str = ', '.join(sorted(list(antecedent)))
    con_str = ', '.join(sorted(list(consequent)))
    rule_str = f"{ant_str} → {con_str}"
    if len(rule_str) > 48:
        rule_str = rule_str[:45] + "..."
    print(f"{rule_str:<50} {support*100:>7.2f}% {confidence*100:>7.2f}% {lift:>8.2f}")

print("\n" + "-"*40)
print("有意义的关联规则分析")
print("-"*40)

high_lift_rules = [r for r in rules if r[4] > 1.5]
high_lift_rules.sort(key=lambda x: -x[4])

print(f"\n高提升度规则 (Lift > 1.5): {len(high_lift_rules)} 条")
for antecedent, consequent, support, confidence, lift in high_lift_rules[:5]:
    ant_str = ', '.join(sorted(list(antecedent)))
    con_str = ', '.join(sorted(list(consequent)))
    print(f"\n  规则: {ant_str} → {con_str}")
    print(f"    支持度: {support*100:.2f}%")
    print(f"    置信度: {confidence*100:.2f}%")
    print(f"    提升度: {lift:.2f}")

print("\n" + "="*60)
print("关联规则挖掘完成!")
print("="*60)