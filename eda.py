"""
추천시스템 대회 EDA
- 데이터: 온라인 스토어 유저 행동 로그
- Train: 19.11.01 ~ 20.02.29
- Test: 20.03.01 ~ 20.03.07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (맥북)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 데이터 로드
# ============================================
print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)

train = pd.read_parquet('data/train.parquet')
submission = pd.read_csv('data/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Submission shape: {submission.shape}")
print(f"\nTrain columns: {train.columns.tolist()}")
print(f"\nTrain dtypes:\n{train.dtypes}")

# ============================================
# 2. 기본 통계
# ============================================
print("\n" + "=" * 50)
print("2. 기본 통계")
print("=" * 50)

print(f"\n총 interaction 수: {len(train):,}")
print(f"유니크 유저 수: {train['user_id'].nunique():,}")
print(f"유니크 아이템 수: {train['item_id'].nunique():,}")
print(f"유니크 세션 수: {train['user_session'].nunique():,}")
print(f"유니크 카테고리 수: {train['category_code'].nunique():,}")
print(f"유니크 브랜드 수: {train['brand'].nunique():,}")

# 제출 유저 수 확인
submission_users = submission['user_id'].nunique()
print(f"\n제출 대상 유저 수: {submission_users:,}")

# ============================================
# 3. Event Type 분석
# ============================================
print("\n" + "=" * 50)
print("3. Event Type 분석")
print("=" * 50)

event_counts = train['event_type'].value_counts()
print(f"\nEvent Type 분포:")
for event, count in event_counts.items():
    print(f"  {event}: {count:,} ({count/len(train)*100:.2f}%)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 파이 차트
axes[0].pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', 
            colors=['#66b3ff', '#99ff99', '#ff9999'])
axes[0].set_title('Event Type 비율')

# 바 차트
sns.barplot(x=event_counts.index, y=event_counts.values, ax=axes[1], palette='viridis')
axes[1].set_title('Event Type 별 건수')
axes[1].set_ylabel('Count')
for i, v in enumerate(event_counts.values):
    axes[1].text(i, v + 50000, f'{v:,}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('eda_01_event_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_01_event_type.png 저장")

# ============================================
# 4. 유저 분석
# ============================================
print("\n" + "=" * 50)
print("4. 유저 분석")
print("=" * 50)

# 유저당 interaction 수
user_interactions = train.groupby('user_id').size()
print(f"\n유저당 interaction 수:")
print(f"  평균: {user_interactions.mean():.2f}")
print(f"  중앙값: {user_interactions.median():.2f}")
print(f"  최소: {user_interactions.min()}")
print(f"  최대: {user_interactions.max()}")
print(f"  표준편차: {user_interactions.std():.2f}")

# 분위수
print(f"\n분위수:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {int(q*100)}%: {user_interactions.quantile(q):.0f}")

# 유저당 event_type 별 수
user_event_counts = train.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
print(f"\n유저당 평균 event 수:")
for col in user_event_counts.columns:
    print(f"  {col}: {user_event_counts[col].mean():.2f}")

# 구매 유저 비율
purchase_users = train[train['event_type'] == 'purchase']['user_id'].nunique()
print(f"\n구매 경험 있는 유저: {purchase_users:,} ({purchase_users/train['user_id'].nunique()*100:.2f}%)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 유저당 interaction 분포 (log scale)
axes[0].hist(user_interactions, bins=100, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Interactions per User')
axes[0].set_ylabel('Number of Users')
axes[0].set_title('유저당 Interaction 분포')
axes[0].set_yscale('log')

# 유저당 interaction 분포 (상위 제외)
user_interactions_clipped = user_interactions[user_interactions <= user_interactions.quantile(0.95)]
axes[1].hist(user_interactions_clipped, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Interactions per User')
axes[1].set_ylabel('Number of Users')
axes[1].set_title('유저당 Interaction 분포 (95% 이하)')

plt.tight_layout()
plt.savefig('eda_02_user_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_02_user_distribution.png 저장")

# ============================================
# 5. 아이템 분석
# ============================================
print("\n" + "=" * 50)
print("5. 아이템 분석")
print("=" * 50)

# 아이템별 interaction 수
item_interactions = train.groupby('item_id').size()
print(f"\n아이템당 interaction 수:")
print(f"  평균: {item_interactions.mean():.2f}")
print(f"  중앙값: {item_interactions.median():.2f}")
print(f"  최소: {item_interactions.min()}")
print(f"  최대: {item_interactions.max()}")

# 분위수
print(f"\n분위수:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {int(q*100)}%: {item_interactions.quantile(q):.0f}")

# 아이템별 구매 수
item_purchases = train[train['event_type'] == 'purchase'].groupby('item_id').size()
print(f"\n구매된 적 있는 아이템: {len(item_purchases):,} ({len(item_purchases)/train['item_id'].nunique()*100:.2f}%)")

# 인기 아이템 Top 20
print(f"\n인기 아이템 Top 10 (interaction 기준):")
for i, (item, count) in enumerate(item_interactions.nlargest(10).items()):
    print(f"  {i+1}. {item[:20]}... : {count:,}")

# 시각화 - Long tail 분포
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 아이템 인기도 분포 (sorted)
sorted_counts = item_interactions.sort_values(ascending=False).values
axes[0].plot(range(len(sorted_counts)), sorted_counts)
axes[0].set_xlabel('Item Rank')
axes[0].set_ylabel('Interaction Count')
axes[0].set_title('아이템 인기도 Long-tail 분포')
axes[0].set_yscale('log')

# 상위 N% 아이템이 차지하는 비율
cumsum = np.cumsum(sorted_counts) / np.sum(sorted_counts)
axes[1].plot(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum * 100)
axes[1].axhline(y=80, color='r', linestyle='--', label='80% interactions')
axes[1].axvline(x=20, color='g', linestyle='--', label='20% items')
axes[1].set_xlabel('Top N% Items')
axes[1].set_ylabel('Cumulative % of Interactions')
axes[1].set_title('아이템 Pareto 분포')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_03_item_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_03_item_distribution.png 저장")

# ============================================
# 6. 시간 분석
# ============================================
print("\n" + "=" * 50)
print("6. 시간 분석")
print("=" * 50)

train['event_time'] = pd.to_datetime(train['event_time'])
train['date'] = train['event_time'].dt.date
train['hour'] = train['event_time'].dt.hour
train['dayofweek'] = train['event_time'].dt.dayofweek
train['week'] = train['event_time'].dt.isocalendar().week

print(f"\n데이터 기간: {train['event_time'].min()} ~ {train['event_time'].max()}")
print(f"총 일수: {(train['event_time'].max() - train['event_time'].min()).days}일")

# 일별 interaction 수
daily_counts = train.groupby('date').size()
print(f"\n일별 평균 interaction: {daily_counts.mean():,.0f}")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 일별 트렌드
daily_events = train.groupby(['date', 'event_type']).size().unstack(fill_value=0)
daily_events.plot(ax=axes[0, 0], alpha=0.8)
axes[0, 0].set_title('일별 Event Type 추이')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend(loc='upper right')

# 시간대별 분포
hourly_counts = train.groupby(['hour', 'event_type']).size().unstack(fill_value=0)
hourly_counts.plot(kind='bar', ax=axes[0, 1], width=0.8)
axes[0, 1].set_title('시간대별 Event Type 분포')
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(loc='upper right')
axes[0, 1].tick_params(axis='x', rotation=0)

# 요일별 분포
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_counts = train.groupby(['dayofweek', 'event_type']).size().unstack(fill_value=0)
dow_counts.index = dow_names
dow_counts.plot(kind='bar', ax=axes[1, 0], width=0.8)
axes[1, 0].set_title('요일별 Event Type 분포')
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(loc='upper right')
axes[1, 0].tick_params(axis='x', rotation=0)

# 주차별 추이
weekly_counts = train.groupby('week').size()
axes[1, 1].plot(weekly_counts.index, weekly_counts.values, marker='o')
axes[1, 1].set_title('주차별 Interaction 추이')
axes[1, 1].set_xlabel('Week')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_04_time_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_04_time_analysis.png 저장")

# ============================================
# 7. 전환율 분석
# ============================================
print("\n" + "=" * 50)
print("7. 전환율 분석 (view → cart → purchase)")
print("=" * 50)

# 유저별 event_type 유무
user_events = train.groupby('user_id')['event_type'].apply(set)

view_users = sum(1 for events in user_events if 'view' in events)
cart_users = sum(1 for events in user_events if 'cart' in events)
purchase_users = sum(1 for events in user_events if 'purchase' in events)
view_to_cart = sum(1 for events in user_events if 'view' in events and 'cart' in events)
view_to_purchase = sum(1 for events in user_events if 'view' in events and 'purchase' in events)
cart_to_purchase = sum(1 for events in user_events if 'cart' in events and 'purchase' in events)

total_users = train['user_id'].nunique()

print(f"\n유저 기준 전환율:")
print(f"  View 경험 유저: {view_users:,} ({view_users/total_users*100:.1f}%)")
print(f"  Cart 경험 유저: {cart_users:,} ({cart_users/total_users*100:.1f}%)")
print(f"  Purchase 경험 유저: {purchase_users:,} ({purchase_users/total_users*100:.1f}%)")
print(f"\n  View → Cart 전환율: {view_to_cart/view_users*100:.2f}%")
print(f"  View → Purchase 전환율: {view_to_purchase/view_users*100:.2f}%")
print(f"  Cart → Purchase 전환율: {cart_to_purchase/cart_users*100:.2f}%")

# 아이템 기준 전환율
item_events = train.groupby('item_id')['event_type'].apply(set)
item_view = sum(1 for events in item_events if 'view' in events)
item_cart = sum(1 for events in item_events if 'cart' in events)
item_purchase = sum(1 for events in item_events if 'purchase' in events)

print(f"\n아이템 기준:")
print(f"  View된 아이템: {item_view:,}")
print(f"  Cart된 아이템: {item_cart:,}")
print(f"  Purchase된 아이템: {item_purchase:,}")

# ============================================
# 8. 카테고리 분석
# ============================================
print("\n" + "=" * 50)
print("8. 카테고리 분석")
print("=" * 50)

# 결측치 확인
cat_missing = train['category_code'].isna().sum()
print(f"\n카테고리 결측치: {cat_missing:,} ({cat_missing/len(train)*100:.2f}%)")

# 상위 카테고리 (. 기준 첫번째)
train['category_main'] = train['category_code'].fillna('unknown').apply(lambda x: x.split('.')[0])
cat_counts = train['category_main'].value_counts()

print(f"\n메인 카테고리 Top 10:")
for i, (cat, count) in enumerate(cat_counts.head(10).items()):
    print(f"  {i+1}. {cat}: {count:,} ({count/len(train)*100:.1f}%)")

# 카테고리별 구매율
cat_purchase_rate = train.groupby('category_main').apply(
    lambda x: (x['event_type'] == 'purchase').sum() / len(x) * 100
).sort_values(ascending=False)

print(f"\n카테고리별 구매율 Top 10:")
for i, (cat, rate) in enumerate(cat_purchase_rate.head(10).items()):
    print(f"  {i+1}. {cat}: {rate:.2f}%")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 카테고리 분포
top_cats = cat_counts.head(15)
sns.barplot(y=top_cats.index, x=top_cats.values, ax=axes[0], palette='viridis')
axes[0].set_title('메인 카테고리 Top 15')
axes[0].set_xlabel('Count')

# 카테고리별 구매율
top_purchase_cats = cat_purchase_rate.head(15)
sns.barplot(y=top_purchase_cats.index, x=top_purchase_cats.values, ax=axes[1], palette='rocket')
axes[1].set_title('카테고리별 구매율 Top 15')
axes[1].set_xlabel('Purchase Rate (%)')

plt.tight_layout()
plt.savefig('eda_05_category_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_05_category_analysis.png 저장")

# ============================================
# 9. 브랜드 분석
# ============================================
print("\n" + "=" * 50)
print("9. 브랜드 분석")
print("=" * 50)

# 결측치 확인
brand_missing = train['brand'].isna().sum()
print(f"\n브랜드 결측치: {brand_missing:,} ({brand_missing/len(train)*100:.2f}%)")

# 인기 브랜드
brand_counts = train['brand'].value_counts()
print(f"\n인기 브랜드 Top 10:")
for i, (brand, count) in enumerate(brand_counts.head(10).items()):
    print(f"  {i+1}. {brand}: {count:,}")

# 브랜드별 구매율
brand_purchase_rate = train.groupby('brand').apply(
    lambda x: (x['event_type'] == 'purchase').sum() / len(x) * 100
)
# 최소 100건 이상인 브랜드만
brand_min_count = brand_counts[brand_counts >= 100].index
brand_purchase_rate_filtered = brand_purchase_rate[brand_purchase_rate.index.isin(brand_min_count)]
brand_purchase_rate_filtered = brand_purchase_rate_filtered.sort_values(ascending=False)

print(f"\n브랜드별 구매율 Top 10 (최소 100건 이상):")
for i, (brand, rate) in enumerate(brand_purchase_rate_filtered.head(10).items()):
    print(f"  {i+1}. {brand}: {rate:.2f}%")

# ============================================
# 10. 가격 분석
# ============================================
print("\n" + "=" * 50)
print("10. 가격 분석")
print("=" * 50)

print(f"\n가격 통계:")
print(f"  평균: ${train['price'].mean():.2f}")
print(f"  중앙값: ${train['price'].median():.2f}")
print(f"  최소: ${train['price'].min():.2f}")
print(f"  최대: ${train['price'].max():.2f}")
print(f"  표준편차: ${train['price'].std():.2f}")

# 가격대 구분
train['price_range'] = pd.cut(train['price'], 
                               bins=[0, 10, 50, 100, 500, float('inf')],
                               labels=['~$10', '$10-50', '$50-100', '$100-500', '$500+'])

price_range_counts = train['price_range'].value_counts().sort_index()
print(f"\n가격대별 분포:")
for range_name, count in price_range_counts.items():
    print(f"  {range_name}: {count:,} ({count/len(train)*100:.1f}%)")

# 가격대별 구매율
price_purchase_rate = train.groupby('price_range').apply(
    lambda x: (x['event_type'] == 'purchase').sum() / len(x) * 100
)
print(f"\n가격대별 구매율:")
for range_name, rate in price_purchase_rate.items():
    print(f"  {range_name}: {rate:.2f}%")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 가격 분포 (log scale)
axes[0].hist(train['price'][train['price'] <= train['price'].quantile(0.99)], 
             bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Price ($)')
axes[0].set_ylabel('Count')
axes[0].set_title('가격 분포 (99% 이하)')

# 가격대별 구매율
sns.barplot(x=price_purchase_rate.index, y=price_purchase_rate.values, ax=axes[1], palette='coolwarm')
axes[1].set_xlabel('Price Range')
axes[1].set_ylabel('Purchase Rate (%)')
axes[1].set_title('가격대별 구매율')

plt.tight_layout()
plt.savefig('eda_06_price_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ eda_06_price_analysis.png 저장")

# ============================================
# 11. 세션 분석
# ============================================
print("\n" + "=" * 50)
print("11. 세션 분석")
print("=" * 50)

# 세션당 아이템 수
session_items = train.groupby('user_session')['item_id'].nunique()
print(f"\n세션당 유니크 아이템 수:")
print(f"  평균: {session_items.mean():.2f}")
print(f"  중앙값: {session_items.median():.2f}")
print(f"  최대: {session_items.max()}")

# 세션당 구매 여부
session_has_purchase = train.groupby('user_session')['event_type'].apply(
    lambda x: 'purchase' in x.values
)
print(f"\n구매가 발생한 세션: {session_has_purchase.sum():,} ({session_has_purchase.mean()*100:.2f}%)")

# 유저당 세션 수
user_sessions = train.groupby('user_id')['user_session'].nunique()
print(f"\n유저당 세션 수:")
print(f"  평균: {user_sessions.mean():.2f}")
print(f"  중앙값: {user_sessions.median():.2f}")

# ============================================
# 12. 최근 데이터 중요도 분석
# ============================================
print("\n" + "=" * 50)
print("12. 최근 데이터 중요도 분석")
print("=" * 50)

# 마지막 7일, 14일, 30일 데이터 비율
max_date = train['event_time'].max()
for days in [7, 14, 30]:
    cutoff = max_date - pd.Timedelta(days=days)
    recent_data = train[train['event_time'] >= cutoff]
    recent_users = recent_data['user_id'].nunique()
    recent_items = recent_data['item_id'].nunique()
    print(f"\n최근 {days}일:")
    print(f"  데이터 비율: {len(recent_data)/len(train)*100:.1f}%")
    print(f"  활성 유저: {recent_users:,} ({recent_users/train['user_id'].nunique()*100:.1f}%)")
    print(f"  활성 아이템: {recent_items:,} ({recent_items/train['item_id'].nunique()*100:.1f}%)")

# ============================================
# 13. Cold Start 분석
# ============================================
print("\n" + "=" * 50)
print("13. Cold Start 분석")
print("=" * 50)

# 제출 대상 유저 중 train에 없는 유저
train_users = set(train['user_id'].unique())
submission_users_set = set(submission['user_id'].unique())
cold_users = submission_users_set - train_users

print(f"\n제출 대상 유저: {len(submission_users_set):,}")
print(f"Train에 있는 유저: {len(train_users & submission_users_set):,}")
print(f"Cold Start 유저: {len(cold_users):,} ({len(cold_users)/len(submission_users_set)*100:.2f}%)")

# Interaction 적은 유저
interaction_thresholds = [1, 2, 3, 5, 10]
for threshold in interaction_thresholds:
    sparse_users = (user_interactions <= threshold).sum()
    print(f"  Interaction <= {threshold}: {sparse_users:,} ({sparse_users/len(user_interactions)*100:.1f}%)")

# ============================================
# Summary 저장
# ============================================
print("\n" + "=" * 50)
print("EDA 완료! 저장된 파일:")
print("=" * 50)
print("  - eda_01_event_type.png")
print("  - eda_02_user_distribution.png")
print("  - eda_03_item_distribution.png")
print("  - eda_04_time_analysis.png")
print("  - eda_05_category_analysis.png")
print("  - eda_06_price_analysis.png")