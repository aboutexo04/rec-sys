import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 1. 데이터 로드 및 전처리
# ============================================

def load_data(train_path='data/train.parquet', submission_path='data/sample_submission.csv'):
    print("Loading data...")
    train = pd.read_parquet(train_path)
    submission = pd.read_csv(submission_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Unique users: {train['user_id'].nunique():,}")
    print(f"Unique items: {train['item_id'].nunique():,}")
    
    return train, submission

def preprocess(train, 
               event_weights={'view': 1, 'cart': 2, 'purchase': 15},
               use_recent_days=None):
    """전처리 + 가중치 적용"""
    
    train = train.copy()
    train['event_time'] = pd.to_datetime(train['event_time'])
    
    # 최근 N일만 사용
    if use_recent_days:
        max_date = train['event_time'].max()
        cutoff = max_date - pd.Timedelta(days=use_recent_days)
        train = train[train['event_time'] >= cutoff]
        print(f"Using last {use_recent_days} days: {len(train):,} rows")
        
    # event_type 가중치
    train['weight'] = train['event_type'].map(event_weights)
    
    print(f"Event weights: {event_weights}")
    
    return train

# ============================================
# 2. 개선된 베이스라인 모델들
# ============================================

class PopularityBaseline:
    """시간 가중치 적용 인기도 모델"""
    def __init__(self, decay_days=7):
        self.popular_items = None
        self.decay_days = decay_days
        
    def fit(self, train):
        print(f"Fitting PopularityBaseline (decay={self.decay_days}d)...")
        
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / self.decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10, exclude_items=None):
        exclude = set(exclude_items) if exclude_items else set()
        recs = [item for item in self.popular_items if item not in exclude]
        return recs[:n]

class UserHistoryBaseline:
    """유저 히스토리 기반 (시간 가중치 강화)"""
    def __init__(self, decay_days=7):
        self.user_item_scores = {}
        self.popular_items = None
        self.decay_days = decay_days
        
    def fit(self, train):
        print(f"Fitting UserHistoryBaseline (decay={self.decay_days}d)...")
        
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / self.decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        # 유저별 아이템 스코어
        user_item_scores = train.groupby(['user_id', 'item_id'])['final_weight'].sum()
        
        print("Building user-item scores...")
        for (user_id, item_id), score in tqdm(user_item_scores.items()):
            if user_id not in self.user_item_scores:
                self.user_item_scores[user_id] = {}
            self.user_item_scores[user_id][item_id] = score
            
        # 인기 아이템
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10):
        if user_id in self.user_item_scores:
            user_scores = self.user_item_scores[user_id]
            sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
            recs = [item for item, _ in sorted_items][:n]
            
            if len(recs) < n:
                exclude = set(recs)
                for item in self.popular_items:
                    if item not in exclude:
                        recs.append(item)
                    if len(recs) >= n:
                        break
        else:
            recs = self.popular_items[:n]
            
        return recs[:n]

class RecentPurchaseBaseline:
    """최근 구매 + Co-occurrence (강화 버전)"""
    def __init__(self, decay_days=7, cooccur_min_count=2):
        self.item_cooccur = defaultdict(lambda: defaultdict(float))
        self.user_recent_items = {}
        self.popular_items = None
        self.decay_days = decay_days
        self.cooccur_min_count = cooccur_min_count
        
    def fit(self, train):
        print(f"Fitting RecentPurchaseBaseline (decay={self.decay_days}d)...")
        
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / self.decay_days)
        
        # 구매 데이터 co-occurrence (시간 가중치 적용)
        purchases = train[train['event_type'] == 'purchase']
        
        print("Building item co-occurrence...")
        session_data = purchases.groupby(['user_id', 'user_session']).agg({
            'item_id': list,
            'time_weight': 'mean'
        })
        
        for _, row in tqdm(session_data.iterrows(), total=len(session_data)):
            items = list(set(row['item_id']))
            tw = row['time_weight']
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    self.item_cooccur[item1][item2] += tw
                    self.item_cooccur[item2][item1] += tw
                    
        # 유저별 최근 아이템 (가중치 적용)
        print("Building user recent items...")
        train['final_weight'] = train['weight'] * train['time_weight']
        
        for user_id, group in tqdm(train.groupby('user_id')):
            items_weights = group.groupby('item_id')['final_weight'].sum()
            self.user_recent_items[user_id] = items_weights.sort_values(ascending=False).index.tolist()[:30]
            
        # 인기 아이템
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10):
        recs = []
        seen = set()
        
        if user_id in self.user_recent_items:
            recent_items = self.user_recent_items[user_id]
            
            # Co-occur 스코어 계산
            cooccur_scores = defaultdict(float)
            for i, item in enumerate(recent_items):
                weight = 1.0 / (i + 1)
                if item in self.item_cooccur:
                    for co_item, count in self.item_cooccur[item].items():
                        if count >= self.cooccur_min_count:
                            cooccur_scores[co_item] += count * weight
                            
            # Co-occur 기반 추천
            sorted_cooccur = sorted(cooccur_scores.items(), key=lambda x: x[1], reverse=True)
            for item, _ in sorted_cooccur:
                if item not in seen:
                    recs.append(item)
                    seen.add(item)
                if len(recs) >= n:
                    break
                    
            # 최근 아이템 추가
            for item in recent_items:
                if item not in seen:
                    recs.append(item)
                    seen.add(item)
                if len(recs) >= n:
                    break
                    
        # 인기 아이템으로 채움
        for item in self.popular_items:
            if item not in seen:
                recs.append(item)
                seen.add(item)
            if len(recs) >= n:
                break
                
        return recs[:n]

class PurchaseOnlyBaseline:
    """구매 데이터만 사용하는 모델"""
    def __init__(self, decay_days=7):
        self.user_purchase_items = {}
        self.popular_purchased = None
        self.decay_days = decay_days
        
    def fit(self, train):
        print(f"Fitting PurchaseOnlyBaseline (decay={self.decay_days}d)...")
        
        # 구매 데이터만
        purchases = train[train['event_type'] == 'purchase'].copy()
        
        max_time = purchases['event_time'].max()
        purchases['days_ago'] = (max_time - purchases['event_time']).dt.total_seconds() / 86400
        purchases['time_weight'] = np.exp(-purchases['days_ago'] / self.decay_days)
        
        # 유저별 구매 아이템
        print("Building user purchase history...")
        user_item_scores = purchases.groupby(['user_id', 'item_id'])['time_weight'].sum()
        
        for (user_id, item_id), score in tqdm(user_item_scores.items()):
            if user_id not in self.user_purchase_items:
                self.user_purchase_items[user_id] = {}
            self.user_purchase_items[user_id][item_id] = score
            
        # 인기 구매 아이템
        item_scores = purchases.groupby('item_id')['time_weight'].sum()
        self.popular_purchased = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10):
        if user_id in self.user_purchase_items:
            user_scores = self.user_purchase_items[user_id]
            sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
            recs = [item for item, _ in sorted_items][:n]
            
            if len(recs) < n:
                exclude = set(recs)
                for item in self.popular_purchased:
                    if item not in exclude:
                        recs.append(item)
                    if len(recs) >= n:
                        break
        else:
            recs = self.popular_purchased[:n]
            
        return recs[:n]

# ============================================
# 3. 앙상블
# ============================================

class TunedEnsemble:
    """튜닝된 앙상블 모델"""
    def __init__(self, weights=None):
        self.models = []
        self.weights = weights
        self.popular_items = None
        
    def fit(self, train, decay_days=7):
        # 모델 구성
        self.models = [
            ('UserHistory', UserHistoryBaseline(decay_days=decay_days), 2.5),
            ('RecentPurchase', RecentPurchaseBaseline(decay_days=decay_days, cooccur_min_count=2), 2.0),
            ('PurchaseOnly', PurchaseOnlyBaseline(decay_days=decay_days), 1.5),
            ('Popularity', PopularityBaseline(decay_days=decay_days), 1.0),
        ]
        
        for name, model, _ in self.models:
            model.fit(train)
            
        # 인기 아이템 (fallback용)
        max_time = train['event_time'].max()
        train = train.copy()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10):
        item_scores = defaultdict(float)
        
        for name, model, weight in self.models:
            recs = model.recommend(user_id, n=n*2)
            for rank, item in enumerate(recs):
                # 순위 기반 스코어
                item_scores[item] += weight / (rank + 1)
                
        # 스코어 순 정렬
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recs = [item for item, _ in sorted_items[:n]]
        
        # 부족하면 인기 아이템
        if len(recs) < n:
            seen = set(recs)
            for item in self.popular_items:
                if item not in seen:
                    recs.append(item)
                if len(recs) >= n:
                    break
                    
        return recs[:n]

# ============================================
# 4. 로컬 검증
# ============================================

def calculate_ndcg(y_true, y_pred, k=10):
    dcg = 0.0
    for i, item in enumerate(y_pred[:k]):
        if item in y_true:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    return dcg / idcg if idcg > 0 else 0.0

def local_validation(train, decay_days=7, val_days=7):
    print(f"\n{'='*50}")
    print(f"Local Validation (decay={decay_days}d, val={val_days}d)")
    print('='*50)
    
    max_date = train['event_time'].max()
    val_start = max_date - pd.Timedelta(days=val_days)
    
    train_data = train[train['event_time'] < val_start].copy()
    val_data = train[train['event_time'] >= val_start].copy()
    
    val_purchases = val_data[val_data['event_type'] == 'purchase']
    val_gt = val_purchases.groupby('user_id')['item_id'].apply(set).to_dict()
    
    print(f"Train: {train_data['event_time'].min().date()} ~ {train_data['event_time'].max().date()}")
    print(f"Val: {val_data['event_time'].min().date()} ~ {val_data['event_time'].max().date()}")
    print(f"Val users: {len(val_gt):,}")
    
    # 모델 학습
    model = TunedEnsemble()
    model.fit(train_data, decay_days=decay_days)
    
    # 평가
    ndcg_scores = []
    for user_id, true_items in tqdm(val_gt.items(), desc="Evaluating"):
        pred_items = model.recommend(user_id, n=10)
        ndcg = calculate_ndcg(true_items, pred_items, k=10)
        ndcg_scores.append(ndcg)
        
    mean_ndcg = np.mean(ndcg_scores)
    print(f"\n>>> NDCG@10: {mean_ndcg:.6f}")
    
    return mean_ndcg

# ============================================
# 5. 제출 파일 생성
# ============================================

def generate_submission(model, submission, output_path='submission_tuned_14.csv'):
    print(f"\n{'='*50}")
    print("Generating submission...")
    print('='*50)
    
    users = submission['user_id'].unique()
    print(f"Total users: {len(users):,}")
    
    results = []
    for user_id in tqdm(users, desc="Recommending"):
        recs = model.recommend(user_id, n=10)
        for item_id in recs:
            results.append({'user_id': user_id, 'item_id': item_id})
            
    submission_df = pd.DataFrame(results)
    
    # 검증
    assert len(submission_df) == len(submission)
    items_per_user = submission_df.groupby('user_id').size()
    assert (items_per_user == 10).all()
    
    submission_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    
    return submission_df

# ============================================
# 6. 메인
# ============================================

def main():
    # 데이터 로드
    train, submission = load_data('data/train.parquet', 'data/sample_submission.csv')
    
    # 전처리 (가중치 튜닝!)
    train = preprocess(
        train,
        event_weights={'view': 1, 'cart': 3, 'purchase': 10},  # purchase 강화
        use_recent_days=None  # 전체 사용 (또는 60, 90)
    )
    
    # ========== 파라미터 ==========
    DECAY_DAYS = 14  # 시간 가중치 (7일 반감기)
    
    # ========== 로컬 검증 (선택) ==========
    # local_validation(train, decay_days=DECAY_DAYS, val_days=7)
    
    # ========== 전체 학습 & 제출 ==========
    model = TunedEnsemble()
    model.fit(train, decay_days=DECAY_DAYS)
    
    generate_submission(model, submission, 'submission_tuned_14.csv')
    
    print("\nDone!")

if __name__ == "__main__":
    main()
