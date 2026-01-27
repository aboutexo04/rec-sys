"""
추천시스템 대회 베이스라인
- 평가지표: NDCG@10 (binary relevance)
- Train: 19.11.01 ~ 20.02.29
- Test: 20.03.01 ~ 20.03.07 (purchase 예측)
- 제출: 638,257 유저 × 10개 아이템
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 데이터 로드
# ============================================
def load_data(train_path='data/train.parquet', submission_path='data/sample_submission.csv'):
    """데이터 로드"""
    print("Loading data...")
    train = pd.read_parquet(train_path)
    submission = pd.read_csv(submission_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Submission shape: {submission.shape}")
    print(f"Unique users in train: {train['user_id'].nunique()}")
    print(f"Unique items in train: {train['item_id'].nunique()}")
    
    return train, submission


# ============================================
# 2. 전처리
# ============================================
def preprocess(train):
    """기본 전처리"""
    # event_time을 datetime으로 변환
    train['event_time'] = pd.to_datetime(train['event_time'])
    
    # event_type 별 가중치 (purchase > cart > view)
    event_weights = {'view': 1, 'cart': 3, 'purchase': 5}
    train['weight'] = train['event_type'].map(event_weights)
    
    return train


# ============================================
# 3. 베이스라인 모델들
# ============================================

class PopularityBaseline:
    """
    인기도 기반 추천 (가장 단순한 베이스라인)
    - 전체 유저에게 동일한 인기 아이템 추천
    """
    def __init__(self, time_decay=False, decay_days=30):
        self.popular_items = None
        self.time_decay = time_decay
        self.decay_days = decay_days
        
    def fit(self, train):
        print("Fitting PopularityBaseline...")
        
        if self.time_decay:
            # 시간 가중치 적용 (최근 데이터에 더 높은 가중치)
            max_time = train['event_time'].max()
            train = train.copy()
            train['days_ago'] = (max_time - train['event_time']).dt.days
            train['time_weight'] = np.exp(-train['days_ago'] / self.decay_days)
            train['final_weight'] = train['weight'] * train['time_weight']
            
            item_scores = train.groupby('item_id')['final_weight'].sum()
        else:
            # 단순 가중치 합
            item_scores = train.groupby('item_id')['weight'].sum()
        
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        print(f"Total items ranked: {len(self.popular_items)}")
        
    def recommend(self, user_id, user_history=None, n=10):
        """유저에게 n개 아이템 추천 (이미 구매한 아이템 제외 가능)"""
        exclude = set(user_history) if user_history is not None else set()
        recs = [item for item in self.popular_items if item not in exclude]
        return recs[:n]


class UserHistoryBaseline:
    """
    유저 개인화 추천
    - 유저가 과거에 interaction한 아이템 기반
    - purchase > cart > view 가중치
    - 시간 decay 적용 가능
    """
    def __init__(self, time_decay=True, decay_days=14):
        self.user_item_scores = {}
        self.popular_items = None
        self.time_decay = time_decay
        self.decay_days = decay_days
        
    def fit(self, train):
        print("Fitting UserHistoryBaseline...")
        
        train = train.copy()
        max_time = train['event_time'].max()
        
        if self.time_decay:
            train['days_ago'] = (max_time - train['event_time']).dt.days
            train['time_weight'] = np.exp(-train['days_ago'] / self.decay_days)
            train['final_weight'] = train['weight'] * train['time_weight']
        else:
            train['final_weight'] = train['weight']
        
        # 유저별 아이템 스코어 계산
        user_item_scores = train.groupby(['user_id', 'item_id'])['final_weight'].sum()
        
        # 딕셔너리로 변환
        print("Building user-item score dictionary...")
        for (user_id, item_id), score in tqdm(user_item_scores.items()):
            if user_id not in self.user_item_scores:
                self.user_item_scores[user_id] = {}
            self.user_item_scores[user_id][item_id] = score
        
        # Fallback용 인기 아이템
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        
        print(f"Users with history: {len(self.user_item_scores)}")
        
    def recommend(self, user_id, n=10):
        if user_id in self.user_item_scores:
            # 유저의 아이템 스코어로 정렬
            user_scores = self.user_item_scores[user_id]
            sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
            recs = [item for item, score in sorted_items][:n]
            
            # 부족하면 인기 아이템으로 채움
            if len(recs) < n:
                exclude = set(recs)
                for item in self.popular_items:
                    if item not in exclude:
                        recs.append(item)
                    if len(recs) >= n:
                        break
        else:
            # Cold start: 인기 아이템 추천
            recs = self.popular_items[:n]
            
        return recs[:n]


class CategoryAwareBaseline:
    """
    카테고리 인지 추천
    - 유저가 선호하는 카테고리 내에서 인기 아이템 추천
    """
    def __init__(self):
        self.user_categories = {}
        self.category_popular = {}
        self.global_popular = None
        
    def fit(self, train):
        print("Fitting CategoryAwareBaseline...")
        
        # 유저별 선호 카테고리 (purchase > cart > view 가중치 적용)
        user_cat_scores = train.groupby(['user_id', 'category_code'])['weight'].sum()
        
        print("Building user category preferences...")
        for (user_id, cat), score in tqdm(user_cat_scores.items()):
            if pd.isna(cat):
                continue
            if user_id not in self.user_categories:
                self.user_categories[user_id] = {}
            self.user_categories[user_id][cat] = score
        
        # 카테고리별 인기 아이템
        print("Building category popular items...")
        for cat in tqdm(train['category_code'].dropna().unique()):
            cat_data = train[train['category_code'] == cat]
            cat_item_scores = cat_data.groupby('item_id')['weight'].sum()
            self.category_popular[cat] = cat_item_scores.sort_values(ascending=False).index.tolist()
        
        # 글로벌 인기 아이템
        item_scores = train.groupby('item_id')['weight'].sum()
        self.global_popular = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10):
        recs = []
        seen = set()
        
        if user_id in self.user_categories:
            # 유저 선호 카테고리 순으로 아이템 추가
            sorted_cats = sorted(
                self.user_categories[user_id].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for cat, _ in sorted_cats:
                if cat in self.category_popular:
                    for item in self.category_popular[cat]:
                        if item not in seen:
                            recs.append(item)
                            seen.add(item)
                        if len(recs) >= n:
                            break
                if len(recs) >= n:
                    break
        
        # 부족하면 글로벌 인기 아이템으로 채움
        for item in self.global_popular:
            if item not in seen:
                recs.append(item)
                seen.add(item)
            if len(recs) >= n:
                break
                
        return recs[:n]


class RecentPurchaseBaseline:
    """
    최근 구매 기반 추천
    - 최근 purchase 이력이 있는 유저: 같이 구매된 아이템 추천
    - 없으면 최근 view/cart 기반 + 인기 아이템
    """
    def __init__(self, cooccur_min_count=5):
        self.item_cooccur = defaultdict(lambda: defaultdict(int))
        self.user_recent_items = {}
        self.popular_items = None
        self.cooccur_min_count = cooccur_min_count
        
    def fit(self, train):
        print("Fitting RecentPurchaseBaseline...")
        
        # 구매 데이터만 추출
        purchases = train[train['event_type'] == 'purchase'].copy()
        
        # 유저-세션별 구매 아이템 (co-occurrence 계산용)
        print("Building item co-occurrence...")
        session_items = purchases.groupby(['user_id', 'user_session'])['item_id'].apply(list)
        
        for items in tqdm(session_items):
            items = list(set(items))  # 중복 제거
            for i, item1 in enumerate(items):
                for item2 in items[i+1:]:
                    self.item_cooccur[item1][item2] += 1
                    self.item_cooccur[item2][item1] += 1
        
        # 유저별 최근 interaction 아이템 (시간순 정렬)
        print("Building user recent items...")
        train_sorted = train.sort_values('event_time')
        
        for user_id, group in tqdm(train_sorted.groupby('user_id')):
            # 최근 20개 아이템 (가중치 포함)
            recent = group.tail(50)
            items_weights = recent.groupby('item_id')['weight'].sum()
            self.user_recent_items[user_id] = items_weights.sort_values(ascending=False).index.tolist()[:20]
        
        # 글로벌 인기 아이템
        item_scores = train.groupby('item_id')['weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()
        
    def recommend(self, user_id, n=10):
        recs = []
        seen = set()
        
        if user_id in self.user_recent_items:
            recent_items = self.user_recent_items[user_id]
            
            # 최근 아이템과 co-occur하는 아이템 추천
            cooccur_scores = defaultdict(float)
            for i, item in enumerate(recent_items):
                weight = 1.0 / (i + 1)  # 최근 아이템일수록 높은 가중치
                if item in self.item_cooccur:
                    for co_item, count in self.item_cooccur[item].items():
                        if count >= self.cooccur_min_count:
                            cooccur_scores[co_item] += count * weight
            
            # 스코어 순 정렬
            sorted_cooccur = sorted(cooccur_scores.items(), key=lambda x: x[1], reverse=True)
            for item, _ in sorted_cooccur:
                if item not in seen:
                    recs.append(item)
                    seen.add(item)
                if len(recs) >= n:
                    break
            
            # 부족하면 최근 본 아이템 추가
            for item in recent_items:
                if item not in seen:
                    recs.append(item)
                    seen.add(item)
                if len(recs) >= n:
                    break
        
        # 부족하면 인기 아이템으로 채움
        for item in self.popular_items:
            if item not in seen:
                recs.append(item)
                seen.add(item)
            if len(recs) >= n:
                break
                
        return recs[:n]


# ============================================
# 4. 앙상블 모델
# ============================================

class EnsembleBaseline:
    """
    여러 베이스라인 앙상블
    - 각 모델의 추천에 가중치 부여
    """
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        
    def fit(self, train):
        for model in self.models:
            model.fit(train)
            
    def recommend(self, user_id, n=10):
        item_scores = defaultdict(float)
        
        for model, weight in zip(self.models, self.weights):
            recs = model.recommend(user_id, n=n*2)  # 더 많이 뽑아서 스코어 계산
            for rank, item in enumerate(recs):
                # 순위 기반 스코어 (1/rank)
                item_scores[item] += weight / (rank + 1)
        
        # 스코어 순 정렬
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:n]]


# ============================================
# 5. 로컬 검증
# ============================================

def calculate_ndcg(y_true, y_pred, k=10):
    """NDCG@K 계산 (binary relevance)"""
    dcg = 0.0
    for i, item in enumerate(y_pred[:k]):
        if item in y_true:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # IDCG: 최대 가능한 DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def local_validation(train, model, val_days=7):
    """
    로컬 검증
    - train 마지막 val_days일을 validation으로 사용
    """
    print(f"\nRunning local validation (last {val_days} days)...")
    
    max_date = train['event_time'].max()
    val_start = max_date - pd.Timedelta(days=val_days)
    
    # Train/Val 분리
    train_data = train[train['event_time'] < val_start].copy()
    val_data = train[train['event_time'] >= val_start].copy()
    
    # Validation: purchase만 ground truth
    val_purchases = val_data[val_data['event_type'] == 'purchase']
    val_gt = val_purchases.groupby('user_id')['item_id'].apply(set).to_dict()
    
    print(f"Train period: {train_data['event_time'].min()} ~ {train_data['event_time'].max()}")
    print(f"Val period: {val_data['event_time'].min()} ~ {val_data['event_time'].max()}")
    print(f"Val users with purchases: {len(val_gt)}")
    
    # 모델 학습
    model.fit(train_data)
    
    # 평가
    ndcg_scores = []
    for user_id, true_items in tqdm(val_gt.items(), desc="Evaluating"):
        pred_items = model.recommend(user_id, n=10)
        ndcg = calculate_ndcg(true_items, pred_items, k=10)
        ndcg_scores.append(ndcg)
    
    mean_ndcg = np.mean(ndcg_scores)
    print(f"\nLocal NDCG@10: {mean_ndcg:.6f}")
    
    return mean_ndcg


# ============================================
# 6. 제출 파일 생성
# ============================================

def generate_submission(model, submission, output_path='submission.csv'):
    """제출 파일 생성"""
    print("\nGenerating submission...")
    
    # 유저 리스트 추출
    users = submission['user_id'].unique()
    print(f"Total users to predict: {len(users)}")
    
    # 추천 생성
    results = []
    for user_id in tqdm(users, desc="Generating recommendations"):
        recs = model.recommend(user_id, n=10)
        for item_id in recs:
            results.append({'user_id': user_id, 'item_id': item_id})
    
    # 데이터프레임 생성
    submission_df = pd.DataFrame(results)
    
    # 검증
    assert len(submission_df) == len(submission), \
        f"Submission length mismatch: {len(submission_df)} vs {len(submission)}"
    
    # 유저당 10개 아이템 확인
    items_per_user = submission_df.groupby('user_id').size()
    assert (items_per_user == 10).all(), "Each user must have exactly 10 items"
    
    # 중복 확인
    duplicates = submission_df.groupby('user_id')['item_id'].apply(lambda x: x.duplicated().any())
    assert not duplicates.any(), "Duplicate items found for some users"
    
    # 저장
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Shape: {submission_df.shape}")
    
    return submission_df


# ============================================
# 7. 메인 실행
# ============================================

def main():
    # 데이터 로드
    train, submission = load_data('data/train.parquet', 'data/sample_submission.csv')
    
    # 전처리
    train = preprocess(train)
    
    # ========== 모델 선택 ==========
    
    # Option 1: 단순 인기도 기반
    # model = PopularityBaseline(time_decay=True, decay_days=30)
    
    # Option 2: 유저 히스토리 기반
    # model = UserHistoryBaseline(time_decay=True, decay_days=14)
    
    # Option 3: 카테고리 인지
    # model = CategoryAwareBaseline()
    
    # Option 4: 최근 구매 + Co-occurrence
    # model = RecentPurchaseBaseline(cooccur_min_count=3)
    
    # Option 5: 앙상블 (추천)
    model = EnsembleBaseline(
        models=[
            UserHistoryBaseline(time_decay=True, decay_days=14),
            RecentPurchaseBaseline(cooccur_min_count=3),
            PopularityBaseline(time_decay=True, decay_days=30),
        ],
        weights=[2.0, 1.5, 1.0]
    )
    
    # ========== 로컬 검증 ==========
    # local_validation(train, model, val_days=7)
    
    # ========== 전체 학습 & 제출 ==========
    model.fit(train)
    submission_df = generate_submission(model, submission, 'submission.csv')
    
    print("\nDone!")


if __name__ == "__main__":
    main()