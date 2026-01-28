import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

# ============================================
# 1. 데이터 로드 및 전처리
# ============================================

def load_data(train_path='data/train.parquet', submission_path='data/sample_submission.csv'):
    print("Loading data...")
    train = pd.read_parquet(train_path)
    submission = pd.read_csv(submission_path)
    return train, submission

def preprocess(train, 
               event_weights={'view': 1, 'cart': 3, 'purchase': 12}, 
               use_recent_days=90): # [수정] 90일치만 사용하여 메모리 절약
    
    train = train.copy()
    train['event_time'] = pd.to_datetime(train['event_time'])
    
    # 최근 N일만 사용 (메모리 절약의 핵심)
    if use_recent_days:
        max_date = train['event_time'].max()
        cutoff = max_date - pd.Timedelta(days=use_recent_days)
        train = train[train['event_time'] >= cutoff]
        print(f"Using last {use_recent_days} days: {len(train):,} rows (Mem Safe Mode)")
        
    train['weight'] = train['event_type'].map(event_weights).fillna(1)
    return train

# ============================================
# 2. 모델 정의 (메모리 최적화)
# ============================================

class EASE:
    def __init__(self, regularization=500):
        self.regularization = regularization
        self.B = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.user_history_matrix = None
        
    def fit(self, train):
        print(f"Fitting EASE (reg={self.regularization})...")
        
        unique_users = train['user_id'].unique()
        unique_items = train['item_id'].unique()
        
        print(f" - Users: {len(unique_users):,}, Items: {len(unique_items):,}")
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {i: idx for idx, i in enumerate(unique_items)}
        self.reverse_item_map = {idx: i for i, idx in self.item_map.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        row = train['user_id'].map(self.user_map)
        col = train['item_id'].map(self.item_map)
        data = train['final_weight'].values
        
        # [수정] float32로 변환하여 메모리 절약
        X = csr_matrix((data.astype(np.float32), (row, col)), shape=(n_users, n_items))
        self.user_history_matrix = X
        
        print("  - Calculating Gram matrix...")
        # G = X.T @ X
        G = X.T.dot(X).toarray().astype(np.float32) # [수정] float32 유지
        
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.regularization
        
        print("  - Inverting matrix...")
        P = np.linalg.inv(G)
        
        # 메모리 정리
        del G
        gc.collect()
        
        B = P / (-np.diag(P))
        B[diag_indices] = 0
        
        self.B = B.astype(np.float32) # [수정] 결과도 float32
        print("  - EASE Fitting Done.")

class PopularityBaseline:
    def __init__(self):
        self.popular_items = []
        
    def fit(self, train):
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()

# ============================================
# 3. 앙상블 클래스
# ============================================

class BatchEnsemble:
    def __init__(self):
        self.ease = None
        self.popularity = None
        
    def fit(self, train, decay_days=14):
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        # EASE 학습
        self.ease = EASE(regularization=500)
        self.ease.fit(train)
        
        # Popularity 학습
        self.popularity = PopularityBaseline()
        self.popularity.fit(train)
        
    def predict_batch(self, user_ids, n=10, batch_size=500): # [수정] 배치 사이즈 축소 (2000->500)
        results = {}
        reverse_item_map = self.ease.reverse_item_map
        
        # 가중치 (EASE + History 보정)
        # 메모리 문제로 별도 History 모델 객체 생성 대신 EASE Matrix 재활용 (가장 효율적)
        
        for i in tqdm(range(0, len(user_ids), batch_size), desc="Batch Inference"):
            batch_users = user_ids[i : i + batch_size]
            
            # 매핑 확인
            batch_indices = []
            valid_batch_pos = [] # 유효한 유저의 batch 내 위치 (0~batch_size)
            
            for idx, u in enumerate(batch_users):
                if u in self.ease.user_map:
                    batch_indices.append(self.ease.user_map[u])
                    valid_batch_pos.append(idx)
            
            if not batch_indices:
                for u in batch_users:
                    results[u] = self.popularity.popular_items[:n]
                continue

            # Matrix Multiplication (Batch x Items)
            user_vectors = self.ease.user_history_matrix[batch_indices]
            
            # 1. EASE 점수 (협업 필터링)
            scores = user_vectors.dot(self.ease.B) * 1.8 
            
            # 2. History 점수 (재소비 부스팅)
            # 유저가 이미 본 아이템 점수를 대폭 올려줌
            # sparse matrix인 user_vectors를 활용해 본 아이템 위치를 찾음
            rows, cols = user_vectors.nonzero()
            
            # rows는 0부터 시작하는 상대 인덱스, cols는 아이템 인덱스
            # user_vectors.data는 해당 view/purchase의 가중치
            
            # 벡터화된 덧셈 (반복문 없이 한방에)
            # scores[row, col] += weight * boosting_factor
            # sparse matrix data access는 1차원 배열 형태임에 주의
            
            # 간단하게 row별로 순회하며 더하기 (안전함)
            for r, c, val in zip(rows, cols, user_vectors.data):
                scores[r, c] += val * 2.5 # History 가중치
            
            # 3. Top-K 추출
            if scores.shape[1] > n:
                # 상위 N개만 정렬
                top_k_part = np.argpartition(-scores, n, axis=1)[:, :n]
                row_indices = np.arange(scores.shape[0])[:, None]
                top_scores = scores[row_indices, top_k_part]
                sorted_indices = np.argsort(-top_scores, axis=1)
                final_indices = top_k_part[row_indices, sorted_indices]
            else:
                final_indices = np.argsort(-scores, axis=1)[:, :n]
            
            # 4. 결과 저장
            for local_idx, real_pos in enumerate(valid_batch_pos):
                u_id = batch_users[real_pos]
                recs = [reverse_item_map[ix] for ix in final_indices[local_idx]]
                results[u_id] = recs
        
        # Cold User 처리
        for u in user_ids:
            if u not in results:
                results[u] = self.popularity.popular_items[:n]
                
        return results

# ============================================
# 4. 메인 실행
# ============================================

def main():
    gc.collect() # 시작 전 메모리 청소
    
    train, submission = load_data()
    
    # [중요] 90일치만 사용 (메모리 부족하면 60일로 줄이세요)
    train = preprocess(train, use_recent_days=90) 
    
    model = BatchEnsemble()
    model.fit(train, decay_days=14)
    
    print(f"\nGeneraring submission...")
    users = submission['user_id'].unique().tolist()
    
    # 배치 사이즈 500 (안전함)
    recommendations = model.predict_batch(users, n=10, batch_size=500)
    
    results = []
    for user_id, recs in recommendations.items():
        # 부족하면 인기 아이템 채움
        if len(recs) < 10:
             seen = set(recs)
             for item in model.popularity.popular_items:
                 if item not in seen:
                     recs.append(item)
                 if len(recs) >= 10:
                     break
        
        for item_id in recs[:10]:
            results.append({'user_id': user_id, 'item_id': item_id})
            
    output = pd.DataFrame(results)
    output.to_csv('submission_optimized.csv', index=False)
    print("\nSaved to: submission_optimized.csv")

if __name__ == "__main__":
    main()