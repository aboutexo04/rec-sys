import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
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
               # [튜닝] Cart(5) 강화: 장바구니는 구매만큼 강력한 힌트입니다.
               event_weights={'view': 1, 'cart': 5, 'purchase': 12}, 
               use_recent_days=90):
    
    train = train.copy()
    train['event_time'] = pd.to_datetime(train['event_time'])
    
    if use_recent_days:
        max_date = train['event_time'].max()
        cutoff = max_date - pd.Timedelta(days=use_recent_days)
        train = train[train['event_time'] >= cutoff]
        print(f"Using last {use_recent_days} days: {len(train):,} rows")
        
    train['weight'] = train['event_type'].map(event_weights).fillna(1)
    return train

# ============================================
# 2. 모델 정의 (L2 Normalization 추가)
# ============================================

class EASE:
    def __init__(self, regularization=700): # [튜닝] 500 -> 700 (노이즈 제거 강화)
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
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {i: idx for idx, i in enumerate(unique_items)}
        self.reverse_item_map = {idx: i for i, idx in self.item_map.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        row = train['user_id'].map(self.user_map)
        col = train['item_id'].map(self.item_map)
        data = train['final_weight'].values
        
        # 1. Matrix 생성 (float32)
        X = csr_matrix((data.astype(np.float32), (row, col)), shape=(n_users, n_items))
        
        # [핵심 추가] L2 Normalization
        # 유저별 활동량 편차를 줄여 패턴 자체에 집중하게 만듦
        # copy=False로 메모리 절약
        print("  - Applying L2 Normalization...")
        X = normalize(X, norm='l2', axis=1, copy=False)
        self.user_history_matrix = X
        
        print("  - Calculating Gram matrix...")
        # G = X.T @ X
        G = X.T.dot(X).toarray().astype(np.float32)
        
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.regularization
        
        print("  - Inverting matrix...")
        P = np.linalg.inv(G)
        
        del G
        gc.collect()
        
        B = P / (-np.diag(P))
        B[diag_indices] = 0
        
        self.B = B.astype(np.float32)
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
        
    def fit(self, train, decay_days=21): # [튜닝] 21일 (3주) - 가장 밸런스 좋은 구간
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        self.ease = EASE(regularization=700) # 위에서 정의한 700 적용
        self.ease.fit(train)
        
        self.popularity = PopularityBaseline()
        self.popularity.fit(train)
        
    def predict_batch(self, user_ids, n=10, batch_size=500):
        results = {}
        reverse_item_map = self.ease.reverse_item_map
        
        # [가중치 전략]
        # L2 Norm을 했기 때문에 EASE 점수 스케일이 달라졌을 수 있음
        # 하지만 상대적 순위가 중요하므로 비율은 유지하되, History를 조금 더 믿음
        W_EASE = 1.5
        W_HIST = 0.5 
        
        for i in tqdm(range(0, len(user_ids), batch_size), desc="Batch Inference"):
            batch_users = user_ids[i : i + batch_size]
            
            batch_indices = []
            valid_batch_pos = []
            
            for idx, u in enumerate(batch_users):
                if u in self.ease.user_map:
                    batch_indices.append(self.ease.user_map[u])
                    valid_batch_pos.append(idx)
            
            if not batch_indices:
                for u in batch_users:
                    results[u] = self.popularity.popular_items[:n]
                continue

            user_vectors = self.ease.user_history_matrix[batch_indices]
            
            # 1. EASE Score
            scores = user_vectors.dot(self.ease.B) * W_EASE 
            
            # 2. History Boosting (L2 Norm 때문에 값 작아짐 -> 가중치 보정 필요 없으나 안전하게 더함)
            rows, cols = user_vectors.nonzero()
            
            # [테크닉] 반복문 대신 벡터 연산으로 속도 향상 & 히스토리 강조
            # user_vectors의 값은 이미 normalized 되어 있음. 
            # 여기에 추가 가중치를 더해 "본 건 무조건 상위권"으로 올림
            for r, c, val in zip(rows, cols, user_vectors.data):
                scores[r, c] += (val * W_HIST) + 0.5 # 0.5는 기본 보너스 점수
            
            # 3. Top-K
            if scores.shape[1] > n:
                top_k_part = np.argpartition(-scores, n, axis=1)[:, :n]
                row_indices = np.arange(scores.shape[0])[:, None]
                top_scores = scores[row_indices, top_k_part]
                sorted_indices = np.argsort(-top_scores, axis=1)
                final_indices = top_k_part[row_indices, sorted_indices]
            else:
                final_indices = np.argsort(-scores, axis=1)[:, :n]
            
            for local_idx, real_pos in enumerate(valid_batch_pos):
                u_id = batch_users[real_pos]
                recs = [reverse_item_map[ix] for ix in final_indices[local_idx]]
                results[u_id] = recs
        
        for u in user_ids:
            if u not in results:
                results[u] = self.popularity.popular_items[:n]
                
        return results

# ============================================
# 4. 메인 실행
# ============================================

def main():
    gc.collect()
    train, submission = load_data()
    
    # 90일치, 90일이 버거우면 60일로
    train = preprocess(train, use_recent_days=90) 
    
    model = BatchEnsemble()
    # 반감기 21일 (3주)
    model.fit(train, decay_days=21) 
    
    print(f"\nGeneraring submission...")
    users = submission['user_id'].unique().tolist()
    
    recommendations = model.predict_batch(users, n=10, batch_size=500)
    
    results = []
    for user_id, recs in recommendations.items():
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
    output.to_csv('submission_l2_norm.csv', index=False)
    print("\nSaved to: submission_l2_norm.csv")

if __name__ == "__main__":
    main()