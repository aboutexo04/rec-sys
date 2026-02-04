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
               # [복구] View=1 (클릭 정보 살림), Purchase=20 (확실한 선호)
               # 0.1511 점수를 냈던 황금 비율로 복귀
               event_weights={'view': 1, 'cart': 5, 'purchase': 20}, 
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
# 2. 모델 정의 (EASE + High Reg)
# ============================================

class EASE:
    def __init__(self, regularization=1500): # [튜닝] 1000 -> 1500 (고가중치 제어용 강한 규제)
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
        
        X = csr_matrix((data.astype(np.float32), (row, col)), shape=(n_users, n_items))
        
        # L2 Normalization (필수)
        print("  - Applying L2 Normalization...")
        X = normalize(X, norm='l2', axis=1, copy=False)
        self.user_history_matrix = X
        
        print("  - Calculating Gram matrix...")
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
        
    def fit(self, train, decay_days=28): # [튜닝] 21일 -> 28일 (한 달 주기 반영)
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        self.ease = EASE(regularization=1500) # 정의한 1500 적용
        self.ease.fit(train)
        
        self.popularity = PopularityBaseline()
        self.popularity.fit(train)
        
    def predict_batch(self, user_ids, n=10, batch_size=500):
        results = {}
        reverse_item_map = self.ease.reverse_item_map
        
        # [튜닝] 0.1511 점수 비율로 회귀하되, History를 아주 살짝만 더 챙김
        W_EASE = 1.5
        W_HIST = 0.6  # (0.5 -> 0.6) 미세 조정
        
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
            
            # EASE Score
            scores = user_vectors.dot(self.ease.B) * W_EASE 
            
            # History Boosting
            rows, cols = user_vectors.nonzero()
            for r, c, val in zip(rows, cols, user_vectors.data):
                scores[r, c] += (val * W_HIST)
            
            # Top-K
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
    
    # 90일 데이터 (안전)
    train = preprocess(train, use_recent_days=90) 
    
    model = BatchEnsemble()
    # 28일 반감기 (월간 패턴)
    model.fit(train, decay_days=28) 
    
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
    # 파일명: 안정성 강화 버전
    output.to_csv('submission_monthly_cycle.csv', index=False)
    print("\nSaved to: submission_monthly_cycle.csv")

if __name__ == "__main__":
    main()