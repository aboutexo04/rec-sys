import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings('ignore')

# ============================================
# 1. 데이터 로드 및 전처리 (멘토 피드백 반영)
# ============================================

def load_data(train_path='data/train.parquet', submission_path='data/sample_submission.csv'):
    print("Loading data...")
    train = pd.read_parquet(train_path)
    submission = pd.read_csv(submission_path)
    return train, submission

def preprocess(train, 
               # [멘토 피드백 1 & 2] 희소 이벤트(Cart/Purchase) 강조, 노이즈(View) 축소
               # View는 빈도가 너무 높고 신호가 약하므로 0.5로 대폭 낮춤
               # Cart는 구매 직전의 가장 강력한 신호이므로 10으로 격상
               event_weights={'view': 0.5, 'cart': 10, 'purchase': 20}, 
               use_recent_days=45): # [멘토 피드백 3] 데이터 축소: 최근 45일(1월 중순~2월 말) 집중
    
    train = train.copy()
    train['event_time'] = pd.to_datetime(train['event_time'])
    
    if use_recent_days:
        max_date = train['event_time'].max()
        cutoff = max_date - pd.Timedelta(days=use_recent_days)
        train = train[train['event_time'] >= cutoff]
        print(f"Using last {use_recent_days} days (Data Reduction): {len(train):,} rows")
        
    train['weight'] = train['event_type'].map(event_weights).fillna(1)
    return train

# ============================================
# 2. 모델 정의 (EASE + L2 Norm)
# ============================================

class EASE:
    def __init__(self, regularization=1000): # 데이터가 줄었으므로 규제는 1000 유지 (과적합 방지)
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
        
        # Matrix 생성
        X = csr_matrix((data.astype(np.float32), (row, col)), shape=(n_users, n_items))
        
        # [핵심] L2 Normalization 유지 (가중치 폭발 방지)
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
        # 인기 아이템도 "진짜 산 것" 위주로 (Purchase/Cart 가중치 반영된 weight 사용)
        item_scores = train.groupby('item_id')['final_weight'].sum()
        self.popular_items = item_scores.sort_values(ascending=False).index.tolist()

# ============================================
# 3. 앙상블 클래스
# ============================================

class BatchEnsemble:
    def __init__(self):
        self.ease = None
        self.popularity = None
        
    def fit(self, train, decay_days=10): # [멘토 피드백] 2월 데이터 집중 -> 반감기 10일로 단축
        train = train.copy()
        max_time = train['event_time'].max()
        train['days_ago'] = (max_time - train['event_time']).dt.total_seconds() / 86400
        train['time_weight'] = np.exp(-train['days_ago'] / decay_days)
        train['final_weight'] = train['weight'] * train['time_weight']
        
        self.ease = EASE(regularization=1000)
        self.ease.fit(train)
        
        self.popularity = PopularityBaseline()
        self.popularity.fit(train)
        
    def predict_batch(self, user_ids, n=10, batch_size=500):
        results = {}
        reverse_item_map = self.ease.reverse_item_map
        
        # [전략] EASE에 거의 모든 것을 맡김
        # History Boost는 아주 살짝만 (L2 Norm으로 이미 패턴이 잡혀있음)
        W_EASE = 1.0
        W_HIST = 0.2
        
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
    
    # [핵심] 멘토 조언: "데이터 축소" -> 최근 45일만 사용
    train = preprocess(train, use_recent_days=45) 
    
    model = BatchEnsemble()
    # [핵심] 멘토 조언: "2월 가중치" -> Decay 10일 (빠르게 감소)
    model.fit(train, decay_days=10) 
    
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
    # 멘토 피드백 반영 버전
    output.to_csv('submission_mentor_feedback.csv', index=False)
    print("\nSaved to: submission_mentor_feedback.csv")

if __name__ == "__main__":
    main()