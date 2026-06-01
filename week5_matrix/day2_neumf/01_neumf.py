"""
=============================================================
Day 2: NeuMF (Neural Matrix Factorization) - 신경망 기반 추천
=============================================================

목표: PyTorch로 신경망 추천 모델을 직접 짜고 학습시켜본다.
방법: GMF 경로 + MLP 경로 결합 → 매칭 확률 예측.

실행 방법:
1. 터미널에서: python Day2_NeuMF.py
2. 또는 VS Code에서 ▶ 재생 버튼 클릭

PyTorch 설치 확인:
  python -c "import torch; print(torch.__version__)"
  설치: pip install torch
"""

import sys
import numpy as np
import torch
import torch.nn as nn

# 윈도우 터미널(cp949) 한글 깨짐 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================
# 1. 데이터 준비 - Day 1보다 큰 데이터로
# =============================================================
# 10명 멘티 × 8명 멘토. 1=만족, 0=불만족 또는 안 본 멘토
# 실제 서비스에서는 매칭 후 만족도 체크한 데이터에 해당

# label=1: 매칭 후 만족 (positive)
# label=0: 매칭 후 불만족 OR 매칭 안 함 (negative)
# 빈 칸은 'None' 처리 - 학습에 안 씀 (Day 1처럼 0과 헷갈리지 않게)
N = None

R = np.array([
    # m0  m1  m2  m3  m4  m5  m6  m7
    [1,   1,   N,   0,   N,   1,   N,   N],   # 멘티 0
    [1,   N,   0,   N,   1,   N,   N,   1],   # 멘티 1
    [N,   1,   1,   N,   N,   0,   N,   1],   # 멘티 2
    [0,   N,   N,   1,   N,   N,   1,   N],   # 멘티 3
    [N,   1,   N,   N,   1,   1,   N,   N],   # 멘티 4
    [1,   1,   N,   N,   N,   N,   0,   1],   # 멘티 5
    [N,   N,   1,   1,   N,   0,   N,   N],   # 멘티 6
    [0,   N,   N,   N,   1,   N,   1,   N],   # 멘티 7
    [N,   1,   N,   0,   N,   1,   N,   1],   # 멘티 8
    [1,   N,   0,   N,   N,   N,   1,   N],   # 멘티 9
])

n_users, n_items = R.shape

# 알려진 칸만 (user_id, item_id, label) 형태로 변환
known_data = []
for u in range(n_users):
    for i in range(n_items):
        if R[u, i] is not None:
            known_data.append((u, i, int(R[u, i])))

print(f"멘티 수: {n_users}, 멘토 수: {n_items}")
print(f"알려진 매칭 데이터: {len(known_data)}개")
print(f"  positive(만족): {sum(1 for _, _, lbl in known_data if lbl==1)}개")
print(f"  negative(불만): {sum(1 for _, _, lbl in known_data if lbl==0)}개")


# =============================================================
# 2. 하이퍼파라미터
# =============================================================
EMBED_DIM_GMF = 8   # GMF 경로 임베딩 차원
EMBED_DIM_MLP = 8   # MLP 경로 임베딩 차원
MLP_HIDDEN = [16, 8]   # MLP 내부 hidden layer 크기들
LEARNING_RATE = 0.01
N_EPOCHS = 500


# =============================================================
# 3. NeuMF 모델 정의 - PyTorch nn.Module 상속
# =============================================================
class NeuMF(nn.Module):
    """
    NeuMF = GMF 경로 + MLP 경로 결합 추천 모델

    구조:
      [멘티ID] → 멘티_GMF_임베딩, 멘티_MLP_임베딩
      [멘토ID] → 멘토_GMF_임베딩, 멘토_MLP_임베딩

      GMF 경로:  멘티_GMF ⊙ 멘토_GMF              → 벡터
      MLP 경로:  [멘티_MLP, 멘토_MLP] → MLP        → 벡터

      두 벡터 concat → Linear → Sigmoid → 매칭 확률
    """

    def __init__(self, n_users, n_items, gmf_dim, mlp_dim, mlp_hidden):
        super().__init__()

        # 임베딩 층: ID → 벡터로 변환하는 lookup table
        # nn.Embedding(num_embeddings, embedding_dim)는 학습 가능한 행렬
        self.user_emb_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_emb_gmf = nn.Embedding(n_items, gmf_dim)
        self.user_emb_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_dim)

        # MLP 경로: Linear → ReLU 반복
        # 입력은 두 MLP 임베딩 concat → 차원 mlp_dim * 2
        layers = []
        input_dim = mlp_dim * 2
        for hidden_dim in mlp_hidden:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        # nn.Sequential = 여러 layer를 순서대로 묶는 컨테이너
        self.mlp = nn.Sequential(*layers)

        # 최종 출력 layer: GMF 결과 + MLP 결과 concat → 점수 1개
        # GMF 결과 차원: gmf_dim
        # MLP 결과 차원: mlp_hidden[-1] (마지막 hidden 크기)
        self.final = nn.Linear(gmf_dim + mlp_hidden[-1], 1)

    def forward(self, user_ids, item_ids):
        """
        순방향 계산: ID들 → 매칭 확률

        Args:
            user_ids: shape (batch,) 멘티 ID들
            item_ids: shape (batch,) 멘토 ID들

        Returns:
            shape (batch,) 0~1 사이 매칭 확률
        """
        # 임베딩 조회
        u_gmf = self.user_emb_gmf(user_ids)   # (batch, gmf_dim)
        i_gmf = self.item_emb_gmf(item_ids)   # (batch, gmf_dim)
        u_mlp = self.user_emb_mlp(user_ids)   # (batch, mlp_dim)
        i_mlp = self.item_emb_mlp(item_ids)   # (batch, mlp_dim)

        # GMF 경로: element-wise multiplication
        gmf_out = u_gmf * i_gmf   # (batch, gmf_dim)

        # MLP 경로: concat → MLP 통과
        mlp_input = torch.cat([u_mlp, i_mlp], dim=1)   # (batch, mlp_dim*2)
        mlp_out = self.mlp(mlp_input)   # (batch, mlp_hidden[-1])

        # 두 경로 결과 concat → 마지막 Linear → sigmoid
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        logit = self.final(combined).squeeze(-1)   # (batch,)
        prob = torch.sigmoid(logit)   # 0~1

        return prob


# =============================================================
# 4. 데이터를 PyTorch 텐서로
# =============================================================
user_ids = torch.tensor([u for u, _, _ in known_data], dtype=torch.long)
item_ids = torch.tensor([i for _, i, _ in known_data], dtype=torch.long)
labels = torch.tensor([lbl for _, _, lbl in known_data], dtype=torch.float)


# =============================================================
# 5. 모델, 손실함수, 옵티마이저 준비
# =============================================================
torch.manual_seed(42)   # 재현 가능하게 (Day 1의 np.random.seed와 같은 역할)

model = NeuMF(
    n_users=n_users,
    n_items=n_items,
    gmf_dim=EMBED_DIM_GMF,
    mlp_dim=EMBED_DIM_MLP,
    mlp_hidden=MLP_HIDDEN,
)

# BCE Loss (이진 교차 엔트로피)
# 0~1 확률을 0/1 정답과 비교하는 손실
loss_fn = nn.BCELoss()

# Adam optimizer - SGD 개선판, 대부분 잘 작동
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n모델 구조:")
print(model)


# =============================================================
# 6. 학습 루프
# =============================================================
print(f"\n학습 시작 (총 {N_EPOCHS} epoch)")
print("-" * 50)

for epoch in range(N_EPOCHS):
    # 모드 전환 (학습용 - dropout 등 활성화. 여기선 별 영향 없지만 관례)
    model.train()

    # 순방향: 예측 확률 계산
    pred_prob = model(user_ids, item_ids)

    # 손실 계산
    loss = loss_fn(pred_prob, labels)

    # 역방향 + 가중치 업데이트
    optimizer.zero_grad()   # 이전 gradient 청소
    loss.backward()          # 자동 미분 (PyTorch의 핵심 기능)
    optimizer.step()         # 가중치 업데이트

    # 50 epoch마다 출력
    if (epoch + 1) % 50 == 0:
        # 정확도도 계산: 0.5 기준으로 분류 시 맞춘 비율
        preds_binary = (pred_prob >= 0.5).float()
        acc = (preds_binary == labels).float().mean().item()
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {acc:.3f}")


# =============================================================
# 7. 학습 완료 - 전체 행렬 예측
# =============================================================
print("\n" + "=" * 50)
print("학습 완료. 전체 예측 행렬 생성")
print("=" * 50)

model.eval()   # 평가 모드 (gradient 계산 안 함, 빠름)
with torch.no_grad():
    all_user_ids = torch.tensor(
        [u for u in range(n_users) for _ in range(n_items)],
        dtype=torch.long,
    )
    all_item_ids = torch.tensor(
        [i for _ in range(n_users) for i in range(n_items)],
        dtype=torch.long,
    )
    all_probs = model(all_user_ids, all_item_ids).reshape(n_users, n_items).numpy()

print(f"\n예측 확률 행렬 (모든 멘티 × 모든 멘토):")
print(np.round(all_probs, 2))


# =============================================================
# 8. 안 본 멘토 Top-3 추천
# =============================================================
print("\n" + "=" * 50)
print("각 멘티에게 안 본 멘토 Top-3 추천")
print("=" * 50)

for u in range(n_users):
    # 안 본 멘토(None 값) 찾기
    unseen = [i for i in range(n_items) if R[u, i] is None]
    if not unseen:
        print(f"  멘티 {u}: 모든 멘토와 매칭 경험 있음")
        continue

    # 안 본 멘토 중 예측 확률 높은 순으로
    scores = [(i, all_probs[u, i]) for i in unseen]
    scores.sort(key=lambda x: -x[1])
    top3 = scores[:3]

    rec_str = ", ".join([f"멘토{i}({p:.2f})" for i, p in top3])
    print(f"  멘티 {u}: {rec_str}")


# =============================================================
# 9. Day 1 vs Day 2 차이 살펴보기
# =============================================================
print("\n" + "=" * 50)
print("Day 1 vs Day 2 차이")
print("=" * 50)
print("""
Day 1 (Matrix Factorization, NumPy):
  - 출력: 평점 예측 (1~5 점수)
  - 학습: 평점 오차 줄이기 (회귀)
  - 한계: 선형 관계만 잡음

Day 2 (NeuMF, PyTorch):
  - 출력: 매칭 확률 (0~1)
  - 학습: 만족(1)/불만족(0) 분류
  - 강점: GMF(선형) + MLP(비선형) 동시 활용
  - 의의: 사용자-아이템 간 복잡한 상호작용 학습 가능
""")
