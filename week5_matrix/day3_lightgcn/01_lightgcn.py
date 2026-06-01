"""
=============================================================
Day 3: LightGCN - 그래프 위에서 임베딩 전파하기
=============================================================

목표: 멘티-멘토 양분 그래프에 임베딩을 K번 전파시켜 추천.
방법: 정규화된 인접 행렬을 K번 곱해서 K-hop 이웃 정보 흡수.

실행 방법:
1. 터미널에서: python 01_lightgcn.py
2. 또는 VS Code에서 ▶ 재생 버튼 클릭
"""

import sys
import numpy as np
import torch
import torch.nn as nn

# 윈도우 한글 깨짐 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================
# 1. 데이터 - 멘티-멘토 매칭 (Day 2보다 약간 크게)
# =============================================================
# 멘티 10명, 멘토 8명
# interactions: (mentee_id, mentor_id) 양성(positive) 쌍 리스트

n_users = 10
n_items = 8

# 매칭이 일어난 (멘티, 멘토) 쌍 — 즉 양성 신호
interactions = [
    (0, 0), (0, 1), (0, 5),
    (1, 1), (1, 3), (1, 7),
    (2, 1), (2, 2), (2, 7),
    (3, 3), (3, 6),
    (4, 1), (4, 4),
    (5, 0), (5, 1), (5, 7),
    (6, 2), (6, 3),
    (7, 4), (7, 6),
    (8, 1), (8, 5), (8, 7),
    (9, 0), (9, 6),
]

print(f"멘티 수: {n_users}, 멘토 수: {n_items}")
print(f"양성 매칭 수: {len(interactions)}")


# =============================================================
# 2. 인접 행렬 만들기 + 정규화
# =============================================================
def build_normalized_adj(interactions, n_users, n_items):
    """양분 그래프의 정규화된 인접 행렬을 만든다."""
    n_total = n_users + n_items

    # 1) 인접 행렬 A 만들기 (n_total × n_total)
    A = torch.zeros(n_total, n_total)
    for u, i in interactions:
        # 멘티 u ↔ 멘토 i 양방향 엣지
        A[u, n_users + i] = 1.0
        A[n_users + i, u] = 1.0

    # 2) 차수 D 계산 (각 노드의 이웃 수)
    D = A.sum(dim=1)  # 각 행의 합 = 그 노드의 이웃 수

    # 3) D^(-1/2) 계산 (이웃 없는 노드는 0)
    D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))

    # 4) 정규화: A_norm = D^(-1/2) @ A @ D^(-1/2)
    # 효율: D_inv_sqrt를 양쪽에 element-wise 곱
    A_norm = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)

    return A_norm


A_norm = build_normalized_adj(interactions, n_users, n_items)
print(f"\n정규화된 인접 행렬 크기: {A_norm.shape}")
print(f"(처음 5×5만 출력)")
print(np.round(A_norm[:5, :5].numpy(), 3))


# =============================================================
# 3. LightGCN 모델 정의
# =============================================================
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_layers, A_norm):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        # 인접 행렬은 학습 안 함 - 그래프 구조 자체는 고정
        self.register_buffer("A_norm", A_norm)

        # 초기 임베딩 (e^0). 멘티 + 멘토 합쳐 한 행렬
        self.embedding = nn.Embedding(n_users + n_items, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self):
        """K번 전파해서 모든 layer의 임베딩을 평균."""
        all_emb = self.embedding.weight  # (n_total, embed_dim), e^0
        embs = [all_emb]

        # K번 propagation: e^(k+1) = A_norm @ e^k
        for _ in range(self.n_layers):
            all_emb = self.A_norm @ all_emb  # 핵심 한 줄!
            embs.append(all_emb)

        # 모든 layer 평균 (residual)
        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        return final_emb

    def forward(self, user_ids, item_ids):
        """매칭 확률 예측"""
        final_emb = self.propagate()
        u_emb = final_emb[user_ids]                       # 멘티 임베딩
        i_emb = final_emb[self.n_users + item_ids]        # 멘토 임베딩
        score = (u_emb * i_emb).sum(dim=-1)                # 내적
        return torch.sigmoid(score)                         # 0~1 확률


# =============================================================
# 4. 학습 데이터 — 양성 + 음성 샘플링
# =============================================================
# 양성: 실제 매칭된 쌍 (label=1)
# 음성: 매칭 안 한 쌍을 랜덤하게 (label=0)
torch.manual_seed(42)
positive_set = set(interactions)

train_data = []
for u, i in interactions:
    train_data.append((u, i, 1))   # 양성

    # 양성 1개당 음성 4개 샘플링
    for _ in range(4):
        neg_i = np.random.randint(n_items)
        while (u, neg_i) in positive_set:
            neg_i = np.random.randint(n_items)
        train_data.append((u, neg_i, 0))

user_ids = torch.tensor([u for u, _, _ in train_data], dtype=torch.long)
item_ids = torch.tensor([i for _, i, _ in train_data], dtype=torch.long)
labels = torch.tensor([lbl for _, _, lbl in train_data], dtype=torch.float)

print(f"\n학습 데이터: {len(train_data)}개 (양성 {len(interactions)} + 음성 {len(train_data) - len(interactions)})")


# =============================================================
# 5. 학습 준비
# =============================================================
EMBED_DIM = 16
N_LAYERS = 3        # K=3, 3-hop까지 정보 전파
LEARNING_RATE = 0.01
N_EPOCHS = 300

model = LightGCN(
    n_users=n_users,
    n_items=n_items,
    embed_dim=EMBED_DIM,
    n_layers=N_LAYERS,
    A_norm=A_norm,
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print(f"\n모델 구조:")
print(model)


# =============================================================
# 6. 학습
# =============================================================
print(f"\n학습 시작 (총 {N_EPOCHS} epoch, {N_LAYERS}-layer propagation)")
print("-" * 50)

for epoch in range(N_EPOCHS):
    model.train()
    pred = model(user_ids, item_ids)
    loss = loss_fn(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        preds_binary = (pred >= 0.5).float()
        acc = (preds_binary == labels).float().mean().item()
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {acc:.3f}")


# =============================================================
# 7. 전체 예측 행렬
# =============================================================
print("\n" + "=" * 50)
print("학습 완료. 전체 예측 행렬")
print("=" * 50)

model.eval()
with torch.no_grad():
    all_users = torch.tensor(
        [u for u in range(n_users) for _ in range(n_items)],
        dtype=torch.long,
    )
    all_items = torch.tensor(
        [i for _ in range(n_users) for i in range(n_items)],
        dtype=torch.long,
    )
    all_probs = model(all_users, all_items).reshape(n_users, n_items).numpy()

print(np.round(all_probs, 2))


# =============================================================
# 8. 안 본 멘토 Top-3 추천
# =============================================================
print("\n" + "=" * 50)
print("각 멘티에게 안 본 멘토 Top-3 추천 (LightGCN)")
print("=" * 50)

seen = {u: set() for u in range(n_users)}
for u, i in interactions:
    seen[u].add(i)

for u in range(n_users):
    unseen = [i for i in range(n_items) if i not in seen[u]]
    if not unseen:
        continue
    scored = sorted([(i, all_probs[u, i]) for i in unseen],
                    key=lambda x: -x[1])
    top3 = scored[:3]
    rec_str = ", ".join([f"멘토{i}({p:.2f})" for i, p in top3])
    print(f"  멘티 {u}: {rec_str}")


# =============================================================
# 9. NeuMF (Day 2) vs LightGCN (Day 3) 차이
# =============================================================
print("\n" + "=" * 50)
print("Day 2 vs Day 3 — 핵심 차이")
print("=" * 50)
print("""
Day 2 (NeuMF):
  - 멘티/멘토를 ID로만 봄 (그래프 구조 ❌)
  - 한 쌍씩 매칭 점수 예측 (1-hop 관계만)
  - MLP의 비선형성으로 복잡한 상호작용 학습

Day 3 (LightGCN):
  - 멘티/멘토를 양분 그래프 노드로 봄
  - K번 propagation으로 K-hop 이웃 정보 자동 흡수
  - 비슷한 멘티/멘토가 임베딩 공간에서 자연스럽게 가까워짐
  - "Light" = W, ReLU 같은 거 다 제거. 단순한 이웃 평균만으로 강함

의의:
  - 사용자-아이템 간 *간접 관계*까지 활용 → 콜드 스타트 일부 개선
  - 추천 SOTA의 표준 백본 중 하나 (2020~2024 많이 인용)
""")
