"""
=============================================================
Day 1: Matrix Factorization (행렬 분해) - 추천의 가장 기본
=============================================================

목표: 멘티 × 멘토 평점 행렬에서 빈 칸을 예측해본다.
방법: NumPy로 작은 두 행렬(멘티 임베딩, 멘토 임베딩)을 학습.

실행 방법:
1. 터미널에서: python Day1_MatrixFactorization.py
2. 또는 VS Code에서 ▶ 재생 버튼 클릭

NumPy 설치 확인:
  python -c "import numpy"   ← 에러 없으면 OK
  설치: pip install numpy
"""

import sys
import numpy as np

# 윈도우 터미널(cp949) 한글 깨짐 방지: stdout을 UTF-8로 재설정
# Python 3.7+에서 작동. 다른 OS에선 무해(이미 utf-8이라 skip).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================
# 1. 데이터 준비: 멘티 × 멘토 평점 행렬
# =============================================================
# 0 = 아직 매칭 안 됨 (빈 칸, 예측 대상)
# 1~5 = 멘티의 만족도 평가

# 멘티 4명, 멘토 3명
# 행: [수영, 성준, 지영, 민호]
# 열: [김데이터, 박소셜, 이융합]
R = np.array([
    [4, 5, 0],   # 수영: 김데 4점, 박소 5점, 이융 ?
    [5, 0, 3],   # 성준: 김데 5점, 박소 ?,    이융 3점
    [0, 4, 0],   # 지영: 김데 ?,    박소 4점, 이융 ?
    [2, 3, 4],   # 민호: 김데 2점, 박소 3점, 이융 4점
])

n_users, n_items = R.shape  # n_users=4, n_items=3
print(f"원본 평점 행렬 R ({n_users}명 멘티 × {n_items}명 멘토):")
print(R)
print(f"빈 칸(0) 개수: {(R == 0).sum()}개")


# =============================================================
# 2. 하이퍼파라미터 (학습 설정값)
# =============================================================
K = 2              # 잠재 차원 수 (멘티/멘토를 몇 차원 벡터로 표현할지)
learning_rate = 0.01   # 한 번에 얼마나 수정할지 (보폭)
n_epochs = 5000        # 몇 번 반복 학습할지
reg = 0.02         # 정규화 강도 (벡터 너무 커지지 않게 제어)


# =============================================================
# 3. 임베딩 초기화: 작은 랜덤 값으로 시작
# =============================================================
# np.random.seed(42)는 결과를 재현 가능하게 만듦 (매번 같은 랜덤)
np.random.seed(42)

# U: 멘티 임베딩 (4명 × K차원)
U = np.random.normal(scale=0.1, size=(n_users, K))

# V: 멘토 임베딩 (3명 × K차원)
V = np.random.normal(scale=0.1, size=(n_items, K))

print(f"\n[초기 상태] 멘티 임베딩 U ({n_users}×{K}):")
print(U)
print(f"\n[초기 상태] 멘토 임베딩 V ({n_items}×{K}):")
print(V)


# =============================================================
# 4. 예측 함수: 멘티 u가 멘토 i에게 줄 점수
# =============================================================
def predict(u_vec, v_vec):
    """두 벡터의 내적 = 예측 점수"""
    return np.dot(u_vec, v_vec)


# =============================================================
# 5. 학습 루프: 알려진 칸으로 임베딩 조정
# =============================================================
print(f"\n학습 시작 (총 {n_epochs}번 반복)")
print("-" * 50)

for epoch in range(n_epochs):
    total_loss = 0.0

    # 행렬의 모든 칸 순회
    for u in range(n_users):
        for i in range(n_items):
            # 0(빈 칸)은 건너뜀 - 알려진 점수만 학습에 사용
            if R[u, i] == 0:
                continue

            # 현재 예측값
            pred = predict(U[u], V[i])

            # 오차: 실제값 - 예측값
            error = R[u, i] - pred

            # 임베딩 업데이트 (gradient descent)
            # 핵심: error * V[i]를 더하면 U[u]가 V[i] 방향으로 이동
            #       → 다음 예측은 실제값에 더 가까워짐
            U[u] += learning_rate * (2 * error * V[i] - reg * U[u])
            V[i] += learning_rate * (2 * error * U[u] - reg * V[i])

            # 손실 누적
            total_loss += error ** 2

    # 500 epoch마다 진행 상황 출력
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:5d} | Loss: {total_loss:.4f}")


# =============================================================
# 6. 학습 결과 확인
# =============================================================
print("\n" + "=" * 50)
print("학습 완료!")
print("=" * 50)

print(f"\n[학습 후] 멘티 임베딩 U:")
print(np.round(U, 3))
print(f"\n[학습 후] 멘토 임베딩 V:")
print(np.round(V, 3))

# 전체 행렬 재구성: U와 V^T의 곱 = 예측 행렬
# V.T는 V의 전치(transpose) = 행과 열을 바꾼 것
R_pred = U @ V.T  # @ 는 행렬 곱 연산자

print(f"\n=== 예측 평점 행렬 (모든 칸) ===")
print(np.round(R_pred, 2))

print(f"\n=== 원본과 비교 ===")
print("원본:")
print(R)
print("\n예측 (소수점 1자리):")
print(np.round(R_pred, 1))


# =============================================================
# 7. 빈 칸 예측 결과 정리 (추천!)
# =============================================================
print("\n" + "=" * 50)
print(" 빈 칸 예측 결과 (추천)")
print("=" * 50)

mentee_names = ["수영", "성준", "지영", "민호"]
mentor_names = ["김데이터", "박소셜", "이융합"]

for u in range(n_users):
    for i in range(n_items):
        if R[u, i] == 0:  # 빈 칸이었던 곳만
            print(f"  {mentee_names[u]} → {mentor_names[i]}: "
                  f"예상 만족도 {R_pred[u, i]:.2f}점")


# =============================================================
# 8. 각 멘티에게 Top-1 추천
# =============================================================
print("\n" + "=" * 50)
print(" 각 멘티에게 새 멘토 Top-1 추천")
print("=" * 50)

for u in range(n_users):
    # 안 본 멘토만 후보
    unseen_indices = [i for i in range(n_items) if R[u, i] == 0]

    if not unseen_indices:
        print(f"  {mentee_names[u]}: 모든 멘토와 매칭됨, 추천 없음")
        continue

    # 안 본 멘토 중 최고 예측값
    scores = [(i, R_pred[u, i]) for i in unseen_indices]
    best_idx, best_score = max(scores, key=lambda x: x[1])

    print(f"  {mentee_names[u]} → 추천: {mentor_names[best_idx]} "
          f"(예상 만족도 {best_score:.2f}점)")
