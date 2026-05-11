import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 비교할 문장들
sentences = {
    "A": "사회적 자본이 부족한 사람들에게 멘토를 연결해주는 서비스를 만들고 싶습니다. 정보 접근성 격차를 해소하는 게 목표입니다.",
    "B": "인적 네트워크가 약한 사용자에게 경험자의 조언을 매칭해주는 플랫폼을 기획 중입니다. 진로 정보 불균형이 해결되길 바랍니다.",
    "C": "오늘 저녁은 김치찌개와 계란말이를 만들어 먹었습니다. 김치가 잘 익어서 정말 맛있었어요. 다음엔 두부도 넣어야겠어요.",
}
# 모든 문장의 임베딩 계산
print("임베딩 계산 중...")
embeddings = {key: get_embedding(text) for key, text in sentences.items()}

# 모든 쌍에 대해 유사도 출력
print("\n=== 유사도 결과 ===")
for k1, v1 in sentences.items():
    for k2, v2 in sentences.items():
        if k1 < k2:  # 같은 쌍 두 번 안 찍게
            sim = cosine_similarity(embeddings[k1], embeddings[k2])
            print(f"{k1} ↔ {k2}: {sim:.4f}   ({v1[:15]}... ↔ {v2[:15]}...)")