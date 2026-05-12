"""BM25(키워드) vs Dense(의미) 검색 결과를 나란히 비교"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Dense 쪽 준비 (Day 3 DB 그대로 사용) ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="mentor_answers")

# === BM25 쪽 준비 ===
okt = Okt()

# Chroma DB에서 모든 문서를 가져와서 BM25 인덱스도 같은 데이터로 구축
all_data = collection.get()
documents = all_data['documents']

tokenized_docs = [okt.morphs(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


def search_bm25(query: str, n: int = 3):
    query_tokens = okt.morphs(query)
    scores = bm25.get_scores(query_tokens)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [(documents[idx], score) for idx, score in top_n]


def search_dense(query: str, n: int = 3):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = response.data[0].embedding
    results = collection.query(query_embeddings=[query_emb], n_results=n)
    return list(zip(results['documents'][0], results['distances'][0]))


# === 실험 ===
queries = [
    "신입 개발자 진로",            # 키워드 정확 매칭 강함 → BM25 유리
    "이직 언제 해야 하나요",        # 의미는 비슷한데 단어 변형 → Dense 유리
    "사람 사이 정보 흐름",          # 도메인 용어 변형 → Dense 유리 추정
    "GitHub 학습 흔적",             # 키워드 정확 → BM25 유리
    "벡터 DB 왜 쓰나요",            # 약자 매칭 → 어느 쪽?
    "오늘 점심 메뉴",                # 무관 질문 → 둘 다 실패해야 정상
]

for query in queries:
    print("=" * 70)
    print(f"질문: {query}")
    print("=" * 70)

    print("\n[BM25 — 키워드 매칭]")
    for rank, (doc, score) in enumerate(search_bm25(query), 1):
        print(f"  {rank}. (점수 {score:.3f}) {doc[:50]}...")

    print("\n[Dense — 의미 매칭]")
    for rank, (doc, distance) in enumerate(search_dense(query), 1):
        print(f"  {rank}. (거리 {distance:.3f}) {doc[:50]}...")

    print()