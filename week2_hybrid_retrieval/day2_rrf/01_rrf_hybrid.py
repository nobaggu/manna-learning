"""RRF (Reciprocal Rank Fusion)으로 BM25 + Dense 결과 융합"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 두 검색 시스템 준비 (Day 1과 동일) ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="mentor_answers")

okt = Okt()

# Chroma에서 문서 + ID 모두 가져옴 (BM25 인덱스도 같은 데이터로 구축)
all_data = collection.get()
documents = all_data['documents']
doc_ids = all_data['ids']

tokenized_docs = [okt.morphs(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


def search_bm25(query: str, n: int = 10):
    """BM25 검색 — [(doc_id, rank), ...] 반환. rank는 1부터."""
    query_tokens = okt.morphs(query)
    scores = bm25.get_scores(query_tokens)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [(doc_ids[idx], rank + 1) for rank, (idx, _) in enumerate(top_n)]


def search_dense(query: str, n: int = 10):
    """Dense 검색 — [(doc_id, rank), ...] 반환."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
    )
    returned_ids = results['ids'][0]
    return [(doc_id, rank + 1) for rank, doc_id in enumerate(returned_ids)]


def reciprocal_rank_fusion(bm25_results, dense_results, k: int = 60, n: int = 3):
    """
    RRF 공식: score(d) = sum(1 / (k + rank(d in each search)))

    bm25_results, dense_results: [(doc_id, rank), ...] 형태
    k: 상수 (60이 표준)
    n: 최종 반환할 개수
    """
    rrf_scores = {}

    # BM25 기여
    for doc_id, rank in bm25_results:
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    # Dense 기여 (이미 있으면 더함, 없으면 새로 추가)
    for doc_id, rank in dense_results:
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    # 점수 큰 순으로 정렬
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:n]


def get_doc_text(doc_id: str) -> str:
    """doc_id로 원본 텍스트 조회"""
    idx = doc_ids.index(doc_id)
    return documents[idx]


# === 실험 ===
queries = [
    "신입 개발자 진로",
    "이직 언제 해야 하나요",
    "사람 사이 정보 흐름",
    "GitHub 학습 흔적",
    "벡터 DB 왜 쓰나요",
    "오늘 점심 메뉴",
]

for query in queries:
    print("=" * 75)
    print(f"질문: {query}")
    print("=" * 75)

    # 각 검색에서 Top 10 가져옴 (후보군 넓게 잡기)
    bm25_res = search_bm25(query, n=10)
    dense_res = search_dense(query, n=10)

    # RRF로 융합해서 최종 Top 3
    hybrid_res = reciprocal_rank_fusion(bm25_res, dense_res, n=3)

    print("\n[BM25 Top 3]")
    for rank, (doc_id, _) in enumerate(bm25_res[:3], 1):
        print(f"  {rank}. [{doc_id}] {get_doc_text(doc_id)[:50]}...")

    print("\n[Dense Top 3]")
    for rank, (doc_id, _) in enumerate(dense_res[:3], 1):
        print(f"  {rank}. [{doc_id}] {get_doc_text(doc_id)[:50]}...")

    print("\n[Hybrid RRF Top 3] ← 융합 결과")
    for rank, (doc_id, score) in enumerate(hybrid_res, 1):
        print(f"  {rank}. [{doc_id}] (RRF={score:.4f}) {get_doc_text(doc_id)[:50]}...")

    print()