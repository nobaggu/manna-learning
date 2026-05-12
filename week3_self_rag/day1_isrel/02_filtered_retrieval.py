"""Hybrid 검색 + isRel 필터 — Self-RAG의 첫 번째 안전망"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 시스템 준비 (Week 2와 동일) ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="mentor_answers")

okt = Okt()
all_data = collection.get()
documents = all_data['documents']
doc_ids = all_data['ids']

tokenized_docs = [okt.morphs(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


def get_doc_text(doc_id: str) -> str:
    idx = doc_ids.index(doc_id)
    return documents[idx]


def search_hybrid(query: str, n: int = 10, k: int = 60) -> list[str]:
    """BM25 + Dense → RRF 융합"""
    # BM25
    query_tokens = okt.morphs(query)
    scores = bm25.get_scores(query_tokens)
    bm25_top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    bm25_ids = [doc_ids[idx] for idx, _ in bm25_top]

    # Dense
    emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    dense_res = collection.query(query_embeddings=[emb], n_results=n)
    dense_ids = dense_res['ids'][0]

    # RRF
    rrf = {}
    for rank, did in enumerate(bm25_ids):
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)
    for rank, did in enumerate(dense_ids):
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [did for did, _ in sorted_ids[:n]]


def is_relevant(query: str, document: str) -> bool:
    prompt = f"""다음 문서가 아래 질문에 답하는 데 관련 있는지 판단해주세요.

[문서]
{document}

[질문]
{query}

답변은 반드시 'yes' 또는 'no'로만 시작하세요."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip().lower().startswith("yes")


def filtered_retrieve(query: str, n_candidates: int = 10, n_final: int = 3) -> list[str]:
    """
    Hybrid로 N개 후보 검색 → isRel로 필터 → 최종 K개 반환
    필터 통과한 게 K개 미만이면 그만큼만 반환 (또는 빈 리스트)
    """
    candidates = search_hybrid(query, n=n_candidates)
    filtered = []
    for doc_id in candidates:
        if is_relevant(query, get_doc_text(doc_id)):
            filtered.append(doc_id)
        if len(filtered) >= n_final:
            break
    return filtered


# === 실험: 검색만 vs 검색+isRel 비교 ===
queries = [
    "신입 개발자 진로",
    "이직 언제 해야",
    "사람 사이 정보 흐름",
    "오늘 점심 메뉴",           # 무관 질문 — isRel이 다 걸러야 정상
    "프롬프트 엔지니어링",
    "내일 날씨 어때요",          # 또 다른 무관 질문
]

for query in queries:
    print("=" * 75)
    print(f"질문: {query}")
    print("=" * 75)

    # 검색만 (Week 2 결과)
    only_search = search_hybrid(query, n=3)
    print("\n[검색만 — Top 3]")
    for rank, did in enumerate(only_search, 1):
        print(f"  {rank}. [{did}] {get_doc_text(did)[:50]}...")

    # 검색 + isRel 필터
    filtered = filtered_retrieve(query, n_candidates=10, n_final=3)
    print(f"\n[검색 + isRel 필터 — {len(filtered)}개 통과]")
    if not filtered:
        print("  → 모든 후보가 isRel 필터에서 거부됨 (= 시스템이 *모른다*고 말함)")
    else:
        for rank, did in enumerate(filtered, 1):
            print(f"  {rank}. [{did}] {get_doc_text(did)[:50]}...")
    print()