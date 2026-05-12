"""세 가지 검색 방식의 정량 평가 — Hit@1, Hit@3, MRR"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 시스템 준비 (Day 1·2와 동일) ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="mentor_answers")

okt = Okt()
all_data = collection.get()
documents = all_data['documents']
doc_ids = all_data['ids']

tokenized_docs = [okt.morphs(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


# === 검색 함수 — Top N의 doc_id 리스트만 반환 ===
def search_bm25(query: str, n: int = 10):
    query_tokens = okt.morphs(query)
    scores = bm25.get_scores(query_tokens)
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return [doc_ids[idx] for idx, _ in top_n]


def search_dense(query: str, n: int = 10):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_emb = response.data[0].embedding
    results = collection.query(query_embeddings=[query_emb], n_results=n)
    return results['ids'][0]


def search_hybrid(query: str, n: int = 10, k: int = 60):
    """RRF로 BM25 + Dense 융합"""
    bm25_ids = search_bm25(query, n=10)
    dense_ids = search_dense(query, n=10)

    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_ids[:n]]


# === 테스트셋 — (질문, 정답 doc_id) ===
# 의도적으로 세 카테고리로 설계
test_set = [
    # ── 키워드 우세 (BM25 강할 것) ──
    ("신입 개발자 커리어 시작", "doc_1"),
    ("이직 시점은 언제가 좋나요", "doc_3"),
    ("GitHub 학습 흔적", "doc_9"),
    ("프롬프트 엔지니어링이란", "doc_4"),

    # ── 의미 우세 (Dense 강할 것 — 단어 안 겹침) ──
    ("직장 옮기는 게 좋을 때", "doc_3"),                # 이직 ≈ 직장 옮김
    ("처음 일 시작하는 개발자", "doc_1"),                # 신입 ≈ 처음
    ("AI 분야 처음 들어가는 사람", "doc_9"),             # 비전공자 진입 ≈ 처음 들어가는
    ("사람들 사이 정보가 어떻게 도는지", "doc_7"),        # 사용자 간 정보 흐름
    ("석박사 진학 고민", "doc_2"),                       # 대학원 ≈ 석박사

    # ── 혼합 ──
    ("멘토 어떻게 매칭하나요", "doc_0"),
]


# === 평가 함수 ===
def evaluate(search_fn, test_set):
    """Hit@1, Hit@3, MRR + 질문별 상세 반환"""
    hit_at_1 = 0
    hit_at_3 = 0
    rr_sum = 0
    details = []

    for query, correct_id in test_set:
        results = search_fn(query, n=10)

        # 정답이 몇 위에 있는지 (0이면 Top 10 밖)
        if correct_id in results:
            rank = results.index(correct_id) + 1
        else:
            rank = 0

        # 누적 집계
        if rank == 1:
            hit_at_1 += 1
        if 1 <= rank <= 3:
            hit_at_3 += 1
        rr = 1 / rank if rank > 0 else 0
        rr_sum += rr

        details.append({"query": query, "correct": correct_id, "rank": rank})

    n = len(test_set)
    return {
        "hit_at_1": hit_at_1 / n,
        "hit_at_3": hit_at_3 / n,
        "mrr": rr_sum / n,
        "details": details,
    }


# === 세 방식 평가 ===
print("=" * 80)
print(f"테스트셋 크기: {len(test_set)}개 질문")
print("=" * 80)

methods = {
    "BM25  ": search_bm25,
    "Dense ": search_dense,
    "Hybrid": search_hybrid,
}

results = {name: evaluate(fn, test_set) for name, fn in methods.items()}


# === 전체 지표 표 ===
print("\n[종합 지표]")
print(f"{'방식':<10} {'Hit@1':>10} {'Hit@3':>10} {'MRR':>10}")
print("-" * 42)
for name, res in results.items():
    print(f"{name:<10} {res['hit_at_1']*100:>9.1f}% {res['hit_at_3']*100:>9.1f}% {res['mrr']:>10.4f}")


# === 질문별 정답 순위 비교 ===
print("\n[질문별 정답 rank — rank=0은 Top 10에 못 찾음]")
print(f"{'질문':<34} {'정답':<8} {'BM25':>6} {'Dense':>6} {'Hybrid':>6}")
print("-" * 70)

for i, (query, correct) in enumerate(test_set):
    bm25_rank = results["BM25  "]["details"][i]["rank"]
    dense_rank = results["Dense "]["details"][i]["rank"]
    hybrid_rank = results["Hybrid"]["details"][i]["rank"]

    q_short = (query[:31] + "..") if len(query) > 33 else query
    print(f"{q_short:<34} {correct:<8} {bm25_rank:>6} {dense_rank:>6} {hybrid_rank:>6}")