"""완전한 Self-RAG 파이프라인 — Hybrid 검색 + isRel + isSup + isUse"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 시스템 준비 ===
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="mentor_answers")

okt = Okt()
all_data = collection.get()
documents = all_data['documents']
doc_ids = all_data['ids']

tokenized_docs = [okt.morphs(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


def get_doc_text(doc_id: str) -> str:
    return documents[doc_ids.index(doc_id)]


# === Hybrid 검색 (Week 2) ===
def search_hybrid(query: str, n: int = 10, k: int = 60) -> list[str]:
    query_tokens = okt.morphs(query)
    scores = bm25.get_scores(query_tokens)
    bm25_top = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    bm25_ids = [doc_ids[idx] for idx, _ in bm25_top]

    emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    dense_res = collection.query(query_embeddings=[emb], n_results=n)
    dense_ids = dense_res['ids'][0]

    rrf = {}
    for rank, did in enumerate(bm25_ids):
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)
    for rank, did in enumerate(dense_ids):
        rrf[did] = rrf.get(did, 0) + 1 / (k + rank + 1)

    return [did for did, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:n]]


# === Self-RAG 3가지 검증기 ===
def is_relevant(query: str, document: str) -> bool:
    prompt = f"""다음 문서가 아래 질문에 답하는 데 관련 있는지 판단해주세요.

[문서]
{document}

[질문]
{query}

답변은 반드시 'yes' 또는 'no'로만 시작하세요."""
    res = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=10, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip().lower().startswith("yes")


def is_supported(answer: str, documents_used: list[str]) -> tuple[bool, str]:
    docs_text = "\n\n".join([f"[문서 {i+1}]\n{d}" for i, d in enumerate(documents_used)])
    prompt = f"""다음 답변이 아래 자료들로 뒷받침되는지 판단해주세요.

[제공된 자료]
{docs_text}

[답변]
{answer}

판단 기준:
1. 모든 주장이 자료에 직접 명시되거나 합리적으로 추론 가능 → 'yes'
2. 자료에 없는 새 사실·숫자·인용 포함 → 'no'
3. 자료를 왜곡 → 'no'

'yes' 또는 'no'로 시작, 그 뒤에 한 문장 이내 설명."""
    res = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=80, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    out = res.choices[0].message.content.strip()
    return out.lower().startswith("yes"), out


def is_useful(query: str, answer: str) -> tuple[bool, str]:
    prompt = f"""다음 답변이 사용자 질문에 실질적으로 유용한지 판단해주세요.

[질문]
{query}

[답변]
{answer}

판단 기준:
1. 질문에 직접 답하는가
2. 구체적이고 실행 가능한 정보가 있는가
3. 사용자가 다음 행동을 판단할 수 있게 돕는가
4. 회피성·동어반복 답변은 아닌가

'yes' 또는 'no'로 시작, 한 문장 이내 설명."""
    res = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=80, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    out = res.choices[0].message.content.strip()
    return out.lower().startswith("yes"), out


# === 답변 생성 ===
def generate_answer(query: str, context_docs: list[str]) -> str:
    context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
    prompt = f"""아래 자료들만 참고해 질문에 답하세요. 자료에 없는 정보는 절대 추가 금지.

[자료]
{context}

[질문]
{query}

답변은 두 문장 이내로, [1], [2] 식으로 인용 표시하세요."""
    res = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=300, temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()


# === 완전한 Self-RAG 파이프라인 ===
def self_rag(query: str) -> dict:
    """
    너 도식 ④번 완성형:
    Hybrid 검색 → isRel 필터 → 생성 → isSup 검증 → isUse 검증 → 분기
    """
    log = {"query": query, "stage": "", "answer": None, "branch": None}

    # 1. Hybrid 검색
    candidates = search_hybrid(query, n=10)

    # 2. isRel 필터
    relevant = []
    for did in candidates:
        if is_relevant(query, get_doc_text(did)):
            relevant.append(did)
        if len(relevant) >= 3:
            break

    log["filtered_ids"] = relevant

    if not relevant:
        log["stage"] = "isRel_거부"
        log["branch"] = "멘토_연결"  # 너 도식의 분기점
        log["reason"] = "관련 문서 없음"
        return log

    # 3. 답변 생성
    context_texts = [get_doc_text(d) for d in relevant]
    answer = generate_answer(query, context_texts)
    log["generated"] = answer

    # 4. isSup 검증
    sup_ok, sup_reason = is_supported(answer, context_texts)
    log["issup_verdict"] = sup_ok
    log["issup_reason"] = sup_reason

    if not sup_ok:
        log["stage"] = "isSup_거부"
        log["branch"] = "멘토_연결"
        log["reason"] = f"답변이 문서로 뒷받침 안 됨: {sup_reason}"
        return log

    # 5. isUse 검증
    use_ok, use_reason = is_useful(query, answer)
    log["isuse_verdict"] = use_ok
    log["isuse_reason"] = use_reason

    if not use_ok:
        log["stage"] = "isUse_거부"
        log["branch"] = "멘토_연결"
        log["reason"] = f"답변이 충분히 유용하지 않음: {use_reason}"
        return log

    # 6. 모두 통과
    log["stage"] = "최종승인"
    log["branch"] = "사용자_전달"
    log["answer"] = answer
    return log


# === 실험 ===
queries = [
    "신입 개발자 진로",
    "이직 언제 해야 하나요",
    "사람 사이 정보 흐름",
    "오늘 점심 메뉴",           # 무관 → isRel에서 멈춰야
    "내일 날씨 어때요",          # 무관
    "프롬프트 엔지니어링 입문",
    "AI 분야 비전공자 진입",
]

stats = {"최종승인": 0, "isRel_거부": 0, "isSup_거부": 0, "isUse_거부": 0}

for query in queries:
    print("=" * 80)
    print(f"질문: {query}")
    print("=" * 80)

    res = self_rag(query)
    stats[res["stage"]] += 1

    print(f"\n도달 단계: {res['stage']}")
    print(f"분기 결과: {res['branch']}")
    print(f"isRel 통과 문서: {res.get('filtered_ids', [])}")

    if res.get("generated"):
        print(f"\n생성된 답변:\n  {res['generated']}")

    if "issup_verdict" in res:
        print(f"\nisSup: {res['issup_verdict']} — {res['issup_reason'][:60]}")
    if "isuse_verdict" in res:
        print(f"isUse: {res['isuse_verdict']} — {res['isuse_reason'][:60]}")

    if res["answer"]:
        print(f"\n✅ 사용자에게 전달:\n  {res['answer']}")
    else:
        print(f"\n❌ 멘토 연결로 분기")
        print(f"   이유: {res['reason']}")
    print()

# 통계
print("=" * 80)
print("실험 통계 — 분기 분포")
print("=" * 80)
for stage, count in stats.items():
    pct = count / len(queries) * 100
    print(f"  {stage}: {count}/{len(queries)} ({pct:.0f}%)")