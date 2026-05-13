"""Full RAG + isRel + isSup — Self-RAG의 첫 두 안전망까지 통합"""

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


# === Week 2 검색 함수들 ===
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


# === Day 1 isRel ===
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


# === Day 2 isSup ===
def is_supported(answer: str, documents_used: list[str]) -> tuple[bool, str]:
    docs_text = "\n\n".join([f"[문서 {i+1}]\n{d}" for i, d in enumerate(documents_used)])
    prompt = f"""다음 답변이 아래 자료들로 *뒷받침*되는지 판단해주세요.

[제공된 자료]
{docs_text}

[답변]
{answer}

판단 기준:
1. 답변의 모든 주장이 자료에 직접 명시되었거나 합리적으로 추론 가능하면 'yes'
2. 자료에 없는 새 사실·숫자·인용이 포함되면 'no'
3. 자료를 반대로 말하거나 왜곡하면 'no'

답변은 'yes' 또는 'no'로 시작, 그 뒤에 한 문장 이내 설명."""
    res = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=80, temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    out = res.choices[0].message.content.strip()
    return out.lower().startswith("yes"), out


# === LLM 답변 생성 ===
def generate_answer(query: str, context_docs: list[str]) -> str:
    context = "\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context_docs)])
    prompt = f"""아래 자료들만 참고하여 질문에 답해주세요. 자료에 없는 정보는 절대 추가하지 마세요.

[자료]
{context}

[질문]
{query}

답변은 두 문장 이내로, 어떤 자료를 참고했는지 [1], [2] 식으로 표시하세요."""
    res = client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=300, temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content.strip()


# === 통합 파이프라인 ===
def full_pipeline(query: str) -> dict:
    """검색 → isRel 필터 → 생성 → isSup 검증 → 최종 의사결정"""
    result = {"query": query, "stage_reached": "", "final_answer": None, "reason": ""}

    # 1. Hybrid 검색
    candidates = search_hybrid(query, n=10)
    result["stage_reached"] = "검색완료"

    # 2. isRel 필터
    relevant_docs = []
    for did in candidates:
        if is_relevant(query, get_doc_text(did)):
            relevant_docs.append(did)
        if len(relevant_docs) >= 3:
            break

    result["filtered_doc_ids"] = relevant_docs

    if not relevant_docs:
        result["stage_reached"] = "isRel 전부 거부"
        result["reason"] = "관련 문서 없음 → 멘토 연결로 분기 필요"
        return result

    # 3. LLM 답변 생성
    context_texts = [get_doc_text(d) for d in relevant_docs]
    answer = generate_answer(query, context_texts)
    result["generated_answer"] = answer
    result["stage_reached"] = "답변생성"

    # 4. isSup 검증
    supported, sup_reason = is_supported(answer, context_texts)
    result["issup_verdict"] = supported
    result["issup_reason"] = sup_reason

    if not supported:
        result["stage_reached"] = "isSup 거부"
        result["reason"] = f"답변이 문서로 뒷받침 안 됨 → 재생성 또는 멘토 연결 필요. (LLM 판단: {sup_reason})"
        return result

    # 모두 통과
    result["stage_reached"] = "최종승인"
    result["final_answer"] = answer
    return result


# === 실험 ===
queries = [
    "신입 개발자 진로",        # 정상 질문 → 다 통과해야 함
    "이직 언제 해야 하나요",   # 정상 질문
    "오늘 점심 메뉴",          # 무관 질문 → isRel에서 거부돼야 함
    "내일 비 와요?",            # 또 다른 무관 질문
    "프롬프트 엔지니어링",     # 정상 질문
]

for query in queries:
    print("=" * 80)
    print(f"질문: {query}")
    print("=" * 80)
    res = full_pipeline(query)
    print(f"\n도달 단계: {res['stage_reached']}")
    print(f"isRel 통과 문서: {res.get('filtered_doc_ids', [])}")
    if res.get("generated_answer"):
        print(f"\n생성 답변:\n  {res['generated_answer']}")
    if "issup_verdict" in res:
        print(f"\nisSup 판단: {res['issup_verdict']}")
        print(f"  이유: {res['issup_reason']}")
    if res["final_answer"]:
        print(f"\n✅ 최종 답변 (사용자에게 전달):\n  {res['final_answer']}")
    else:
        print(f"\n❌ 사용자에게 전달 안 함")
        print(f"   이유: {res['reason']}")
    print()