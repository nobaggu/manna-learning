import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Day 3에서 만든 DB에 그대로 연결
chroma_client = chromadb.PersistentClient(path="./chroma_db")
#현재 폴더의 chroma_db 라는 저장소에 연결한다.
collection = chroma_client.get_collection(name="mentor_answers")
#그 안에 있는 mentor_answers라는 컬렉션 가져온다.


def get_embedding(text: str) -> list[float]:
    """텍스트 → 1536차원 벡터"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve(query: str, n: int = 3) -> list[str]:
    """질문에 가장 관련 있는 멘토 답변 N개 검색 (RAG의 R)"""
    query_emb = get_embedding(query)#질문 임베딩 벡터화
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n,
    )
    return results['documents'][0]


def generate_with_rag(query: str) -> str:
    """검색된 답변을 컨텍스트로 LLM이 답변 생성 (RAG의 A + G)"""
    retrieved_docs = retrieve(query, n=3)

    context = "\n".join(
        [f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)]
    )

    prompt = f"""다음은 우리 서비스에 저장된 멘토들의 과거 답변입니다.

{context}

위 답변들을 참고하여 아래 질문에 한 문단으로 답해주세요.
답변할 때 어떤 멘토 답변을 참고했는지 [1], [2], [3]으로 인용 표시해주세요.

질문: {query}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def generate_without_rag(query: str) -> str:
    """비교용: 검색 없이 LLM에 그대로 질문 — 일반 ChatGPT 호출"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", #최종 답변 생성
        max_tokens=500,
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content


# === 실험 ===
question = "AI 공부 어떻게 시작해야 하나요?"

print("=" * 60)
print(f"질문: {question}")
print("=" * 60)

print("\n[1] RAG 없이 (일반 LLM)")
print("-" * 60)
print(generate_without_rag(question))

print("\n\n[2] RAG 적용 (멘토 답변 검색 + 컨텍스트 주입)")
print("-" * 60)
print(generate_with_rag(question))