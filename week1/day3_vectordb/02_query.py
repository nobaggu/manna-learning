import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="mentor_answers")


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# 사용자가 물어본 질문
query = "사람 사이 정보가 어떻게 흐르나요?"
print(f"질문: {query}\n")

query_embedding = get_embedding(query)

# 가장 가까운 문서 3개 찾기- chroma는 거리를 반환함
#작을수록 가까움
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)

print("=== 가장 관련 있는 답변 Top 3 ===\n")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"{i+1}. [거리 {distance:.4f}]")
    print(f"   {doc}\n")