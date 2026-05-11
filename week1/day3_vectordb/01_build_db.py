import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 멘토들이 과거에 남긴 답변 10개 (실제 서비스라면 수백~수천 개)
documents = [
    "사회적 자본이 부족한 사람에게 멘토를 매칭하려면 먼저 멘티의 목표와 제약을 구조화해야 합니다.",
    "신입 개발자가 커리어를 시작할 때는 작은 회사에서 다양한 경험을 쌓는 것이 유리합니다.",
    "대학원 진학을 고민한다면 학교 랭킹보다 지도교수와의 핏이 훨씬 중요합니다.",
    "이직 시점은 현재 회사에서 더 이상 배울 게 없다고 느낄 때가 적기입니다.",
    "프롬프트 엔지니어링은 모델의 출력을 의도한 방향으로 유도하는 기술입니다.",
    "RAG 시스템의 검색 정확도는 임베딩 모델 선택과 청크 크기에 크게 좌우됩니다.",
    "벡터 데이터베이스는 의미 기반 검색을 가능하게 하는 핵심 인프라입니다.",
    "사회연결망분석(SNA)은 사용자 간 정보 흐름을 정량적으로 측정할 수 있게 해줍니다.",
    "팀 협업에서 가장 중요한 것은 동일한 지표 위에서 토론하는 구조를 만드는 일입니다.",
    "비전공자가 AI 분야에 진입하려면 학습 흔적을 GitHub에 꾸준히 남기는 것이 가장 효과적입니다.",
]

# Chroma의 로컬 영구 저장 클라이언트
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="mentor_answers")

# 각 문서를 임베딩으로 변환
def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

print(f"임베딩 계산 중... (총 {len(documents)}개)")
embeddings = [get_embedding(doc) for doc in documents]

# DB에 저장
collection.upsert(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings,
)

print(f"\n저장 완료. 컬렉션 내 문서 수: {collection.count()}")