"""BM25 한국어 검색의 기본 — 같은 10개 문서를 키워드 기반으로 검색"""

from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

# Day 3와 동일한 멘토 답변 10개 (BM25는 임베딩 안 쓰니까 DB 없어도 가능)
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

# 1. 한국어 형태소 분석기 준비
okt = Okt()

# 2. 모든 문서를 토큰화 (단어 단위로 자르기)
#문장을 형태소 단위(영어와의 차이)로 쪼개는 과정-BM25는 문자열 단위가 아닌
#토큰 단위로 검색 수행 ['신입', '개발자'] 같은
print("문서 토큰화 중...")
tokenized_docs = [okt.morphs(doc) for doc in documents]

# 첫 번째 문서의 토큰화 결과 보기 (감 잡기)
print(f"\n[원본 문서 0]")
print(f"  {documents[0]}")
print(f"\n[토큰화 결과 0]")
print(f"  {tokenized_docs[0]}")

# 3. BM25 인덱스 구축
bm25 = BM25Okapi(tokenized_docs)
print(f"\nBM25 인덱스 완성. 총 문서 수: {len(documents)}\n")

# 4. 검색 함수 정의
def search_bm25(query: str, n: int = 3):
    """질문을 토큰화한 뒤 BM25 점수가 높은 N개 문서 반환"""
    query_tokens = okt.morphs(query)
    scores = bm25.get_scores(query_tokens)
    # 점수 높은 순으로 정렬, 상위 N개 인덱스 추출
    top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
    return query_tokens, [(documents[idx], score) for idx, score in top_n]


# 5. 테스트 쿼리
query = "신입 개발자 진로"
query_tokens, results = search_bm25(query)

print(f"질문: {query}")
print(f"질문 토큰: {query_tokens}\n")

print("=== BM25 Top 3 ===")
for rank, (doc, score) in enumerate(results, 1):
    print(f"{rank}. (점수 {score:.4f}) {doc}")