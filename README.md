# Manna Learning

> **사회학과 학부생이 AI Rookie 대회를 준비하며 진행한 5주 자기주도 학습 기록.**
> 막연한 진로 고민에 답하는 멘토-멘티 매칭 서비스 *Manna(맨투멘)* 를 구현하기 위해,
> RAG → Hybrid Retrieval → Self-RAG → Graph RAG → 추천 시스템 순서로 깊이 들어갑니다.

---

## 1. 왜 이걸 시작했나

사회학 학부 커리큘럼만으로는 LLM, RAG, 추천 시스템 등 핵심 기술을 다루기 어려웠습니다. 그래서 **대회 전 자기주도 학습**을 설계해, 단순한 모방이 아닌 *왜 그렇게 작동하는지*를 코드로 직접 짜보며 익혔습니다. 본 레포는 그 흔적입니다.

---

## 2. 학습 로드맵

| 주차 | 주제 | 핵심 키워드 | 상태 |
|------|------|------|------|
| **Week 1** | Basic RAG 파이프라인 | API · Embedding · VectorDB · Retrieval | ✅ 완료 |
| **Week 2** | Hybrid Retrieval | BM25 · RRF · Retrieval 평가 | ✅ 완료 |
| **Week 3** | Self-RAG (자기검증 LLM) | isRel · isSup · isUse | ✅ 완료 |
| **Week 4** | Graph RAG | NetworkX · 지식 그래프 · 관계 탐색 | ✅ 완료 |
| **Week 5** | 추천 시스템 | Matrix Factorization → NeuMF → LightGCN → SOTA | 🔄 진행 중 |

---

## 3. 주차별 상세 — Day by Day

### Week 1: Basic RAG 파이프라인 구축
LLM 호출부터 시작해 검색 증강 생성(RAG)의 가장 단순한 형태까지를 *직접* 한 줄씩 작성했습니다.

- `day1_first_api/` — open ai API 첫 호출, 프롬프트 설계 기초
- `day2_embedding/` — 임베딩이 뭔지, 텍스트가 벡터가 되는 의미
- `day3_vectordb/` — ChromaDB로 벡터 인덱스 구축
- `day4_basic_rag/` — Retrieval + Augmentation + Generation 통합

**배운 점**: LLM은 마법이 아니라 "검색해서 가져온 컨텍스트로 그럴듯하게 답하는 모듈"임을 체감.

---

### Week 2: Hybrid Retrieval — 검색의 두 가지 시선
의미 기반 검색(Vector)만으론 부족하다는 걸 알고 키워드 기반 검색을 더해보는 주차.

- `day1_bm25/` — 전통적 키워드 검색 알고리즘 BM25 구현
- `day2_rrf/` — Reciprocal Rank Fusion으로 두 검색 결과 결합
- `day3_evaluation/` — Retrieval 품질 평가 (Precision, Recall, MRR 등)

**배운 점**: 단일 검색 알고리즘은 한 면만 본다. 의미와 키워드를 결합하면 *cold start*와 *고유명사 검색* 모두 강해진다.

---

### Week 3: Self-RAG — 자기검증하는 LLM
LLM이 자기 답변을 스스로 평가하게 만들어, 무작정 답하지 않고 *판단*하게 만드는 주차.

- `day1_isrel/` — isRelevant: 검색 결과가 질문과 관련 있는가?
- `day2_issup/` — isSupported: 답변이 근거에 잘 뒷받침되는가?
- `day3_isuse/` — isUseful: 사용자에게 실제로 유용한 답변인가?

**배운 점**: LLM의 환각(hallucination)은 "검증 단계가 없어서" 발생. 3-step self-evaluation으로 답변 품질이 눈에 띄게 개선됨.

**실험 인사이트**: `isUse` 평가가 지나치게 엄격하면 멀쩡한 답변도 거부됨 → 평가 프롬프트의 *strictness 튜닝*이 핵심임을 발견.

---

### Week 4: Graph RAG — 관계 기반 검색
임베딩 유사도만으로는 잡지 못하는 *구조적 관계*를 그래프로 표현.

- `day1_networkx_basic/` — NetworkX로 노드/엣지 다루는 기초

**배운 점**: 본 프로젝트의 핵심인 *사회적 자본(관계 자산)* 은 본질적으로 그래프적 객체. 벡터 임베딩이 의미 유사성을 잡는다면, 그래프는 *관계의 거리*를 잡는다.

---

### Week 5: 추천 시스템 (현재 진행)
멘토-멘티 매칭의 본질은 추천 문제. GraphRAG의 한계를 보완하기 위해 추천 알고리즘의 발전사를 따라가는 주차.

- `week5_matrix/Day1_MatrixFactorization.py` — 행렬 분해를 NumPy로 직접 구현. 임베딩이 학습으로 의미를 만들어내는 과정 체득
- *(예정)* Day 2: NeuMF — 신경망 기반 추천
- *(예정)* Day 3: LightGCN — 그래프 기반 협업 필터링
- *(예정)* Day 4: SOTA 흐름 (LightGCL, LLMRec, DiffRec)
- *(예정)* Day 5: 본 프로젝트(Manna)에 어떻게 적용할지 설계

**현재 학습 중인 질문**: 콜드 스타트 상황에서 그래프 기반(LightGCN)과 텍스트 매칭(Bridge embedding)을 어떻게 결합할 것인가.

---


## 4. 활용 기술

| 영역 | 도구 |
|------|------|
| 언어 | Python 3.11 |
| 수치 연산 | NumPy |
| 딥러닝 | PyTorch *(Week 5~)* |
| 그래프 | NetworkX, *PyTorch Geometric (예정)* |
| Vector DB | ChromaDB |
| LLM | Upstage Solar, *LG K-EXAONE (프로젝트 예정)* |
| Retrieval | BM25, Embedding-based, Hybrid (RRF) |


