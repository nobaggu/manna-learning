# Manna Learning

공모전 AI Rookie 본선 참여를 위해 LLM·RAG 등 관련 기술을 공부한 기록입니다.
전공이 사회학이라 이쪽 기술을 다채롭게 다뤄본 적이 없어, 프로젝트에 쓰일 기술들을 직접 짜며 익혀왔습니다.

---

## 공부한 주제

- Basic RAG
- Hybrid Retrieval (BM25 · RRF)
- Self-RAG (isRel · isSup · isUse)
- Graph RAG
- 추천 시스템 (Matrix Factorization → NeuMF → LightGCN → ...)

프로젝트에 필요한 기술이 더 생기면 계속 추가될 예정입니다.

---

## 폴더 구조

```
manna-learning/
├── week1/                       Basic RAG
│   ├── day1_first_api
│   ├── day2_embedding
│   ├── day3_vectordb
│   └── day4_basic_rag
├── week2_hybrid_retrieval/      Hybrid Retrieval
│   ├── day1_bm25
│   ├── day2_rrf
│   └── day3_evaluation
├── week3_self_rag/              Self-RAG
│   ├── day1_isrel
│   ├── day2_issup
│   └── day3_isuse
├── week4_graph_rag/             Graph RAG
│   └── day1_networkx_basic
└── week5_matrix/                추천 시스템
    └── Day1_MatrixFactorization.py
```

---

## 활용 기술

- Python 3.11
- NumPy, PyTorch
- ChromaDB
- NetworkX

---

## 연관 프로젝트

**Manna (맨투멘)** — AI Rookie 본선 출품작. 이 레포의 학습이 적용되는 실제 서비스입니다.
