"""그래프 탐색 — 멘토 매칭의 기본 동작"""

import networkx as nx

# === 더 큰 그래프 — 멘토 3명 ===
G = nx.MultiDiGraph()

# 멘토들
G.add_node("mentor_kim", type="Mentor", name="김OO", years=5)
G.add_node("mentor_park", type="Mentor", name="박OO", years=8)
G.add_node("mentor_lee", type="Mentor", name="이OO", years=3)

# 직무
G.add_node("DS", type="Role", label="데이터 사이언티스트")
G.add_node("MLE", type="Role", label="ML 엔지니어")
G.add_node("BE", type="Role", label="백엔드 개발자")

# Trait
G.add_node("비전공자_전환", type="Trait")
G.add_node("사회과학_출신", type="Trait")
G.add_node("이직_경험", type="Trait")

# Topic
G.add_node("학습_로드맵", type="Topic")
G.add_node("포트폴리오", type="Topic")
G.add_node("면접_준비", type="Topic")

# 김OO: 비전공자 출신 DS
G.add_edge("mentor_kim", "DS", relation="has_role")
G.add_edge("mentor_kim", "비전공자_전환", relation="has_trait")
G.add_edge("mentor_kim", "사회과학_출신", relation="has_trait")
G.add_edge("mentor_kim", "학습_로드맵", relation="can_advise")
G.add_edge("mentor_kim", "포트폴리오", relation="can_advise")

# 박OO: 이직 경험 많은 MLE
G.add_edge("mentor_park", "MLE", relation="has_role")
G.add_edge("mentor_park", "이직_경험", relation="has_trait")
G.add_edge("mentor_park", "포트폴리오", relation="can_advise")
G.add_edge("mentor_park", "면접_준비", relation="can_advise")

# 이OO: BE 개발자
G.add_edge("mentor_lee", "BE", relation="has_role")
G.add_edge("mentor_lee", "이직_경험", relation="has_trait")
G.add_edge("mentor_lee", "면접_준비", relation="can_advise")

# === 탐색 함수들 ===

def get_neighbors_by_relation(G, node, relation):
    """특정 relation을 가진 이웃 노드만 반환"""
    result = []
    for _, neighbor, attrs in G.out_edges(node, data=True):
        if attrs.get("relation") == relation:
            result.append(neighbor)
    return result


def get_incoming_by_relation(G, node, relation):
    """특정 relation으로 *이 노드를 향하는* 노드들"""
    result = []
    for source, _, attrs in G.in_edges(node, data=True):
        if attrs.get("relation") == relation:
            result.append(source)
    return result


def get_nodes_by_type(G, node_type):
    """특정 type의 모든 노드"""
    return [n for n, attrs in G.nodes(data=True) if attrs.get("type") == node_type]


# === 실습 1: 김OO 멘토는 무엇을 갖고 있나? ===
print("=" * 60)
print("실습 1 — mentor_kim의 모든 연결")
print("=" * 60)
for _, neighbor, attrs in G.out_edges("mentor_kim", data=True):
    print(f"  --{attrs['relation']}--> {neighbor}")


# === 실습 2: 'DS' 직무를 가진 멘토는 누구? ===
print("\n" + "=" * 60)
print("실습 2 — DS 직무를 가진 멘토 찾기")
print("=" * 60)
ds_mentors = get_incoming_by_relation(G, "DS", "has_role")
for m in ds_mentors:
    name = G.nodes[m].get("name", m)
    print(f"  {m} ({name})")


# === 실습 3: '비전공자_전환' trait을 가진 멘토? ===
print("\n" + "=" * 60)
print("실습 3 — 비전공자 전환 경험 있는 멘토")
print("=" * 60)
trait_mentors = get_incoming_by_relation(G, "비전공자_전환", "has_trait")
for m in trait_mentors:
    name = G.nodes[m].get("name", m)
    print(f"  {m} ({name})")


# === 실습 4: 가상 멘티 ─ 매칭 점수 계산 ===
print("\n" + "=" * 60)
print("실습 4 — 가상 멘티 매칭")
print("=" * 60)

# 멘티 정보 (Bridge 형태로)
mentee = {
    "role_target": "DS",
    "path_traits": ["비전공자_전환", "사회과학_출신"],
    "topic_tags": ["학습_로드맵", "포트폴리오"],
}

print(f"멘티 정보: {mentee}\n")

# 모든 멘토에 대해 점수 계산
all_mentors = get_nodes_by_type(G, "Mentor")
scores = []

for mentor in all_mentors:
    # role 일치
    mentor_roles = get_neighbors_by_relation(G, mentor, "has_role")
    role_match = mentee["role_target"] in mentor_roles
    
    # trait 교집합
    mentor_traits = set(get_neighbors_by_relation(G, mentor, "has_trait"))
    trait_overlap = len(set(mentee["path_traits"]) & mentor_traits)
    
    # topic 교집합
    mentor_topics = set(get_neighbors_by_relation(G, mentor, "can_advise"))
    topic_overlap = len(set(mentee["topic_tags"]) & mentor_topics)
    
    # 점수 — Phase 0에서 정한 가중치 적용
    score = (
        (0.30 if role_match else 0)
        + 0.30 * (trait_overlap / max(len(mentee["path_traits"]), 1))
        + 0.25 * (topic_overlap / max(len(mentee["topic_tags"]), 1))
    )
    
    scores.append((mentor, score, role_match, trait_overlap, topic_overlap))

# 점수 순으로 정렬
scores.sort(key=lambda x: x[1], reverse=True)

print(f"{'멘토':<20} {'점수':<8} {'role일치':<10} {'trait일치':<10} {'topic일치':<10}")
print("-" * 60)
for mentor, score, role_m, trait_n, topic_n in scores:
    name = G.nodes[mentor].get("name", mentor)
    print(f"{mentor} ({name})  {score:.3f}    {'✓' if role_m else '✗'}          {trait_n}          {topic_n}")