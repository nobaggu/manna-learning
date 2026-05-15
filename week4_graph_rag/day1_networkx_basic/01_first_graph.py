"""NetworkX 첫 그래프 — 멘토 1명 + 관계 노드들"""

import networkx as nx

# === 그래프 객체 생성 ===
G = nx.MultiDiGraph()

# === 노드 추가 (type 속성으로 종류 구분) ===
G.add_node("mentor_kim", type="Mentor", name="김OO", years=5)
G.add_node("DS", type="Role", label="데이터 사이언티스트")
G.add_node("BE", type="Role", label="백엔드 개발자")
G.add_node("핀테크", type="Domain")
G.add_node("Python", type="Skill")
G.add_node("SQL", type="Skill")
G.add_node("비전공자_전환", type="Trait")
G.add_node("학습_로드맵", type="Topic")
G.add_node("포트폴리오", type="Topic")

# === 엣지 추가 (relation 속성으로 의미 구분) ===
G.add_edge("mentor_kim", "DS", relation="has_role")
G.add_edge("mentor_kim", "핀테크", relation="works_in")
G.add_edge("mentor_kim", "Python", relation="has_skill")
G.add_edge("mentor_kim", "SQL", relation="has_skill")
G.add_edge("mentor_kim", "비전공자_전환", relation="has_trait")
G.add_edge("mentor_kim", "학습_로드맵", relation="can_advise")
G.add_edge("mentor_kim", "포트폴리오", relation="can_advise")

# Role 간 인접 관계
G.add_edge("DS", "BE", relation="adjacent_to")

# === 기본 정보 출력 ===
print("=" * 60)
print("그래프 기본 정보")
print("=" * 60)
print(f"노드 수: {G.number_of_nodes()}")
print(f"엣지 수: {G.number_of_edges()}")

print("\n[모든 노드]")
for node, attrs in G.nodes(data=True):
    print(f"  {node} {attrs}")

print("\n[모든 엣지]")
for u, v, attrs in G.edges(data=True):
    print(f"  {u} --{attrs['relation']}--> {v}")