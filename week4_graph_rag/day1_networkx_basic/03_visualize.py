"""그래프를 그림으로 — 시각적 직관 얻기"""

import networkx as nx
import matplotlib.pyplot as plt

# 폰트 깨짐 방지 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# (위 02_traversal.py의 그래프를 그대로 다시 만들기 — 또는 import)
G = nx.MultiDiGraph()
# ... (위 그래프 코드 복사) ...
G.add_node("mentor_kim", type="Mentor", name="김OO")
G.add_node("mentor_park", type="Mentor", name="박OO")
G.add_node("mentor_lee", type="Mentor", name="이OO")
G.add_node("DS", type="Role", label="DS")
G.add_node("MLE", type="Role", label="MLE")
G.add_node("BE", type="Role", label="BE")
G.add_node("비전공자_전환", type="Trait")
G.add_node("사회과학_출신", type="Trait")
G.add_node("이직_경험", type="Trait")
G.add_node("학습_로드맵", type="Topic")
G.add_node("포트폴리오", type="Topic")
G.add_node("면접_준비", type="Topic")

G.add_edge("mentor_kim", "DS")
G.add_edge("mentor_kim", "비전공자_전환")
G.add_edge("mentor_kim", "사회과학_출신")
G.add_edge("mentor_kim", "학습_로드맵")
G.add_edge("mentor_kim", "포트폴리오")
G.add_edge("mentor_park", "MLE")
G.add_edge("mentor_park", "이직_경험")
G.add_edge("mentor_park", "포트폴리오")
G.add_edge("mentor_park", "면접_준비")
G.add_edge("mentor_lee", "BE")
G.add_edge("mentor_lee", "이직_경험")
G.add_edge("mentor_lee", "면접_준비")

# === 노드 타입별 색깔 ===
color_map = {
    "Mentor": "lightblue",
    "Role": "lightgreen",
    "Trait": "lightyellow",
    "Topic": "lightpink",
}
node_colors = [color_map.get(G.nodes[n].get("type"), "gray") for n in G.nodes]

# === 그리기 ===
pos = nx.spring_layout(G, k=1.5, seed=42)
plt.figure(figsize=(12, 8))
nx.draw(G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2500,
        font_size=9,
        font_family='Malgun Gothic',
        arrows=True,
        arrowsize=15,
        edge_color='gray')

plt.title("멘토 그래프 — Mentor·Role·Trait·Topic", fontsize=14)
plt.tight_layout()
plt.savefig('week4_graph_rag/day1_networkx_basic/graph.png', dpi=100)
plt.show()
print("그래프를 graph.png로 저장했어.")