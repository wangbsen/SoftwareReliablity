from collections import deque, defaultdict
from typing import List, Tuple, Dict

class SAPNNodeType:
    PLACE = 'Place'
    TRANSITION = 'Transition'

class SAPNNode:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

class SAPNGraph:
    def __init__(self, nodes: List[SAPNNode], edges: List[Tuple[str, str]], S: List[str], EN: List[str],
                 PT: Dict[str, float], RC: Dict[str, float], RL: Dict[str, float], RT: Dict[str, float]):
        self.nodes = nodes
        self.S = S
        self.EN = EN
        self.PT = PT
        self.RC = RC
        self.RL = RL
        self.RT = RT
        self.g = defaultdict(list)
        self.name_to_index = {node.name: i for i, node in enumerate(nodes)}

        for u, v in edges:
            if u in self.name_to_index and v in self.name_to_index:
                self.g[self.name_to_index[u]].append(self.name_to_index[v])
            else:
                raise ValueError(f"Invalid edge: {u} -> {v}")

    def get_paths(self) -> List[List[SAPNNode]]:
        ans = []
        start_node = self.nodes[self.name_to_index[self.S[0]]]
        que = deque([(start_node, [start_node])])

        while que:
            node, path = que.popleft()
            if node.name in self.EN:
                ans.append(path)
                continue
            l = self.g[self.name_to_index[node.name]]
            if not l:
                continue
            if len(l) == 1:
                next_node = self.nodes[l[0]]
                if path.count(next_node) >= 2:
                    continue
                path.append(next_node)
                que.append((next_node, list(path)))
                continue
            for idx in l:
                next_node = self.nodes[idx]
                if path.count(next_node) >= 2:
                    continue
                p = list(path) + [next_node]
                que.append((next_node, p))
        return ans

    def get_path_probability(self, path: List[SAPNNode]) -> float:
        ans = 1.0
        for node in path:
            if node.type == SAPNNodeType.TRANSITION:
                ans *= self.PT[node.name]
        return ans

    def get_path_reliability(self, path: List[SAPNNode]) -> float:
        ans = 1.0
        for node in path:
            if node.name not in self.S and node.name not in self.EN:
                if node.type == SAPNNodeType.PLACE:
                    ans *= self.RC[node.name.replace('P', 'C')]
                else:
                    ans *= self.RT[node.name]
                    ans *= self.RL[node.name.replace('T', 'L')]
        return ans

    def get_system_reliability(self, path_probabilities: List[float], path_reliabilities: List[float]) -> float:
        assert len(path_probabilities) == len(path_reliabilities)
        total_prob = sum(path_probabilities)
        reliability = sum(p * r for p, r in zip(path_probabilities, path_reliabilities)) / total_prob
        return reliability

def convert_place_names(path: List[SAPNNode]) -> List[str]:
    """Convert place names in the path from 'P' to 'C'."""
    return [node.name.replace('P', 'C') if node.type == SAPNNodeType.PLACE else node.name for node in path]

if __name__ == "__main__":
    nodes = [
        SAPNNode("S", SAPNNodeType.PLACE),
        SAPNNode("EN", SAPNNodeType.PLACE),
        SAPNNode("P1", SAPNNodeType.PLACE),
        SAPNNode("P2", SAPNNodeType.PLACE),
        SAPNNode("P3", SAPNNodeType.PLACE),
        SAPNNode("P4", SAPNNodeType.PLACE),
        SAPNNode("P5", SAPNNodeType.PLACE),
        SAPNNode("P6", SAPNNodeType.PLACE),
        SAPNNode("P7", SAPNNodeType.PLACE),
        SAPNNode("P8", SAPNNodeType.PLACE),
        SAPNNode("P9", SAPNNodeType.PLACE),
        SAPNNode("T1", SAPNNodeType.TRANSITION),
        SAPNNode("T2", SAPNNodeType.TRANSITION),
        SAPNNode("T3", SAPNNodeType.TRANSITION),
        SAPNNode("T4", SAPNNodeType.TRANSITION),
        SAPNNode("T5", SAPNNodeType.TRANSITION),
        SAPNNode("T6", SAPNNodeType.TRANSITION),
        SAPNNode("T7", SAPNNodeType.TRANSITION),
        SAPNNode("T8", SAPNNodeType.TRANSITION),
        SAPNNode("T9", SAPNNodeType.TRANSITION),
        SAPNNode("T10", SAPNNodeType.TRANSITION),
    ]
    edges = [
        ("S", "P1"),
        ("P1", "T1"),
        ("T1", "P2"),
        ("P2", "T2"),
        ("T2", "P3"),
        ("P3", "T3"),
        ("T3", "P4"),
        ("P4", "T4"),
        ("T4", "P5"),
        ("P5", "T5"),
        ("T5", "P6"),
        ("P6", "T6"),
        ("T6", "P7"),
        ("P7", "T7"),
        ("T7", "P8"),
        ("P8", "T8"),
        ("T8", "P2"),
        ("P4", "T10"),
        ("T10", "P9"),
        ("P9", "EN"),
        ("P7", "T9"),
        ("T9", "P9"),
    ]
    PT = {
        "T1": 1,
        "T2": 0.99,
        "T3": 0.98,
        "T4": 0.80,
        "T5": 1.0,
        "T6": 1.0,
        "T7": 0.30,
        "T8": 0.98,
        "T9": 0.98,
        "T10": 0.20
    }
    RC = {
        "C1": 1,
        "C2": 0.99,
        "C3": 0.98,
        "C4": 1,
        "C5": 0.99,
        "C6": 0.99,
        "C7": 1,
        "C8": 0.98,
        "C9": 1,
    }
    RL = {
        "L1": 0.99,
        "L2": 1,
        "L3": 1,
        "L4": 0.98,
        "L5": 1,
        "L6": 0.99,
        "L7": 0.99,
        "L8": 1,
        "L9": 0.98,
        "L10": 1
    }
    RT = {
        "T1": 1,
        "T2": 0.99,
        "T3": 1,
        "T4": 0.98,
        "T5": 0.99,
        "T6": 1,
        "T7": 0.98,
        "T8": 0.98,
        "T9": 0.99,
        "T10": 1
    }

    graph = SAPNGraph(nodes, edges, ["S"], ["EN"], PT, RC, RL, RT)
    paths = graph.get_paths()
    path_probabilities = []
    path_reliabilities = []

    for path in paths:
        probability = graph.get_path_probability(path)
        reliability = graph.get_path_reliability(path)
        path_probabilities.append(probability)
        path_reliabilities.append(reliability)

        # 转换路径中的位置名称，并输出路径信息
        converted_path_names = ' -> '.join(convert_place_names(path))
        print(f"路径: {converted_path_names}")
        print(f"迁移概率: {probability:.3f}")
        print(f"可靠度: {reliability:.3f}\n")

    system_reliability = graph.get_system_reliability(path_probabilities, path_reliabilities)
    print(f"系统可靠度: {system_reliability:.3f}")