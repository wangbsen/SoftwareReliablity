import tkinter as tk
from tkinter import messagebox
from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import reduce  # 导入 reduce 函数
# 故障树节点类
class FaultTreeNode:
    def __init__(self, name, probability=0.0):
        self.name = name
        self.probability = probability
        self.children = []
        self.type = 'basic'  # 'basic', 'and', 'or', 'top'
    def add_child(self, child):
        self.children.append(child)
    def is_leaf(self):
        return len(self.children) == 0
# 构建故障树
def build_fault_tree():
    X1 = FaultTreeNode("X1", 0.5)
    X2 = FaultTreeNode("X2", 0.2)
    X3 = FaultTreeNode("X3", 0.5)
    P1 = FaultTreeNode("P1 (and)", 0.0)
    P1.type = 'and'
    P1.add_child(X2)
    P1.add_child(X3)
    TopEvent = FaultTreeNode("T (Top Event)", 0.0)
    TopEvent.type = 'top'
    P2 = FaultTreeNode("P2 (or)", 0.0)
    P2.type = 'or'
    P2.add_child(X1)
    P2.add_child(P1)
    TopEvent.add_child(P2)
    return TopEvent
# 下行法计算最小割集
def bottom_up_min_cut_sets(node):
    if node.is_leaf():
        return [[node.name]]
    if node.type == 'and':
        child_cut_sets = [bottom_up_min_cut_sets(child) for child in node.children]
        combined_cut_sets = []
        for cut_set_combination in product(*child_cut_sets):
            combined_cut_sets.append(list(set.union(*map(set, cut_set_combination))))
        return combined_cut_sets
    if node.type == 'or':
        return sum([bottom_up_min_cut_sets(child) for child in node.children], [])
    if node.type == 'top':
        if len(node.children) != 1:
            raise ValueError("顶事件必须只有一个孩子。")
        return bottom_up_min_cut_sets(node.children[0])
    raise ValueError("不支持的门类型")
# 计算最小径集
def bottom_up_min_path_sets(node):
    if node.is_leaf():
        return [[node.name]]
    if node.type == 'or':
        child_path_sets = [bottom_up_min_path_sets(child) for child in node.children]
        combined_path_sets = []
        for path_set_combination in product(*child_path_sets):
            combined_path_sets.append(list(set.union(*map(set, path_set_combination))))
        return combined_path_sets
    if node.type == 'and':
        return sum([bottom_up_min_path_sets(child) for child in node.children], [])
    if node.type == 'top':
        if len(node.children) != 1:
            raise ValueError("顶事件必须只有一个孩子。")
        return bottom_up_min_path_sets(node.children[0])
    raise ValueError("不支持的门类型。")
# 计算结构重要度的方法
def calculate_structural_importance(root, root_dict):
    basic_events = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf() and node.type == 'basic':
            basic_events.add(node.name)
        else:
            stack.extend(node.children)
    n = len(basic_events)
    importance = {event: 0 for event in basic_events}
    basic_events_list = list(basic_events)
    for i, event in enumerate(basic_events_list):
        for state_combination in product([0, 1], repeat=n - 1):
            state_vector_0 = list(state_combination[:i]) + [0] + list(state_combination[i:])
            state_vector_1 = list(state_combination[:i]) + [1] + list(state_combination[i:])
            phi_0 = evaluate_structure_function(root, state_vector_0, basic_events_list, root_dict)
            phi_1 = evaluate_structure_function(root, state_vector_1, basic_events_list, root_dict)
            importance[event] += phi_1 - phi_0
    # 归一化
    for event in importance:
        importance[event] /= 2 ** (n - 1)
    return importance
# 评估结构函数值的辅助函数
def evaluate_structure_function(node, state_vector, basic_events_list, root_dict):
    if node.is_leaf():
        index = basic_events_list.index(node.name)
        return state_vector[index]
    children_values = [evaluate_structure_function(child, state_vector, basic_events_list, root_dict) for child in
                       node.children]
    if node.type == 'and':
        return reduce(lambda x, y: x * y, children_values, 1.0)
    elif node.type == 'or':
        return 1 - reduce(lambda x, y: x * (1 - y), children_values, 1.0)
    elif node.type == 'top':
        return children_values[0] if len(children_values) == 1 else None
    else:
        raise ValueError("不支持的门类型。")
# 定量分析 - 计算顶事件概率
def calculate_top_event_probability(node):
    if node.is_leaf():
        return node.probability
    children_probabilities = [calculate_top_event_probability(child) for child in node.children]

    if node.type == 'and':
        probability = 1.0
        for prob in children_probabilities:
            probability *= prob
        return probability
    elif node.type == 'or':
        probability = 1.0
        for prob in children_probabilities:
            probability *= (1 - prob)
        return 1 - probability
    elif node.type == 'top':
        if len(node.children) != 1:
            raise ValueError("顶事件必须只有一个孩子。")
        return calculate_top_event_probability(node.children[0])
    else:
        raise ValueError("不支持的门类型。")
# 可视化故障树
def draw_fault_tree(node, G=None, pos=None, level=0, node_dict=None):
    if G is None: G = nx.DiGraph()
    if pos is None: pos = {}
    if node_dict is None: node_dict = {}
    if not node.is_leaf():
        for child in node.children:
            G.add_edge(node.name, child.name)
            draw_fault_tree(child, G, pos, level + 1, node_dict)
    pos[node.name] = (level, len(G.nodes()) - 1)
    node_dict[node.name] = node  # 存储节点对象
    return G, pos, node_dict
def visualize_fault_tree(root, canvas, root_dict):
    G, pos, _ = draw_fault_tree(root, node_dict=root_dict)
    fig, ax = plt.subplots()
    # 根据节点类型设置不同的颜色和形状
    colors = {'basic': 'lightblue', 'and': 'lightgreen', 'or': 'lightcoral', 'top': 'gold'}
    shapes = {'basic': 'o', 'and': 's', 'or': 'd', 'top': '^'}  # 'o' 圆形, 's' 正方形, 'd' 菱形, '^' 三角形
    # 绘制节点
    for node in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=colors[root_dict[node].type],
                               node_shape=shapes[root_dict[node].type],
                               node_size=1000 if root_dict[node].is_leaf() else 2000, ax=ax)
    # 绘制边
    nx.draw_networkx_edges(G, pos, ax=ax)
    # 添加节点标签
    nx.draw_networkx_labels(G, pos, ax=ax)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    ax.set_title("故障树")
    canvas.figure = fig
    canvas.draw()
# 计算概率重要度
def calculate_probabilistic_importance(root, root_dict):
    basic_events = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf() and node.type == 'basic':
            basic_events.add(node.name)
        else:
            stack.extend(node.children)
    importance = {}
    original_probs = {event: root_dict[event].probability for event in basic_events}
    for event_name in basic_events:
        event = root_dict[event_name]
        # 保存事件的原始概率，以便稍后恢复
        original_prob = event.probability
        # 增加基础事件故障概率的微小增量
        event.probability += 0.0001  # 微小增量
        prob_with_increase = calculate_top_event_probability(root)# 增加后计算系统故障概率
        # 恢复基础事件的原始故障概率
        event.probability = original_prob
        # 恢复后计算系统故障概率
        prob_original = calculate_top_event_probability(root)
        # 计算基础事件的概率重要度，这里的差异大致等于 ΔF_S / ΔF_i
        importance[event_name] = (prob_with_increase - prob_original) / 0.0001
        # 恢复所有事件的原始概率
        for ev, prob in original_probs.items():
            root_dict[ev].probability = prob
    return importance
# 计算关键重要度
def calculate_criticality_importance(root, root_dict):
    # 首先计算顶事件的整体失效概率 F_S
    F_S = calculate_top_event_probability(root)
    if F_S == 0:
        raise ValueError("顶事件概率为零，无法计算关键性重要性.")
    # 然后计算每个基础事件的概率重要度 Δg_i(t)
    probabilistic_importance = calculate_probabilistic_importance(root, root_dict)
    # 最后计算每个基础事件的关键重要度 I_i^CR(t)
    criticality_importance = {}
    for event_name in probabilistic_importance.keys():
        event = root_dict[event_name]
        F_i = event.probability  # 获取基础事件的失效概率 F_i(t)
        delta_g_i = probabilistic_importance[event_name]  # 获取基础事件的概率重要度 Δg_i(t)
        # 计算关键重要度
        criticality_importance[event_name] = (F_i / F_S) * delta_g_i
    return criticality_importance
# GUI部分
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("故障树分析工具")
        self.geometry("800x600")
        self.root = build_fault_tree()
        self.canvas = FigureCanvasTkAgg(plt.figure(), master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        # 创建字典以存储所有节点以便快速访问
        self.root_dict = {}
        draw_fault_tree(self.root, node_dict=self.root_dict)
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Button(btn_frame, text="可视化故障树", command=self.visualize).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(btn_frame, text="定性分析", command=self.qualitative_analysis).pack(side=tk.LEFT, padx=5,pady=5)
        tk.Button(btn_frame, text="顶事件概率", command=self.quantitative_analysis).pack(side=tk.LEFT,padx=5, pady=5)
        tk.Button(btn_frame, text="结构重要度", command=self.structural_importance).pack(side=tk.LEFT,padx=5, pady=5)
        tk.Button(btn_frame, text="概率重要度", command=self.probabilistic_importance).pack(side=tk.LEFT,padx=5,pady=5)
        tk.Button(btn_frame, text="关键重要度", command=self.criticality_importance).pack(side=tk.LEFT,padx=5, pady=5)
    def visualize(self):
        visualize_fault_tree(self.root, self.canvas, self.root_dict)
    def qualitative_analysis(self):
        min_cut_sets = bottom_up_min_cut_sets(self.root)
        min_path_sets = bottom_up_min_path_sets(self.root)
        messagebox.showinfo("最小割集", f"最小割集: {min_cut_sets}")
        messagebox.showinfo("最小径集", f"最小径集: {min_path_sets}")
    def quantitative_analysis(self):
        prob = calculate_top_event_probability(self.root)
        messagebox.showinfo("顶事件概率", f"顶事件概率: {prob:.4f}")
    def structural_importance(self):
        importance = calculate_structural_importance(self.root, self.root_dict)
        message = "结构重要度:\n"
        for event, imp in importance.items():
            message += f"{event}: {imp:.4f}\n"
        messagebox.showinfo("结构重要度", message)
    def probabilistic_importance(self):
        importance = calculate_probabilistic_importance(self.root, self.root_dict)
        message = "概率重要度:\n"
        for event, imp in importance.items():
            message += f"{event}: {imp:.4f}\n"
        messagebox.showinfo("概率重要度", message)
    def criticality_importance(self):
        importance = calculate_criticality_importance(self.root, self.root_dict)
        message = "关键重要度:\n"
        for event, imp in importance.items():
            message += f"{event}: {imp:.4f}\n"
        messagebox.showinfo("关键重要度", message)
if __name__ == "__main__":
    app = Application()
    app.mainloop()