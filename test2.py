import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Set
from itertools import chain, combinations
from graphviz import Digraph


class EventNodeType:
    BASIC = "Basic"
    AND = "And"
    OR = "Or"


class EventNode:
    def __init__(self, name: str, node_type: EventNodeType):
        self.name = name
        self.type = node_type
        self.children: List['EventNode'] = []


class FaultTree:
    def __init__(self, root: EventNode):
        self.root = root

    def get_minimal_cut_sets(self) -> List[Set[str]]:
        """Compute the minimal cut sets for the fault tree."""
        cut_sets = self._get_minimal_cut_sets(self.root)
        self._simplify_sets(cut_sets)
        return cut_sets

    def _get_minimal_cut_sets(self, node: EventNode) -> List[Set[str]]:
        if node.type == EventNodeType.BASIC:
            return [{node.name}]
        elif node.type == EventNodeType.AND:
            sets = [self._get_minimal_cut_sets(child) for child in node.children]
            return [set.union(*combination) for combination in zip(*sets)]
        elif node.type == EventNodeType.OR:
            return [cut_set for child in node.children for cut_set in self._get_minimal_cut_sets(child)]
        else:
            raise ValueError("Unknown EventNodeType")

    @staticmethod
    def _simplify_sets(sets: List[Set[str]]):
        sets.sort(key=len)
        i = 0
        while i < len(sets):
            j = i + 1
            while j < len(sets):
                if sets[i] <= sets[j]:
                    sets.pop(j)
                elif sets[j] <= sets[i]:
                    sets.pop(i)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1

    @staticmethod
    def calculate_top_event_probability(cut_sets: List[Set[str]], probabilities: Dict[str, float]) -> float:
        """Calculate the top event probability using the inclusion-exclusion principle."""
        n = len(cut_sets)
        probability = 0.0

        for bitmask in range(1, 1 << n):
            selected_events = set()
            for i in range(n):
                if bitmask & (1 << i):
                    selected_events |= cut_sets[i]

            term_probability = 1.0
            for event in selected_events:
                term_probability *= probabilities.get(event, 0.0)

            if bin(bitmask).count("1") % 2 == 1:
                probability += term_probability
            else:
                probability -= term_probability

        return probability


class FaultTreeGUI:
    def __init__(self, root):
        self.root = root
        self.nodes = {}
        self.current_node = None
        self.basic_event_probabilities = {}

        self.setup_gui()

    def setup_gui(self):
        self.root.title("Fault Tree Builder")
        self.root.geometry("800x600")

        # Tree frame
        self.tree_frame = ttk.Frame(self.root)
        self.tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_select)

        tree_scroll = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        # Controls frame
        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(self.controls_frame, text="Node Name:").pack(pady=5)
        self.node_name_entry = ttk.Entry(self.controls_frame)
        self.node_name_entry.pack(pady=5)

        ttk.Label(self.controls_frame, text="Node Type:").pack(pady=5)
        self.node_type_var = tk.StringVar(value=EventNodeType.BASIC)
        ttk.Combobox(self.controls_frame, textvariable=self.node_type_var, values=[
            EventNodeType.BASIC, EventNodeType.AND, EventNodeType.OR]).pack(pady=5)

        self.add_node_button = ttk.Button(self.controls_frame, text="Add Node", command=self.add_node)
        self.add_node_button.pack(pady=5)

        ttk.Label(self.controls_frame, text="Probability (Basic Events):").pack(pady=5)
        self.probability_entry = ttk.Entry(self.controls_frame)
        self.probability_entry.pack(pady=5)

        self.link_node_button = ttk.Button(self.controls_frame, text="Link to Parent", command=self.link_node)
        self.link_node_button.pack(pady=5)

        self.calculate_button = ttk.Button(self.controls_frame, text="Calculate Results", command=self.calculate_results)
        self.calculate_button.pack(pady=20)

        self.visualize_button = ttk.Button(self.controls_frame, text="Visualize Tree", command=self.visualize_tree)
        self.visualize_button.pack(pady=20)

    def add_node(self):
        name = self.node_name_entry.get().strip()
        node_type = self.node_type_var.get()

        if not name:
            messagebox.showerror("Error", "Node name cannot be empty.")
            return

        if name in self.nodes:
            messagebox.showerror("Error", "Node name must be unique.")
            return

        node = EventNode(name, node_type)
        self.nodes[name] = node

        if node_type == EventNodeType.BASIC:
            probability = self.probability_entry.get().strip()
            try:
                self.basic_event_probabilities[name] = float(probability)
            except ValueError:
                self.basic_event_probabilities[name] = 0.0

        self.tree.insert("", tk.END, iid=name, text=f"{name} ({node_type})")
        messagebox.showinfo("Success", f"Node '{name}' added successfully.")

    def link_node(self):
        if not self.current_node:
            messagebox.showerror("Error", "No parent node selected.")
            return

        child_name = self.node_name_entry.get().strip()
        if not child_name or child_name not in self.nodes:
            messagebox.showerror("Error", "Invalid child node name.")
            return

        parent_node = self.nodes[self.current_node]
        child_node = self.nodes[child_name]

        parent_node.children.append(child_node)
        self.tree.insert(self.current_node, tk.END, iid=f"{self.current_node}->{child_name}",
                         text=f"{child_name} ({child_node.type})")
        messagebox.showinfo("Success", f"Node '{child_name}' linked to '{self.current_node}'.")

    def on_tree_select(self, event):
        selected_item = self.tree.focus()
        if selected_item:
            self.current_node = selected_item

    def calculate_results(self):
        if "Top Event" not in self.nodes:
            messagebox.showerror("Error", "You must have a node named 'Top Event'.")
            return

        root_node = self.nodes["Top Event"]
        fault_tree = FaultTree(root_node)
        cut_sets = fault_tree.get_minimal_cut_sets()
        probability = FaultTree.calculate_top_event_probability(cut_sets, self.basic_event_probabilities)

        results = f"Top Event Probability: {probability:.6f}\nMinimal Cut Sets: {cut_sets}"
        messagebox.showinfo("Calculation Results", results)

    def visualize_tree(self):
        if not self.nodes:
            messagebox.showerror("Error", "No nodes to visualize.")
            return

        if "Top Event" not in self.nodes:
            messagebox.showerror("Error", "You must have a node named 'Top Event'.")
            return

        root_node = self.nodes["Top Event"]
        graph = Digraph(format="png")
        self._add_to_graph(graph, root_node)
        output_file = "fault_tree_visual"
        graph.render(output_file, cleanup=True)
        messagebox.showinfo("Success", f"Tree visualized and saved as {output_file}.png")

    def _add_to_graph(self, graph, node, parent_id=None):
        node_id = id(node)
        label = f"{node.name}\n({node.type})"
        graph.node(str(node_id), label)

        if parent_id is not None:
            graph.edge(str(parent_id), str(node_id))

        for child in node.children:
            self._add_to_graph(graph, child, node_id)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaultTreeGUI(root)
    root.mainloop()
