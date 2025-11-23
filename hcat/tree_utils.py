"""
Hierarchical Tree utilities for navigating and computing distances
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random


@dataclass
class TreeNode:
    """Represents a node in the hierarchical tree"""
    id: int
    type: str  # 'leaf' or 'cluster'
    name: Optional[str] = None
    distance: Optional[float] = None
    count: int = 1
    children: List['TreeNode'] = None
    parent: Optional['TreeNode'] = None
    depth: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class HierarchicalTree:
    """Parse and navigate hierarchical tree structure"""

    def __init__(self, tree_data: Dict):
        """
        Load tree from dictionary or JSON file

        Args:
            tree_data: Either dict with 'hierarchy' key or path to JSON file (str)
        """
        if isinstance(tree_data, str):
            with open(tree_data, 'r') as f:
                tree_data = json.load(f)

        self.root = self._parse_tree(tree_data['hierarchy'])
        self.leaves = {}  # Map leaf name to TreeNode
        self.all_nodes = {}  # Map node id to TreeNode
        self._build_indices(self.root)
        self._compute_depths(self.root, 0)

        # Compute max depth for normalization
        self.max_depth = max(node.depth for node in self.all_nodes.values())

    def _parse_tree(self, node_dict: Dict, parent=None) -> TreeNode:
        """Recursively parse tree structure"""
        node = TreeNode(
            id=node_dict['id'],
            type=node_dict['type'],
            name=node_dict.get('name'),
            distance=node_dict.get('distance'),
            count=node_dict.get('count', 1),
            parent=parent
        )

        if 'children' in node_dict:
            node.children = [
                self._parse_tree(child, parent=node)
                for child in node_dict['children']
            ]

        return node

    def _build_indices(self, node: TreeNode):
        """Build lookup indices for leaves and all nodes"""
        self.all_nodes[node.id] = node

        if node.type == 'leaf':
            self.leaves[node.name] = node

        for child in node.children:
            self._build_indices(child)

    def _compute_depths(self, node: TreeNode, depth: int):
        """Compute depth for each node"""
        node.depth = depth
        for child in node.children:
            self._compute_depths(child, depth + 1)

    def find_lca(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        """Find lowest common ancestor of two nodes"""
        # Get ancestors of node1
        ancestors1 = set()
        current = node1
        while current is not None:
            ancestors1.add(current.id)
            current = current.parent

        # Find first common ancestor for node2
        current = node2
        while current is not None:
            if current.id in ancestors1:
                return current
            current = current.parent

        return self.root

    def tree_distance(self, node1: TreeNode, node2: TreeNode) -> float:
        """
        Compute hierarchical distance between two nodes.
        Returns depth difference from LCA (higher = farther apart).
        """
        lca = self.find_lca(node1, node2)
        # Distance = how far down from LCA to reach the nodes
        max_depth = max(node1.depth, node2.depth)
        return max_depth - lca.depth

    def get_leaves_in_subtree(self, node: TreeNode) -> List[TreeNode]:
        """Get all leaf nodes under a given node"""
        if node.type == 'leaf':
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self.get_leaves_in_subtree(child))
        return leaves

    def sample_triplet_with_distances(
        self,
        anchor_name: str,
        strategy: str = 'hierarchical'
    ) -> Tuple[str, str, float, float]:
        """
        Sample positive and negative with their tree distances.

        Returns:
            (positive_name, negative_name, distance_to_positive, distance_to_negative)
        """
        anchor_node = self.leaves[anchor_name]
        all_leaves = [l for l in self.leaves.values() if l.name != anchor_name]

        if strategy == 'hierarchical':
            # Compute all distances
            distances = [(l, self.tree_distance(anchor_node, l)) for l in all_leaves]
            distances.sort(key=lambda x: x[1])

            # Positive: closest leaves
            close_threshold = distances[0][1] if distances else 0
            positive_candidates = [l for l, d in distances if d <= close_threshold + 1]
            positive = random.choice(positive_candidates) if positive_candidates else distances[0][0]

            # Negative: distant leaves
            far_threshold = distances[-1][1] if distances else 0
            negative_candidates = [l for l, d in distances if d >= far_threshold - 1]
            negative = random.choice(negative_candidates) if negative_candidates else distances[-1][0]

        else:  # sibling strategy
            # Positive: same parent cluster
            parent = anchor_node.parent
            if parent and len(parent.children) > 1:
                siblings = self.get_leaves_in_subtree(parent)
                siblings = [s for s in siblings if s.name != anchor_name]
                if siblings:
                    positive = random.choice(siblings)
                else:
                    positive = random.choice(all_leaves)
            else:
                positive = random.choice(all_leaves)

            # Negative: distant cluster
            current = anchor_node.parent
            for _ in range(3):  # Go up 3 levels
                if current and current.parent:
                    current = current.parent

            anchor_subtree = set(l.name for l in self.get_leaves_in_subtree(current))
            distant_leaves = [l for l in all_leaves if l.name not in anchor_subtree]

            if distant_leaves:
                negative = random.choice(distant_leaves)
            else:
                negative = random.choice([l for l in all_leaves if l.name != positive.name])

        # Compute distances
        dist_pos = self.tree_distance(anchor_node, positive)
        dist_neg = self.tree_distance(anchor_node, negative)

        return positive.name, negative.name, dist_pos, dist_neg