import matplotlib.pyplot as plt

import torch as t
import networkx as nx

from typing import Tuple, List

import logging

log = logging.getLogger(__name__)


class GraphVisualization:
    def __init__(self):

        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    def parentOf(self, node):
        for edge in self.visual:
            # childs are always in [0]
            if node == edge[0]:
                return edge[1]
        return node

    def childOf(self, node):
        childs = []
        for edge in self.visual:
            # childs are always in [0]
            if node == edge[1]:
                childs.append(edge[1])
        return childs

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self, opt="max", ax=None):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        pos = None
        try:
            if opt == "min":
                pos = self.hierarchy_pos(G, min(min(self.visual)))
            elif opt == "max":
                pos = self.hierarchy_pos(G, max(max(self.visual)))
        except TypeError:
            log.warning(
                "Provided LCA is not a tree. Will use default graph style for visualization"
            )
            # raise RuntimeError
        nx.draw_networkx(G, pos, ax=ax)
        # plt.show()

    def lca2graph(self, lca):

        nodes = [
            i for i in range(lca.size(0))
        ]  # first start with all nodes available directly from the lca
        processed = (
            []
        )  # keeps track of all nodes which are somehow used to create an edge

        while lca.max() > 0:
            directPairs = list((lca == 1.0).nonzero())
            directPairs.sort(key=lambda dp: sum(dp))
            directPairs = directPairs[1::2]

            def convToPair(pair: t.Tensor) -> Tuple[int, int]:
                return (int(pair[0]), int(pair[1]))

            def getOverlap(
                edge: Tuple[int, int], ref: List[int]
            ) -> List[Tuple[int, int]]:
                return list(set(edge) & set(ref))

            def addNodeNotInSet(
                node: int, parent: int, ovSet: List[Tuple[int, int]], appendSet: bool
            ) -> List[Tuple[int, int]]:
                if node not in ovSet:
                    self.addEdge(node, parent)
                    if appendSet:
                        ovSet.append(node)
                return ovSet

            def addPairNotInSet(
                pair: Tuple[int, int],
                parent: int,
                ovSet: List[Tuple[int, int]],
                appendSet: bool = True,
            ) -> List[Tuple[int, int]]:
                if pair[0] == pair[1]:
                    return ovSet

                for node in pair:
                    ovSet = addNodeNotInSet(node, parent, ovSet, appendSet)
                return ovSet

            for tensor_pair in directPairs:
                pair = convToPair(tensor_pair)

                while True:
                    # this works since they are sorted by pair[0]
                    overlap = getOverlap(pair, processed)

                    # no overlap -> create new ancestor
                    if len(overlap) == 0:
                        # create a new parent and set ancestor accordingly
                        nodes.append(max(nodes) + 1)
                        ancestor = nodes[-1]

                        # found two new edged, cancel this subroutine and  process next pair
                        processed = addPairNotInSet(pair, ancestor, processed)
                        break

                    # only 1 common node -> set the ancestor to the parent of this overlap
                    elif len(overlap) == 1:
                        # overlap has only one element here
                        ancestor = self.parentOf(overlap[0])

                        generation = -1
                        while True:
                            if self.parentOf(ancestor) == ancestor:
                                break
                            ancestor = self.parentOf(ancestor)
                            generation -= 1

                        if lca[0][0] <= generation:
                            nodes.append(max(nodes) + 1)

                            addNodeNotInSet(ancestor, nodes[-1], processed, True)

                            ancestor = nodes[-1]

                        processed = addPairNotInSet(pair, ancestor, processed)
                        # found a new edge, cancel this subroutine and process next pair
                        break

                    # full overlap -> meaning they were both processed previously
                    else:
                        ancestor_a = self.parentOf(overlap[0])
                        while True:
                            if self.parentOf(ancestor_a) == ancestor_a:
                                break
                            ancestor_a = self.parentOf(ancestor_a)

                        ancestor_b = self.parentOf(overlap[1])
                        while True:
                            if self.parentOf(ancestor_b) == ancestor_b:
                                break
                            ancestor_b = self.parentOf(ancestor_b)

                        # cancel if they have the same parent
                        if ancestor_a == ancestor_b:
                            break
                        # cancel if they are already connected (cause this would happen again in the next round)
                        elif self.parentOf(ancestor_a) == ancestor_b:
                            break  # TODO check here if this break is ok

                        # overwrite edge by new set of parents
                        pair = (ancestor_a, ancestor_b)

                        # don't cancel here, process this set instead again (calc overlap...)

            lca -= 1
            if len(directPairs) == 0:
                if lca.max() == 0:
                    raise RuntimeError(f"Invalid LCAG detected. LCAG was {lca}")
                lca += t.diag(t.ones(lca.size(0), dtype=t.long))
            # processed = []

    def save(self, filename):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.save(filename)

    def hierarchy_pos(
        self, G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5
    ):
        return hierarchy_pos(G, root, width, vert_gap, vert_loc)


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
    the root will be found and used
    - if the tree is directed and this is given, then
    the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
    then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    def _hierarchy_pos(
        G,
        root,
        width=1.0,
        vert_gap=0.2,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
    ):
        """
        see hierarchy_pos docstring for most arguments from https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def testLca2Graph(self):
    """
                    +---+
                    | a |
                    ++-++
                    | |
            +-------+ +------+
            |                |
            +-+-+            +-+-+
            | b |            | c |
            +++++            ++-++
            |||              | |
        +----+|+----+     +---+ +---+
        |     |     |     |         |
    +-+-+ +-+-+ +-+-+ +-+-+     +-+-+
    | d | | e | | f | | g |     | h |
    +---+ +---+ +---+ +---+     +---+
    """
    example_1 = t.tensor(
        [
            # d  g  f  h  e
            [0, 2, 1, 2, 1],  # d
            [2, 0, 2, 1, 2],  # g
            [1, 2, 0, 2, 1],  # f
            [2, 1, 2, 0, 2],  # h
            [1, 2, 1, 2, 0],  # e
        ]
    )
    """ Testing with disjointed levels

                    +---+
                    | 7 |
                    ++-++
                    | |
            +-------+ +------+
            |                |
            +-+-+            +-+-+
            | 5 |            | 6 |
            +++++            ++-++
            |||              | |
        +----+|+----+     +---+ +---+
        |     |     |     |         |
    +-+-+ +-+-+ +-+-+ +-+-+     +-+-+
    | 0 | | 4 | | 2 | | 1 |     | 3 |
    +---+ +---+ +---+ +---+     +---+
    """
    example_2 = t.tensor(
        [
            # d  g  f  h  e
            [0, 5, 2, 5, 2],  # d
            [5, 0, 5, 2, 5],  # g
            [2, 5, 0, 5, 2],  # f
            [5, 2, 5, 0, 5],  # h
            [2, 5, 2, 5, 0],  # e
        ]
    )
    """
                +---+
                | 8 |
                ++-++
                | |
            +------+ +------+
            |               |
            |             +-+-+
            |             | 7 |
            |             ++-++
            |              | |
            |           +--+ +---+
            |           |        |
        +-+-+       +-+-+      |
        | 5 |       | 6 |      |
        ++-++       ++-++      |
        | |         | |       |
        +-+ +-+     +-+ +-+     |
        |     |     |     |     |
    +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
    | 0 | | 1 | | 2 | | 3 | | 4 |
    +---+ +---+ +---+ +---+ +---+
    """
    example_3 = t.tensor(
        [
            # e  f  g  h  i
            [0, 1, 3, 3, 3],  # e
            [1, 0, 3, 3, 3],  # f
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0],  # i
        ]
    )
    """
                        +---+
                        | 7 |
                        ++-++
                        | |
                    +----+ +----+
                    |           |
                +-+-+         |
                | 6 |         |
                +++++         |
                |||          |
        +-------+|+----+     |
        |        |     |     |
        +---+      |     |     |
        | 5 |      |     |     |
        ++-++      |     |     |
        | |       |     |     |
        +-+ +-+     |     |     |
        |     |     |     |     |
    +-+-+ +-+-+ +-+-+ +-+-+ +-+-+
    | 0 | | 1 | | 2 | | 3 | | 4 |
    +---+ +---+ +---+ +---+ +---+
    """
    example_4 = t.tensor(
        [
            # g  h  d  f  c
            [0, 1, 2, 2, 3],  # g
            [1, 0, 2, 2, 3],  # h
            [2, 2, 0, 2, 3],  # d
            [2, 2, 2, 0, 3],  # f
            [3, 3, 3, 3, 0],  # c
        ]
    )

    example_5 = t.tensor([[0, 2, 1, 2], [2, 0, 2, 1], [1, 2, 0, 2], [2, 1, 2, 0]])
    examples = [example_1, example_2, example_3, example_4, example_5]

    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    fig.tight_layout()
    it = 0
    for lcag_ref in examples:
        graph_ref = GraphVisualization()
        try:
            self.lca2graph(lcag_ref, graph_ref)
            plt.sca(ax[it])
            graph_ref.visualize(opt="max", ax=ax[it])
        except:
            continue

        it += 1
    plt.show()
    input()
