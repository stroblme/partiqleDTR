from ordered_set import T
from .nri import bb_NRIModel, rel_pad_collate_fn
from torch.nn.parallel import DataParallel
# from generate.load import load_nri
import numpy as np

import torch as t
# import config as cfg
from torch.nn.functional import cross_entropy
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import numpy as np

import mlflow

import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple, List

import logging
log = logging.getLogger(__name__)


def train_qgnn(torch_dataset_lca_and_leaves, n_hid:int, n_momenta:int, dropout_rate:float,
                learning_rate:float, learning_rate_decay:int, gamma:float, batch_size:int, epochs:int):
    n_fsps = int(max([len(subset[0]) for _, subset in torch_dataset_lca_and_leaves.items()]))+1

    model = bb_NRIModel(n_momenta, n_fsps)
    model = DataParallel(model)
    ins = Instructor(model, torch_dataset_lca_and_leaves, None, learning_rate, learning_rate_decay, gamma, batch_size, epochs)
    # ins.testLca2Graph()
    
    return ins.train()


class DataWrapper(Dataset):
    """
    A wrapper for t.utils.data.Dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Instructor():
    """
    Train the encoder in an supervised manner given the ground truth relations.
    """
    def __init__(self, model: t.nn.DataParallel, data: dict,  es: np.ndarray,
                learning_rate: float, learning_rate_decay: int, gamma: float, batch_size:int, epochs:int):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        self.model = model
        
        # self.data = {key: TensorDataset(*value)
        #              for key, value in data.items()}
        self.data = data
        # self.data = data
        # self.es = t.LongTensor(es)
        # number of nodes
        self.epochs = epochs
        self.batch_size = batch_size
        # optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        # learning rate scheduler, same as in NRI
        self.scheduler = StepLR(self.opt, step_size=learning_rate_decay, gamma=gamma)


    def train(self):
        log.info(f'Training started with a batch size of {self.batch_size}')
        result = None            
        best_acc = 0
        for epoch in range(1, 1 + self.epochs):
            for mode in ["train", "val"]:
                data_batch = DataLoader(
                                        DataWrapper(self.data[mode]),
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        collate_fn=rel_pad_collate_fn) # to handle varying input size

                epoch_loss = 0.
                epoch_acc = 0.

                log.info(f"Running epoch {epoch} in mode {mode}")
                for states, labels in data_batch:
                    scale = 1 / labels.size(1) # get the scaling dependend on the number of classes

                    if mode == "train":
                        self.model.train() # set the module in training mode

                        prob = self.model.module(states)
                        loss = cross_entropy(prob, labels, ignore_index=-1)
                        acc = self.edge_accuracy(prob, labels)

                        # do the actual optimization
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        if labels.numpy().min() < -1:
                            raise Exception(f"Found graph with negative values: {labels.numpy()}")
                    elif mode == "val":
                        self.model.module.eval() # trigger evaluation forward mode
                        with t.no_grad(): # disable autograd in tensors

                            prob = self.model.module(states)

                            loss = cross_entropy(prob, labels, ignore_index=-1)
                            acc = self.edge_accuracy(prob, labels)
                    elif mode == "test":
                        self.model.module.eval() # trigger evaluation forward mode
                        with t.no_grad(): # disable autograd in tensors

                            prob = self.model.module(states)

                            loss = cross_entropy(prob, labels, ignore_index=-1)
                            acc = self.edge_accuracy(prob, labels)
                    else:
                        log.error("Unknown mode")

                    epoch_loss += scale * loss
                    epoch_acc += scale * acc

                    if acc > best_acc and mode == "val":
                        # update the current best model when approaching a higher accuray
                        best_acc = acc
                        result = self.model
                        try:
                            c_plt = self.plotBatchGraphs(prob, labels)
                        except Exception as e:
                            log.error(f"Exception occured when trying to plot graphs: {e}\n\tThe lcag matrices were:\n\t{labels.numpy()}\n\tand\n\t{prob.numpy()}")

                        mlflow.log_figure(c_plt.gcf(), f"e{epoch}_sample_graph.png")

                epoch_loss /= len(data_batch) # to the already scaled loss, apply the batch size scaling
                epoch_acc /= len(data_batch) # to the already scaled accuracy, apply the batch size scaling

                mlflow.log_metric(key=f"{mode}_accuracy", value=epoch_acc.item(), step=epoch)
                mlflow.log_metric(key=f"{mode}_loss", value=epoch_loss.item(), step=epoch)

                # learning rate scheduling
                self.scheduler.step()

    

        return {
            "model_qgnn":result
        }


    def edge_accuracy(self, logits:t.Tensor, labels:t.Tensor)->float:
        # logits: [Batch, Classes, LCA_0, LCA_1]
        probs = logits.softmax(1) # get softmax for probabilities
        preds = probs.max(1)[1] # find maximum across the classes
        correct = (labels==preds).sum().float()
        return correct/(labels.size(1)*labels.size(2))           

    def lca2graph(self, lca, graph):

        nodes = [i for i in range(lca.size(0))] # first start with all nodes available directly from the lca
        processed = [] # keeps track of all nodes which are somehow used to create an edge
        
        while lca.max() > 0:
            directPairs = list((lca==1.0).nonzero())
            directPairs.sort(key=lambda dp: sum(dp))
            directPairs = directPairs[1::2]

            def convToPair(pair:t.Tensor) -> Tuple[int,int]:
                return (int(pair[0]), int(pair[1]))

            def getOverlap(edge:Tuple[int,int], ref:List[int]) -> List[Tuple[int,int]]:
                return list(set(edge) & set(ref))

            def addNodeNotInSet(node:int, parent:int, ovSet:List[Tuple[int,int]], appendSet:bool) -> List[Tuple[int,int]]:
                if node not in ovSet:
                        graph.addEdge(node, parent)
                        if appendSet:
                            ovSet.append(node)
                return ovSet

            def addPairNotInSet(pair:Tuple[int,int], parent:int, ovSet:List[Tuple[int,int]], appendSet:bool=True) -> List[Tuple[int,int]]:
                if pair[0] == pair[1]:
                    return ovSet

                for node in pair:
                    ovSet = addNodeNotInSet(node, parent, ovSet, appendSet)
                return ovSet

            for tensor_pair in directPairs:
                pair = convToPair(tensor_pair)

                while(True):
                    # this works since they are sorted by pair[0]
                    overlap = getOverlap(pair, processed)

                    # no overlap -> create new ancestor
                    if len(overlap) == 0:
                        # create a new parent and set ancestor accordingly
                        nodes.append(max(nodes)+1)
                        ancestor = nodes[-1]

                        # found two new edged, cancel this subroutine and  process next pair
                        processed = addPairNotInSet(pair, ancestor, processed)
                        break

                    # only 1 common node -> set the ancestor to the parent of this overlap
                    elif len(overlap) == 1:
                        # overlap has only one element here
                        ancestor = graph.parentOf(overlap[0])

                        generation = -1
                        while True:
                            if graph.parentOf(ancestor) == ancestor:
                                break
                            ancestor = graph.parentOf(ancestor)
                            generation -= 1

                        if lca[0][0] <= generation:
                            nodes.append(max(nodes)+1)

                            addNodeNotInSet(ancestor, nodes[-1], processed, True)

                            ancestor = nodes[-1]

                        processed = addPairNotInSet(pair, ancestor, processed)
                        # found a new edge, cancel this subroutine and process next pair
                        break

                    # full overlap -> meaning they were both processed previously
                    else:
                        ancestor_a = graph.parentOf(overlap[0])
                        while True:
                            if graph.parentOf(ancestor_a) == ancestor_a:
                                break
                            ancestor_a = graph.parentOf(ancestor_a)

                        ancestor_b = graph.parentOf(overlap[1])
                        while True:
                            if graph.parentOf(ancestor_b) == ancestor_b:
                                break
                            ancestor_b = graph.parentOf(ancestor_b)

                        # cancel if they have the same parent
                        if ancestor_a == ancestor_b:
                            break
                        # cancel if they are already connected (cause this would happen again in the next round)
                        elif graph.parentOf(ancestor_a) == ancestor_b:
                            break # TODO check here if this break is ok

                        # overwrite edge by new set of parents
                        pair = (ancestor_a, ancestor_b)

                        # don't cancel here, process this set instead again (calc overlap...)

            lca -= 1
            if len(directPairs) == 0:
                lca += t.diag(t.ones(lca.size(0), dtype=t.long))
            # processed = []

    def plotBatchGraphs(self, batch_logits, batch_ref):
        fig, ax = plt.subplots(4, 2, figsize=(15,15), gridspec_kw={'width_ratios': [1, 1]})
        fig.tight_layout()
        it = 0
        for logits, lcag_ref in zip(batch_logits, batch_ref):
            lcag = logits.max(0)[1]
            graph = GraphVisualization()

            self.lca2graph(lcag, graph)
            plt.sca(ax[it][0])
            graph.visualize(opt="max", ax=ax[it][0])

            graph_ref = GraphVisualization()

            self.lca2graph(lcag_ref, graph_ref)
            plt.sca(ax[it][1])
            graph_ref.visualize(opt="max", ax=ax[it][1])

            if it*2>4:
                break

            it += 1

        return plt
            

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
        example_1 = t.tensor([
            #d  g  f  h  e
            [0, 2, 1, 2, 1], # d
            [2, 0, 2, 1, 2], # g
            [1, 2, 0, 2, 1], # f
            [2, 1, 2, 0, 2], # h
            [1, 2, 1, 2, 0], # e
        ])
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
        example_2 = t.tensor([
            #d  g  f  h  e
            [0, 5, 2, 5, 2], # d
            [5, 0, 5, 2, 5], # g
            [2, 5, 0, 5, 2], # f
            [5, 2, 5, 0, 5], # h
            [2, 5, 2, 5, 0], # e
        ])
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
        example_3 = t.tensor([
            #e  f  g  h  i
            [0, 1, 3, 3, 3],  # e
            [1, 0, 3, 3, 3],  # f
            [3, 3, 0, 1, 2],  # g
            [3, 3, 1, 0, 2],  # h
            [3, 3, 2, 2, 0]   # i
        ])
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
        example_4 = t.tensor([
            #g  h  d  f  c
            [0, 1, 2, 2, 3],  # g
            [1, 0, 2, 2, 3],  # h
            [2, 2, 0, 2, 3],  # d
            [2, 2, 2, 0, 3],  # f
            [3, 3, 3, 3, 0]   # c
        ])

        example_5 = t.tensor([
            [0, 2, 1, 2],
            [2, 0, 2, 1],
            [1, 2, 0, 2],
            [2, 1, 2, 0]
        ])
        examples = [example_1, example_2, example_3, example_4, example_5]

        fig, ax = plt.subplots(1, 5, figsize=(15,5))
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



    def save(self, prob, lca_ref, postfix):
        graph, graph_ref = self.generateGraphsFromProbAndRef(prob, lca_ref)

        graph.save(f"pred_{postfix}")
        graph_ref.save(f"ref_{postfix}")

        # Defining a Class
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
            if opt=="min":
                pos = self.hierarchy_pos(G, min(min(self.visual)))
            elif opt=="max":
                pos = self.hierarchy_pos(G, max(max(self.visual)))
        except TypeError:
            log.warning("Provided LCA is not a tree. Will use default graph style for visualization")
            # raise RuntimeError
        nx.draw_networkx(G, pos, ax=ax)
        # plt.show()


    def save(self, filename):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.save(filename)
  
    def hierarchy_pos(self, G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

        '''
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
        '''
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
            '''
            see hierarchy_pos docstring for most arguments from https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            '''
        
            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)  
            if len(children)!=0:
                dx = width/len(children) 
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos

                
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)