from ordered_set import T
import torch
from .models.encoder import GNNENC
from .models.nri import bb_NRIModel, rel_pad_collate_fn
from torch.nn.parallel import DataParallel
# from generate.load import load_nri
from itertools import permutations
import numpy as np
from torch import LongTensor, FloatTensor

from torch.utils.data.dataset import TensorDataset

import torch
from torch import Tensor
# import config as cfg
from torch.nn.functional import cross_entropy
from .utils.torch_extension import edge_accuracy, asym_rate
from torch.utils.data.dataset import TensorDataset
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
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
    # load data
    # SIZE = model_parameters["SIZE"] if "SIZE" in model_parameters else None
    # n_hid = model_parameters["N_HID"] if "N_HID" in model_parameters else None
    # n_momenta = model_parameters["DIM"] if "DIM" in model_parameters else None

    # log.info(f"Model parameters:\n{model_parameters}")

    # data, es, _ = load_nri(all_leaves_shuffled, num_of_leaves)
    # generate edge list of a fully connected graph

    # max_depth = int(max([np.array(subset[0]).max() for _, subset in torch_dataset_lca_and_leaves.items()]))+1 # get the num of childs from the label list
    n_fsps = int(max([len(subset[0]) for _, subset in torch_dataset_lca_and_leaves.items()]))+1
    # n_fsps = 4
    # es = LongTensor(np.array(list(permutations(range(SIZE), 2))).T)
    # es = list(permutations(range(n_fsps), 2))

    # get ut
    # es.sort(key=lambda es: sum(es))
    # es = es[1::2]

    # es = LongTensor(np.array(es).T)

    # encoder = GNNENC(n_momenta, n_hid, max_depth, dropout_rate=dropout_rate)
    # model = NRIModel(encoder, es, n_fsps)
    model = bb_NRIModel(n_momenta, n_fsps)
    model = DataParallel(model)
    ins = Instructor(model, torch_dataset_lca_and_leaves, None, learning_rate, learning_rate_decay, gamma, batch_size, epochs)
    
    return ins.train()



class DataWrapper(Dataset):
    """
    A wrapper for torch.utils.data.Dataset.
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
    def __init__(self, model: torch.nn.DataParallel, data: dict,  es: np.ndarray,
                learning_rate: float, learning_rate_decay: int, gamma: float, batch_size:int, epochs:int):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        # learning_rate = model_parameters["LR"] if "LR" in model_parameters else None
        # learning_rate_decay = model_parameters["LR_DECAY"] if "LR_DECAY" in model_parameters else None
        # gamma = model_parameters["GAMMA"] if "GAMMA" in model_parameters else None
        # size = model_parameters["SIZE"] if "SIZE" in model_parameters else None
        # batch_size = model_parameters["BATCH_SIZE"] if "BATCH_SIZE" in model_parameters else None
        # epochs = model_parameters["EPOCHS"] if "EPOCHS" in model_parameters else None


        # super(XNRIENCIns, self).__init__(cmd)
        self.model = model
        
        # self.data = {key: TensorDataset(*value)
        #              for key, value in data.items()}
        self.data = data
        # self.data = data
        # self.es = torch.LongTensor(es)
        # number of nodes
        self.epochs = epochs
        self.batch_size = batch_size
        # optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        # learning rate scheduler, same as in NRI
        self.scheduler = StepLR(self.opt, step_size=learning_rate_decay, gamma=gamma)

    @staticmethod
    def optimize(opt: Optimizer, loss: torch.Tensor):
        """
        Optimize the parameters based on the loss and the optimizer.

        Args:
            opt: optimizer
            loss: loss, a scalar
        """
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def load_data(inputs, batch_size: int, shuffle: bool=True):
        """
        Return a dataloader given the input and the batch size.
        """
        data = DataWrapper(inputs)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=rel_pad_collate_fn) # to handle varying input size
        return batches

    def train(self):
        # use the accuracy as the metric for model selection, default: 0
        val_best = 0
        # path to save the current best model
        # prefix = '/'.join(cfg.log.split('/')[:-1])
        # name = '{}/best.pth'.format(prefix)train_steps
        try:
            mlflow.end_run()
            log.warn("There was an existing run which is now cancelled.")
        except:
            pass

        with mlflow.start_run():
            log.info(f'Training started with a batch size of {self.batch_size}')
            result = None            
            best_acc = 0
            for epoch in range(1, 1 + self.epochs):
                for mode in ["train", "val"]:
                    data_batch = self.load_data(self.data[mode], self.batch_size) # get the date for the current mode
                    epoch_loss = 0.
                    epoch_acc = 0.

                    log.info(f"Running epoch {epoch} in mode {mode}")
                    for states, labels in data_batch:
                        scale = 1 / labels.size(1) # get the scaling dependend on the number of classes

                        if mode == "train":
                            self.model.train() # set the module in training mode

                            prob = self.model.module(states)
                            loss = cross_entropy(prob, labels, ignore_index=-1)
                            acc = edge_accuracy(prob, labels)

                            self.optimize(self.opt, loss)
                        elif mode == "val":
                            self.model.evaluate() # trigger evaluation forward mode
                            with torch.no_grad(): # disable autograd in tensors

                                prob = self.model.module(states)

                                loss = cross_entropy(prob, labels, ignore_index=-1)
                                acc = edge_accuracy(prob, labels)
                        elif mode == "test":
                            self.model.evaluate() # trigger evaluation forward mode
                            with torch.no_grad(): # disable autograd in tensors

                                prob = self.model.module(states)

                                loss = cross_entropy(prob, labels, ignore_index=-1)
                                acc = edge_accuracy(prob, labels)
                        else:
                            log.error("Unknown mode")

                        epoch_loss += scale * loss
                        epoch_acc += scale * acc

                        if acc > best_acc:
                            # update the current best model when approaching a higher accuray
                            best_acc = acc
                            result = self.model

                    epoch_loss /= len(data_batch) # to the already scaled loss, apply the batch size scaling
                    epoch_acc /= len(data_batch) # to the already scaled accuracy, apply the batch size scaling

                    mlflow.log_metric(key=f"{mode}_accuracy", value=epoch_acc.item(), step=epoch)
                    mlflow.log_metric(key=f"{mode}_loss", value=epoch_loss.item(), step=epoch)

                    # learning rate scheduling
                    self.scheduler.step()

        

        return {
            "model_qgnn":result
        }

    # def report(self, name: str) -> float:
    #     """
    #     Evaluate the accuracy.

    #     Args:
    #         name: 'train' / 'val' / 'test'
        
    #     Return:
    #         acc: accuracy of relation reconstruction
    #     """
    #     loss, acc, rate, sparse = self.evaluate(self.data[name])
    #     # log.info('{} acc {:.4f} _acc {:.4f} rate {:.4f} sparse {:.4f}'.format(name, acc, 1 - acc, rate, sparse))
    #     log.info(f"acc: {acc}, loss: {loss}")
    #     return acc

    # def train_nri(self, states: Tensor, lca: Tensor) -> Tensor:
    #     """
    #     Args:
    #         states: [batch, step, node, dim], observed node states
    #         lca: [batch, E, K], ground truth interacting relations

    #     Return:
    #         loss: cross-entropy of edge classification
    #     """
    #     # prob = self.model.module.predict_relations(states)
    #     prob = self.model.module(states)
    #     # self.view(prob, lca)
    #     loss = cross_entropy(prob, lca, ignore_index=-1)
    #     # loss = torch.nn.CrossEntropyLoss(prob, lca, ignore_index=-1)
    #     self.optimize(self.opt, loss)
    #     return loss

    #     lca_filtered = []
    #     for batch in range(lca.shape[0]):
    #         lca_ut_batch = []
    #         for row in range(lca.shape[1]):
    #             for col in range(lca.shape[2]):
    #                 if self.sideSelect(row, col):
    #                     lca_ut_batch.append(lca[batch][row][col])
    #                     # lca_ut.append(lca[batch][row][col])
    #         lca_filtered.append(lca_ut_batch)
    #     lca_filtered = LongTensor(lca_filtered)


    #     # loss = cross_entropy(prob.view(-1, prob.shape[-1]), lca.transpose(0, 1).flatten())
    #     loss = cross_entropy(prob.view(-1, prob.shape[-1]), lca_filtered.view(-1))
    #     # self.view(prob, lca)
    #     # loss = loss / lca.shape[1]
    #     self.optimize(self.opt, loss)
    #     return loss

    def sideSelect(self, row, col):
        # return row < col # lower (results in node*(node-1)/2)
        # return row > col # upper (results in node*(node-1)/2)
        return row != col # all but diagonal (results in node*(node-1))

    def evaluate(self, test):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted

        Return:
            loss: loss_nll + loss_kl (+ loss_reg) 
            acc: accuracy of relation reconstruction
            rate: rate of assymmetry
            sparse: rate of sparsity in terms of the first type of edge
        """
        acc, rate, sparse, losses = [], [], [], []
        data_batch = self.load_data(test, self.batch_size)
        with torch.no_grad():
            for lca, states in data_batch:
                # prob = self.model.module.predict_relations(states)
                prob = self.model.module(states)
                # self.view(prob, lca)
                loss = cross_entropy(prob, lca)
                # self.view(prob, lca)

                scale = 1 / lca.size(1) #running only a single batch here

                # lca_ut = []
                # for batch in range(lca.shape[0]):
                #     lca_ut_batch = []
                #     for row in range(lca.shape[1]):
                #         for col in range(lca.shape[2]):
                #             if self.sideSelect(row, col):
                #                 lca_ut_batch.append(lca[batch][row][col])
                #                 # lca_ut.append(lca[batch][row][col])
                #     lca_ut.append(lca_ut_batch)
                # lca_ut = LongTensor(lca_ut)

                # # use loss as the validation metric
                # loss = cross_entropy(prob.view(-1, prob.shape[-1]), lca_ut.view(-1))
                # # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                # acc.append(scale * edge_accuracy(prob, lca_ut))
                # acc.append(scale * edge_accuracy(prob, lca))
                acc.append(0)
                # _, p = prob.max(-1)
                # rate.append(scale * asym_rate(p.t(), self.size))
                # sparse.append(prob.max(-1)[1].float().mean() * scale)
        # loss = sum(losses) / self.batch_size
        loss = sum(losses) / len(data_batch)
        # acc = sum(acc) / self.batch_size
        acc = sum(acc) / len(data_batch)
        # rate = sum(rate) / N
        # sparse = sum(sparse) / N
        return loss, acc, rate, sparse

    def prob2lca(self, prob, size):
        batchSize = prob.size(1)
        lca = torch.zeros((batchSize, size, size))

        for batch in range(batchSize):
            prob_idx = 0
            for row in range(size):
                for col in range(size):
                    if self.sideSelect(row, col):
                        _, particle_idx = torch.max(prob[prob_idx][batch], 0) # get the max prob of this prediction
                        # careful, that batch is on dim 1 in the particles whereas it is in dim 0 in the lca
                        lca[batch][row][col] = particle_idx
                        prob_idx += 1 # use seperate indexing to ensure that we don't fall in this matrix scheme
                    elif row == col:
                        lca[batch][row][col] = 0 # so that after transpose and add it will sum up to -1

        lca_sym = lca + lca.transpose(1, 2)

        return lca_sym                

    def lca2graph(self, lca, graph):

        nodes = [i for i in range(lca.size(0))] # first start with all nodes available directly from the lca
        processed = [] # keeps track of all nodes which are somehow used to create an edge

        
        while lca.max() > 0:
            directPairs = list((lca==1.0).nonzero())
            directPairs.sort(key=lambda dp: sum(dp))
            directPairs = directPairs[1::2]

            def convToPair(pair:torch.Tensor) -> Tuple[int,int]:
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
                lca += torch.diag(torch.ones(lca.size(0), dtype=torch.long))
            # processed = []


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
        example_1 = torch.tensor([
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
        example_2 = torch.tensor([
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
        example_3 = torch.tensor([
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
        example_4 = torch.tensor([
            #g  h  d  f  c
            [0, 1, 2, 2, 3],  # g
            [1, 0, 2, 2, 3],  # h
            [2, 2, 0, 2, 3],  # d
            [2, 2, 2, 0, 3],  # f
            [3, 3, 3, 3, 0]   # c
        ])

        graph = GraphVisualization()
        self.lca2graph(example_1, graph)
        plt.figure(1)
        graph.visualize("max")
        plt.show()

        graph = GraphVisualization()
        self.lca2graph(example_2, graph)
        plt.figure(2)
        graph.visualize("max")
        plt.show()
        
        graph = GraphVisualization()
        self.lca2graph(example_3, graph)
        plt.figure(3)
        graph.visualize("max")
        plt.show()

        # pruned_lca_ref = self.prune_lca(example_4)

        # adj_ref = lca2adjacency(pruned_lca_ref)

        # graph_ref =self.generateGraphFromAdj(adj_ref)
        # graph_ref.visualize()
        
        graph = GraphVisualization()
        self.lca2graph(example_4, graph)
        plt.figure(4)
        graph.visualize("max")
        plt.show()
        pass



    def generateGraphFromAdj(self, adj):
        graph = GraphVisualization()
        for row in range(len(adj)):
            for col in range(len(adj)):
                if adj[row][col] and self.sideSelect(row, col):
                    graph.addEdge(row, col)

        return graph

    def generateGraphFromLca(self, lca):
        graph = GraphVisualization()
        self.lca2graph(lca[0], graph)

        return graph

    def prune_lca(self,lca):
        index = sum(lca[:])>0
        pruned_lca = lca[index][:,index]

        return pruned_lca                

    def generateGraphsFromProbAndRef(self, prob, lca_ref):
        lca = self.prob2lca(prob, lca_ref.size(1))
        # pruned_lca = self.prune_lca(lca[0])
        # pruned_lca_ref = self.prune_lca(lca_ref[0])

        graph = self.generateGraphFromLca(lca)
        graph_ref = self.generateGraphFromLca(lca_ref)

        return graph, graph_ref

    def view(self, prob, lca_ref):
        graph, graph_ref = self.generateGraphsFromProbAndRef(prob, lca_ref)
        plt.figure(1)
        plt.title("Reference")
        graph_ref.visualize()
        plt.figure(2)
        plt.title("Prediction")
        try:
            graph.visualize()
        except:
            print("Whoops")
            lca = self.prob2lca(prob, lca_ref.size(1))
            graph = self.generateGraphFromLca(lca)
        # plt.show()

        input()
        del graph, graph_ref


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
    def visualize(self, opt="max"):
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
        nx.draw_networkx(G, pos)
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

        # if root is None:
        #     if isinstance(G, nx.DiGraph):
        #         root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        #     else:
        #         root = random.choice(list(G.nodes))

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