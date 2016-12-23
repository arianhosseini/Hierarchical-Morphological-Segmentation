import theano
from theano import tensor
from theano.compat.python2x import OrderedDict
import numpy

import common


thenao.config.floatX = 'float32'


class Node():
    def __init__(self, value=None):
        self.children = []
        self.parent = None
        self.value = value
        self.label = None
        self.id = None
        self.index = None

    def add_left_child(left_node):
        if not self.children:
            self.children = [None, None]
        self.children[0] = left_node
        left_node.parent = self

    def add_rigth_child(right_node):
        if not self.children:
            self.children = [None, None]
        self.children[1] = right_node
        right_node.parent = self

class RecursiveNN():
    def __init__(self, embedding_dim, vocab_size, hidden_dim, output_dim, tree_degree=2, learning_rate=0.01, train_word_embeddings=True):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tree_degree = tree_degree
        self.learning_rate = learning_rate
        self.train_word_embeddings = train_word_embeddings

        self.params = []
        self.tparams = []
        self.embeddings = theano.shared(common.init_norm_weight(shape=(self.vocab_size,self.embedding_dim)))
        if self.train_word_embeddings:
            self.params.append(self.embeddings)

        #TODO: write a general function to create the recursive method
        self.rec_w_he = theano.shared(common.init_norm_weight(shape=(self.hidden_dim, self.embedding_dim)))
        self.rec_w_hh = theano.shared(common.init_norm_weight(shape=(self.hidden_dim, self.hidden_dim)))
        self.rec_b_h = theano.shared(common.init_vector_weight(self.hidden_dim))
        self.params.extend([self.rec_w_he, self.rec_w_hh, self.rec_b_h])

        self.w_output = theano.shared(common.init_norm_weight(shape=(self.hidden_dim, self.output_dim)))
        self.b_output = thenao.shared(common.init_vector_weight(dim=self.output_dim))
        self.params.extend([self.w_output, self.b_output])

        self.input_words = tensor.ivector(name='input_words') #input word indices
        self.tree = tensor.imatrix(name='tree') #tree matrix [g(a, b) -> c]
        self.y = tensor.fvector(name='y') #output -> output_dim

        self.input_embeddings = self.embeddings[self.input_words]
        self.num_of_words = self.input_words.shape[0]
        self.computed_tree = self.compute_recursive_tree()
        self.last_node_hidden = self.computed_tree[-1]
        self.predicted_y = self.compute_final_output()
        self.computed_loss = self.compute_loss()
        

    #TODO: remember to change this for re-ranking
    def compute_loss(self):
        return tensor.sum(tensor.sqrt(self.y - self.predicted_y))

    #can modify how to compute the output
    def compute_final_output(self):
        return tensor.nnet.softmax(tensor.dot(self.w_output, self.last_node_hidden) + self.b_output)

    def get_recursiveNN_inputs(self, root):
        """
        returns input_words and tree matrix.
        traversing the tree layer by layer (without recursion) to get all leaves.

        """
        current_layer = [root]
        all_layers = []
        leaves = []
        while current_layer:
            next_layer = []
            for node in current_layer:
                node.index = None
                leaf = True
                for child in node.children: #compute from rightmost
                    if child != None:
                        leaf = False
                        next_layer.append(child)
                if leaf:
                    leaves.append(node)
            current_layer = next_layer

        leaf_values, leaf_labels = [], []
        index = 0
        for index, node in enumerate(leaves):
            node.index = index
            leaf_values.append(node.value)
            leaf_labels.append(node.label)
        print "%d leaves" %index
        current_layer = [root]
        next_layer = []
        while current_layer:
            all_layers.append(current_layer)
            for node in layer:
                for child in node.children:
                    if child:
                        next_layer.append(child)
            current_layer = next_layer
        tree = []
        internal_nodes_values = []
        internal_nodes_labels = []
        index += 1 #internal nodes' indices starting point
        for layer in all_layers[::-1]: #starting from the bottom
            for node in layer:
                if node.index == None:
                    child_indices = []
                    for child in node.children:
                        if child:
                            child_indices.append(child.index)
                        else:
                            print "going to put -1 in child_indices"
                            child_indices.append(-1)
                    node.index = index
                    index += 1
                    tree.append(child_indices+[node.index])
                    internal_nodes_values.append(node.value)
                    internal_nodes_labels.append(node.label)

        all_labels = leaf_labels + internal_nodes_labels
        all_values = leaf_values + internal_nodes_values
        labels = []
        labels_mask = []
        # for ind, label in enumerate(all_labels):
        #     if label is None:
        #         labels_mask.append(False)
        #         all_labels[ind] = 0
        #     else:
        #         labels_mask.append(True)

        return numpy.array(leaf_values, dtype='int32'), numpy.array(tree, dtype='int32')




    def compute_recursive_tree(self):

        def _recursive_node(parent_node_embedding, children_hidden):
            children_sum = tensor.sum(children_hidden, axis=0)
            cell = tensor.tanh(self.rec_b_h + tensor.dot(self.rec_w_he, parent_node_embedding) + tensor.dot(self.rec_w_hh, children_sum))
            return cell

        leaf_dummy_child = 0 * theano.shared(common.init_norm_weight(self.tree_degree, self.hidden_dim)) #can also use a vector
        num_of_internal_nodes = self.tree.shape[0]
        num_of_leaves = self.num_of_words - num_of_internal_nodes
        leaf_hiddens , _ = theano.map(
            fn=self.recursive_node,
            sequences=[self.input_embeddings[:self.num_of_leaves]],
            non_sequences=[leaf_dummy_child]
            )

        #node_info is the each row of the tree
        def _compute_recurrence_internals(current_embedding, node_info, internal_index, nodes_hidden, _ ):
            child_mask = node_info > -1
            decrese_index = - child_mask * internal_index #cause we are omiting the head of nodes_hidden at each iteration
            child_hiddens = nodes_hidden[node_info+decrese_index] * child_mask.dimshuffle(0,'x') #broadcast
            current_node_hidden = _recursive_node(current_embedding, child_hiddens)
            nodes_hidden = tensor.concatenate([nodes_hidden, current_node_hidden.reshape([1,self.hidden_dim])]) #adding current node to hiddens
            return nodes_hidden[1:], current_node_hidden

        (_ , last_node_hidden ) , _ = theano.scan(
            fn=_compute_recurrence_internals,
            sequences=[self.embeddings[num_of_leaves:], self.tree, tensor.arange(num_of_internal_nodes)],
            outputs_info=[leaf_hiddens, thano.shared(common.init_vector_weight(dim=self.hidden_dim))],
            n_steps=num_of_internal_nodes
            )

        return tensor.concatenate([leaf_hiddens, last_node_hidden], axis=0)
