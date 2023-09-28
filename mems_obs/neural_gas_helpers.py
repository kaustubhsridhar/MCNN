import numpy as np
from neupy.algorithms import GrowingNeuralGas, NeuralGasGraph, NeuronNode
from neupy.algorithms.competitive.growing_neural_gas import make_edge_id
from neupy.exceptions import StopTraining
from operator import attrgetter
from scipy.spatial import Voronoi
import time 
from collections import deque, defaultdict
from tqdm import tqdm 
from copy import deepcopy 

class MyIncrementalGrowingNeuralGas(GrowingNeuralGas):

    def __init__(self, *args, **kwargs):
        prev_graph = kwargs.pop('prev_graph')
        super(GrowingNeuralGas, self).__init__(*args, **kwargs) # directly goes to the init of the parent of GrowingNeuralGas; i.e. skips the init of GrowingNeuralGas where self.graph is set to empty NeuralGasGraph()
        self.n_updates = 0
        if prev_graph is None:
            self.graph = NeuralGasGraph()
        else:
            self.graph = prev_graph

    def train_epoch(self, input_train, target_train=None):
        """
        Copied from GrowingNeuralGas.train_epoch. Edits are marked with '# New' comment.
        """
        graph = self.graph
        step = self.step
        neighbour_step = self.neighbour_step

        max_nodes = self.max_nodes
        max_edge_age = self.max_edge_age

        error_decay_rate = self.error_decay_rate
        after_split_error_decay_rate = self.after_split_error_decay_rate
        n_iter_before_neuron_added = self.n_iter_before_neuron_added

        # We square this value, because we deal with
        # squared distances during the training.
        min_distance_for_update = np.square(self.min_distance_for_update)

        n_samples = len(input_train)
        total_error = 0
        did_update = False

        for sample in input_train:
            nodes = graph.nodes
            weights = np.concatenate([node.weight for node in nodes])

            distance = np.linalg.norm(weights - sample, axis=1)
            neuron_ids = np.argsort(distance)

            closest_neuron_id, second_closest_id = neuron_ids[:2]
            closest_neuron = nodes[closest_neuron_id]
            second_closest = nodes[second_closest_id]
            total_error += distance[closest_neuron_id]

            if distance[closest_neuron_id] < min_distance_for_update:
                continue

            self.n_updates += 1
            did_update = True

            closest_neuron.error += distance[closest_neuron_id]
            closest_neuron.weight += step * (sample - closest_neuron.weight)
            closest_neuron.sample_input = deepcopy(sample) # New

            graph.add_edge(closest_neuron, second_closest)

            for to_neuron in list(graph.edges_per_node[closest_neuron]):
                edge_id = make_edge_id(to_neuron, closest_neuron)
                age = graph.edges[edge_id]

                if age >= max_edge_age:
                    graph.remove_edge(to_neuron, closest_neuron)

                    if not graph.edges_per_node[to_neuron]:
                        graph.remove_node(to_neuron)

                else:
                    graph.edges[edge_id] += 1
                    to_neuron.weight += neighbour_step * (
                        sample - to_neuron.weight)

            time_to_add_new_neuron = (
                self.n_updates % n_iter_before_neuron_added == 0 and
                graph.n_nodes < max_nodes)

            if time_to_add_new_neuron:
                nodes = graph.nodes
                largest_error_neuron = max(nodes, key=attrgetter('error'))
                neighbour_neuron = max(
                    graph.edges_per_node[largest_error_neuron],
                    key=attrgetter('error'))

                largest_error_neuron.error *= after_split_error_decay_rate
                neighbour_neuron.error *= after_split_error_decay_rate

                new_weight = 0.5 * (
                    largest_error_neuron.weight + neighbour_neuron.weight
                )
                new_neuron = NeuronNode(weight=new_weight.reshape(1, -1))

                graph.remove_edge(neighbour_neuron, largest_error_neuron)
                graph.add_node(new_neuron)
                graph.add_edge(largest_error_neuron, new_neuron)
                graph.add_edge(neighbour_neuron, new_neuron)

            for node in graph.nodes:
                node.error *= error_decay_rate

        if not did_update and min_distance_for_update != 0 and n_samples > 1:
            raise StopTraining(
                "Distance between every data sample and neurons, closest "
                "to them, is less then {}".format(min_distance_for_update))

        print(f'Number of nodes = {graph.n_nodes}')

        return total_error / n_samples


def data2gas(states, max_memories, gng_epochs, step=0.2, n_start_nodes=2, max_edge_age=50, prev_graph=None):
    # GrowingNeuralGas
    gng = MyIncrementalGrowingNeuralGas(
            n_inputs=states.shape[1],
            n_start_nodes=max_memories,
            shuffle_data=True,
            verbose=True,
            max_nodes=max_memories,
            prev_graph=prev_graph,
        )
    gng.train(states, epochs=gng_epochs)    

    return gng 

