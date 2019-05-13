"""
Matthew R. Jenkins
Professor Helmuth
AI
6 May 2019

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum_of_vectors = 0
    for i in range(len(v1)):
        sum_of_vectors += v1[i] * v2[i]
    return sum_of_vectors

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class Node:
    """Node for neural network."""
    def __init__(self):
        """Constructs a node.

        Arguments:
            None
        """

        # setting output and input nodes
        self.output_nodes = None
        self.input_nodes = None

        # setting defaults for inputs, and output
        self.input = 0
        self.output = 0

    def __str__(self):
        """Prints Node in a nice way.

        Arguments:
            None
        """
        string = "Node with:" + "\n"
        string += ("\t" + "INPUT CONNECTIONS: " + str(self.input_nodes) + "\n")
        string += ("\t" + "OUTPUT CONNECTIONS: " + str(self.output_nodes) + "\n")
        string += ("\t" + "INPUTS: " + str(self.input) + "\n")
        string += ("\t" + "OUTPUT: " + str(self.output) + "\n")
        return(string)

class NeuralNetwork:
    """Neural Network class."""
    def __init__(self, list_of_params):
        """Given a list of parameters input layer, hidden layers, and output
        layer, constructs a neural network.
        
        Arguments:
            list_of_params - list of ints - element 1 will be input layer, 
            elements 2 to n will be hidden layers, and last element (n) will be
            output layer.
        """
        # first element is input layer node count
        # last element is output layer node count
        # everything else is hidden layer node count

        # generate input_layer
        self.input_layer = [Node() for _ in range(list_of_params[0])]

        # generate hidden_layer
        self.hidden_layer = [[Node() for _ in range(list_of_params[i])] for i in range(1, len(list_of_params[1:-1]) + 1)]

        # generate output_layer
        self.output_layer = [Node() for _ in range(list_of_params[-1])]

        # connecting nodes to eachother + setting weights
        # NOTE: Weights are only one direction. Input nodes are used to get access to weights between nodes.

        # input layer
        for node in self.input_layer:
            random_numbers = [random.uniform(-5, 5) for _ in range(len(self.hidden_layer[0]))]
            node.output_nodes = list(zip(self.hidden_layer[0], random_numbers))

        # hidden layer / layers
        for layer in range(len(self.hidden_layer)):
            for node in self.hidden_layer[layer]:
                if len(self.hidden_layer) > 1:
                    # first - input to hidden layer 0
                    if layer == 0:
                        # set input nodes
                        node.input_nodes = self.input_layer
                        # set random weights and make output nodes with weights
                        random_numbers = [random.uniform(-5, 5) for _ in range(len(self.hidden_layer[layer + 1]))]
                        node.output_nodes = list(zip(self.hidden_layer[layer + 1], random_numbers))

                    # last - hidden layer n to output layer
                    elif layer == len(self.hidden_layer) - 1:
                        # set input nodes
                        node.input_nodes = self.hidden_layer[layer - 1]
                        # set random weights and make output nodes with weights
                        random_numbers = [random.uniform(-5, 5) for _ in range(len(self.output_layer))]
                        node.output_nodes = list(zip(self.output_layer, random_numbers))

                    # middle - hidden layer 0 to hidden layer n
                    else:
                        # set input nodes
                        node.input_nodes = self.hidden_layer[layer - 1]
                        # set random weights and make output nodes with weights
                        random_numbers = [random.uniform(-5, 5) for _ in range(len(self.hidden_layer[layer + 1]))]
                        node.output_nodes = list(zip(self.hidden_layer[layer + 1], random_numbers))

                # assuming we have a layer of length 1
                else:
                    node.input_nodes = self.input_layer
                    random_numbers = [random.uniform(-5, 5) for _ in range(len(self.output_layer))]
                    node.output_nodes = list(zip(self.output_layer, random_numbers))
        
        # output
        for node in self.output_layer:
            node.input_nodes = self.hidden_layer[-1]

    def __str__(self):
        """Prints NN in a nice way.

        Arguments:
            None
        """
        string = ""

        # input layer
        string += "Input Layer: \n"
        for node in self.input_layer:
            string += "    " + str(node) + "\n"

        # hidden layers
        for num in range(len(self.hidden_layer)):
            string += "Hidden Layer " + str(num + 1) + ": \n"
            for node in self.hidden_layer[num]:
                string += "    " + str(node) + "\n"

        # output layer
        string += "Output Layer: \n"
        for i, node in enumerate(self.output_layer):
            if i == len(self.output_layer):
                string += "    " + str(node) + "\n"
            else:
                string += "    " + str(node)
        return string

    def back_propagation_learning(self, data_set):
        """Given training data, sets weights and backpropagates weights to readjust based on error.
        
        Arguments:
            training - list of lists, where element 1 is a list of floats, and element 2 is a float
        """
        error_vector = {}

        epochs = 0
        while epochs < 1000:
            for (x, y) in data_set:
        
                # propagate inputs forward and compute outputs
                # input layer
                for i, node in enumerate(self.input_layer):
                    # setting inputs and outputs without weighting
                    node.input = x[i+1]
                    node.output = x[i+1]

                # hidden layer
                for layer in range(len(self.hidden_layer)):
                    for node in self.hidden_layer[layer]:
                        prev_inputs = [i.output for i in node.input_nodes]

                        # getting prev_weights by going through prev outputs and looking for ones with the node we're using in it
                        # once done, we add the weight to a list
                        prev_weights = []
                        for outputs in [i.output_nodes for i in node.input_nodes]:
                            for elem in outputs:
                                if elem[0] == node:
                                    prev_weights.append(elem[1])
                        
                        node.input = dot_product(prev_inputs, prev_weights)
                        node.output = logistic(node.input)

                # output layer
                # redundant but necessary because of how I implemented my data structure
                for node in self.output_layer:
                    prev_inputs = [i.output for i in node.input_nodes]

                    prev_weights = []
                    for outputs in [i.output_nodes for i in node.input_nodes]:
                        for elem in outputs:
                            if elem[0] == node:
                                prev_weights.append(elem[1])
                    node.input = dot_product(prev_inputs, prev_weights)
                    node.output = logistic(node.input)

                # back propagation
                # output
                for i, node in enumerate(self.output_layer):
                    error_vector[node] = node.output * (1 - node.output) * (y[i] - node.output)

                # hidden layers
                for layer in range(len(self.hidden_layer) - 1, - 1, - 1):
                    for node in self.hidden_layer[layer]:
                        prev_errors = [error_vector[i] for (i, _) in node.output_nodes]
                        prev_weights = [i[1] for i in node.output_nodes]
                        sum_of_errors_and_weights = dot_product(prev_errors, prev_weights)
                        error_vector[node] = node.output * (1 - node.output) * sum_of_errors_and_weights

                # input layer
                for node in self.input_layer:
                    prev_errors = [error_vector[i] for (i, _) in node.output_nodes]
                    prev_weights = [i[1] for i in node.output_nodes]
                    sum_of_errors_and_weights = dot_product(prev_errors, prev_weights)
                    error_vector[node] = node.output * (1 - node.output) * sum_of_errors_and_weights

                # updating weights
                # we only care about input and hidden because outputs don't really have stuff to connect to.
                for node in self.input_layer:
                    for i, node_weight_pair in enumerate(node.output_nodes):
                        new_weight = node_weight_pair[1] + (1000 / (1000 + epochs)) * node.output * error_vector[node_weight_pair[0]]
                        node.output_nodes[i] = (node_weight_pair[0], new_weight)

                for layer in range(len(self.hidden_layer)):
                    for node in self.hidden_layer[layer]:
                        for i, node_weight_pair in enumerate(node.output_nodes):
                            new_weight = node_weight_pair[1] + (1000 / (1000 + epochs)) * node.output * error_vector[node_weight_pair[0]]
                            node.output_nodes[i] = (node_weight_pair[0], new_weight)
            
            epochs += 1

    def test_on_example(self, test_data):
        """Simulates a test run on a given example of testing data."""
        # propagate inputs forward and compute outputs
        # input layer
        x, y = test_data
        output = []

        for i, node in enumerate(self.input_layer):
            # setting inputs and outputs without weighting
            node.input = x[i+1]
            node.output = x[i+1]

        # hidden layer
        for layer in range(len(self.hidden_layer)):
            for node in self.hidden_layer[layer]:
                prev_inputs = [i.output for i in node.input_nodes]

                # getting prev_weights by going through prev outputs and looking for ones with the node we're using in it
                # once done, we add the weight to a list
                prev_weights = []
                for outputs in [i.output_nodes for i in node.input_nodes]:
                    for elem in outputs:
                        if elem[0] == node:
                            prev_weights.append(elem[1])
                
                node.input = dot_product(prev_inputs, prev_weights)
                node.output = logistic(node.input)

        # output layer
        # redundant but necessary because of how I implemented my data structure
        for node in self.output_layer:
            prev_inputs = [i.output for i in node.input_nodes]

            prev_weights = []
            for outputs in [i.output_nodes for i in node.input_nodes]:
                for elem in outputs:
                    if elem[0] == node:
                        prev_weights.append(elem[1])
            node.input = dot_product(prev_inputs, prev_weights)
            node.output = float(round(logistic(node.input)))

        output = [node.output for node in self.output_layer]
        if output == y:
            return True
        return False

    def k_fold_cross_validation(self, data_set, k):
        """Performs k-fold cross validation on the given NN using the given data set.
        
        Typical values for K include 5, 10, n (where n is length of data_set)
        """

        # split data into k equal sized subsets (these are randomized to make things interesting)
        subsets = []
        for _ in range(k):
            subsets.append(random.sample(data_set, round(len(data_set) / k)))
        
        # run k iterations of learning, where each iteration uses a different subset of the test data.
        for i in range(k):
            self.back_propagation_learning(subsets[i])

        # then, average accuracy over k rounds for a better estimate of accuracy on unseen data
        average = 0
        for i in range(1, k + 1):
            average += self.testing(subsets[i - 1])
        average /= k
        return average

    def testing(self, test_data):
        """For each example of test data, runs a test of the NN on it, and tallies the number of correct runs."""
        tally = 0
        size = len(test_data)
        for example in test_data:
            if self.test_on_example(example):
                tally += 1
        return (tally / size) * 100

def testing_nn_layout(data_set, data_set_name, input_layer_size, output_layer_size):
    """Generates different NNs and generates accuracy based on the layout."""

    # controls: epochs = 1000

    with open("nn_test_" + data_set_name[:-4] + ".out", "w") as file:

        min_val = 0
        max_val = 0
        # 3 layer NNs
        accuracy_list = []
        print("3 Layer NNs:")
        for i in range(min_val, max_val + 1):
            nn_list = [input_layer_size, i, output_layer_size]
            nn = NeuralNetwork(nn_list)
            nn_accuracy = nn.k_fold_cross_validation(data_set, 5)
            print((nn_list, nn_accuracy))
            file.write(str((nn_list, nn_accuracy)) + "\n")
            accuracy_list.append((nn_list, nn_accuracy))
        print()

        # 4 layer NNs
        print("4 Layer NNs:")
        for i in range(min_val, max_val + 1):
            for j in range(min_val, max_val + 1):
                nn_list = [input_layer_size, i, j, output_layer_size]
                nn = NeuralNetwork(nn_list)
                nn_accuracy = nn.k_fold_cross_validation(data_set, 5)
                print((nn_list, nn_accuracy))
                file.write(str((nn_list, nn_accuracy)) + "\n")
                accuracy_list.append((nn_list, nn_accuracy))
        print()

def main():
    """Main method - runs everything necessary to allow NN to function."""
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    data = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    for example in data:
        print(example)

    # testing_nn_layout(data, sys.argv[1], len(data[0][0]) - 1, len(data[0][1]))

    # print()

    nn = NeuralNetwork([2, 1, 1])
    print(nn)
    # nn.k_fold_cross_validation(data, 5)

if __name__ == "__main__":
    main()