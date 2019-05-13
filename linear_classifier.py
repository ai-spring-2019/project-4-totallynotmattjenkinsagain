"""
linear_classifier.py
Usage: python3 linear_classifier.py data_file.csv
"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads data from filename. The optional parameters can allow it
    to read data in different formats. Returns a list of headers and a
    list of lists of data."""
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

def convert_data_to_pairs(data):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        pair = (example[:-1], example[-1])
        pairs.append(pair)
    return pairs

def linear_classification_hard_threshold(training):
    """This is what you need to implement (and the logistic version)"""
    
    EPOCHS = 1000

    # 1. set w = random weights
    w = []
    for _ in range(len(training[0][0])):
        w.append((random.random() * 20) - 10)
    print(w)

    for t in range(EPOCHS):
        x, y = random.choice(training)
        for i in range(len(w)):
            w[i] = w[i] + (1000 / (1000 + t)) * (y - hw(w, x)) * x[i]
    return w

def hw(w, x):
    sum_val = 0
    for i in range(len(w)):
        sum_val += w[i] * x[i]
    if sum_val >= 0:
        return 1
    return 0

def main():
    # Read data from the file provided at command line
    header, data = read_data(sys.argv[1], ",")

    # Convert data into (x, y) tuples
    example_pairs = convert_data_to_pairs(data)

    # Insert 1.0 as first element of each x to work with the dummy weight
    training = [([1.0] + x, y) for (x, y) in example_pairs]

    # See what the data looks like
    # for (x, y) in training:
    #     print("x = {}, y = {}".format(x, y))

    # Run linear classification. This is what you need to implement
    w = linear_classification_hard_threshold(training)
    print(w)

    correct = 0
    size = len(training)
    for val in training:
        x, y = val
        if hw(w, x) == y:
            correct += 1
    print("Percentage correct:", (correct / size) * 100)

if __name__ == "__main__":
    main()
