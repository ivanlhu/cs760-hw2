#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

def parse_hw2(filename):
    '''
    Parses a data file as an array with entries of the form
    {'x': [x0,x1], 'y': y}
    '''
    data = []
    with open(filename) as file:
        for line in file.readlines():
            entry = [x for x in line.split()]
            data.append({'x': [float(entry[0]), float(entry[1])], \
                         'y': int(entry[2])})
    return data

def candidate_splits(data):
    '''
    Returns all candidate splits of the form x_i >= j
    as entries of the form {'j': i, 'c': j}
    Coordinates are indexed by 0 (x0, x1)
    '''
    splits = []
    x0s = set([entry['x'][0] for entry in data])
    x1s = set([entry['x'][1] for entry in data])
    for x0 in x0s:
        splits.append({'j': 0, 'c': x0})
    for x1 in x1s:
        splits.append({'j': 1, 'c': x1})
    return splits

def entropy(outcomes):
    '''
    Entropy formula
    '''
    n = sum(outcomes)
    if n == 0:
        return 0
    return -sum([(x/n) * math.log2(x/n) for x in outcomes if x > 0])

def data_entropy(data):
    '''
    Calculates the entropy for labeled dataset
    '''
    counts = [0,0]
    for entry in data:
        counts[entry['y']] += 1
    return entropy(counts)

def split_entropy(data, split):
    '''
    Calculates the entropy of split on data
    '''
    coord = split['j']
    threshold = split['c']
    counts = [0,0]
    for entry in data:
        if entry['x'][coord] >= threshold:
            counts[1] += 1
        else:
            counts[0] += 1
    return entropy(counts)

def cond_entropy(data, split):
    '''
    Calculates the conditional entropy for data given
    split [j,c]
    '''
    coord = split['j']
    threshold = split['c']
    split_yes = [0,0]
    split_no = [0,0]
    for entry in data:
        if entry['x'][coord] >= threshold:
            split_yes[entry['y']] += 1
        else:
            split_no[entry['y']] += 1


    return (sum(split_yes) / len(data) * entropy(split_yes)) + \
        (sum(split_no) / len(data) * entropy(split_no))

def best_split(data):
    if len(data) == 0:
        return None
    splits = candidate_splits(data)
    e = data_entropy(data)
    best = None
    best_ratio = 1.0e-7
    for split in splits:
        se = split_entropy(data, split)
        ce = cond_entropy(data, split)
        if se > 1.0e-7:
            ratio = (e - ce) / se
            if ratio > best_ratio:
                best = split
                best_ratio = ratio
    return best

class DecisionTree:
    def __init__(self, data):
        self.data = data
        self.split = None
        self.left = None
        self.right = None

        counts = [0,0]
        for entry in self.data:
            if entry['y'] == 0:
                counts[0] += 1
            elif entry['y'] == 1:
                counts[1] += 1
        if counts[0] > counts[1]:
            self.majority = 0
        else:
            self.majority = 1

    def show(self, level=0):
        '''
        Prints out the decision tree in DFS order
        '''
        if self.split is None:
            print(('  |' * level) + ' y={0}'.format(self.majority))
        elif self.left and self.right:
            print(('  |' * level) + \
                  ' x_{0} >= {1:.3f}:'.format(self.split['j']+1, self.split['c']))
            print(('  |' * level) + '  L')
            self.left.show(level+1)

            print(('  |' * level) + '  R')
            self.right.show(level+1)

    def gen(data):
        '''
        Generates decision tree from data
        '''
        split = best_split(data)
        if split is None:
            tree = DecisionTree(data)
            return tree

        data_left = []
        data_right = []
        for entry in data:
            coord = split['j']
            threshold = split['c']
            if entry['x'][coord] >= threshold:
                data_left.append(entry)
            else:
                data_right.append(entry)
        tree = DecisionTree(data)
        tree.split = split
        tree.left = DecisionTree.gen(data_left)
        tree.right = DecisionTree.gen(data_right)
        return tree

    def decide(self,pt):
        if self.split is None:
            return self.majority
        if pt[self.split['j']] >= self.split['c']:
            return self.left.decide(pt)
        else:
            return self.right.decide(pt)

    def show_decision_region(self,name,savefile,xlim=(0,1), ylim=(0,1)):
        # the strategy is to have points equally spaced
        # throughout and color them based on the decision
        spacing = 500
        xs = [xlim[0] + (xlim[1]-xlim[0])*((i % spacing) / spacing + 0.5/spacing) \
              for i in range(spacing*spacing)]
        ys = [ylim[0] + (ylim[1]-ylim[0])*((i // spacing) / spacing + 0.5/spacing) \
              for i in range(spacing*spacing)]
        decisions = map(lambda x,y: self.decide([x,y]), xs, ys)
        colors = ['#ff8888' if y == 0 else '#8888ff' for y in decisions]

        fig, ax = plt.subplots()
        ax.scatter(xs, ys, s=[2 for i in range(spacing*spacing)], \
                   c=colors, marker='s')

        ax.set(xlim=xlim, ylim=ylim)

        plt.title("Decision Boundary of " + name)
        if savefile is not None:
            plt.savefig(savefile)
        plt.show()

    def count_nodes(self):
        if self.left is None and self.right is None:
            return 1
        else:
            return 1 + self.left.count_nodes() + self.right.count_nodes()

def p2plot():
    fig, ax = plt.subplots()

    ax.scatter([0,1,0,1],[0,0,1,1],c=['#ff0000','#0000ff','#0000ff','#ff0000'])
    plt.show()

def p3():
    data = parse_hw2('data/Druns.txt')
    splits = candidate_splits(data)

    e = data_entropy(data)
    for split in splits:
        print('Candidate split x_{0} >= {1:.3f}:'.format(split['j']+1, split['c']))
        se = split_entropy(data, split)
        ce = cond_entropy(data, split)
        if se <= 1.0e-7:
            print('    Split has no entropy')
            print('    Split has information gain {0:.3f}'.format(e-ce))
        else:
            print('    Gain ratio: {0:.3f}'.format((e-ce)/se))
        print('')

def p4():
    data = parse_hw2('data/D3leaves.txt')
    tree = DecisionTree.gen(data)
    tree.show()

def p5_1():
    data = parse_hw2('data/D1.txt')
    tree = DecisionTree.gen(data)
    tree.show()

def p5_2():
    data = parse_hw2('data/D2.txt')
    tree = DecisionTree.gen(data)
    tree.show()

def scatter(filename, savefile):
    data = parse_hw2(filename)
    x0s = [entry['x'][0] for entry in data]
    x1s = [entry['x'][1] for entry in data]
    ys = [entry['y'] for entry in data]
    colors = ['#ff0000' if y == 0 else '#0000ff' for y in ys]

    fig, ax = plt.subplots()
    ax.scatter(x0s, x1s, c=colors)

    ax.set(xlim=(0,1), ylim=(0,1))

    plt.title("Scatterplot of " + filename)
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()

def show_decision_region(filename, savefile):
    data = parse_hw2(filename)
    tree = DecisionTree.gen(data)
    tree.show_decision_region(filename,savefile)

def p6():
    scatter('data/D1.txt', 'D1plot.png')
    show_decision_region('data/D1.txt', 'D1region.png')
    scatter('data/D2.txt', 'D2plot.png')
    show_decision_region('data/D2.txt', 'D2region.png')

def p7():
    data = parse_hw2('data/Dbig.txt')
    data = np.random.permutation(data)
    test_data = data[8192:]

    data_size = [32, 128, 512, 2048, 8192]
    evaluation = []
    for n in data_size:
        tree = DecisionTree.gen(data[:n])
        correct = 0
        for entry in test_data:
            y1 = tree.decide(entry['x'])
            if y1 == entry['y']:
                correct += 1
        print(' n = ' + str(n))
        print(' nodes: ' + str(tree.count_nodes()))
        print(' err_n: ' + str(1 - correct/len(test_data)))
        evaluation.append(1 - correct/len(test_data))
        tree.show_decision_region("Decision Boundary of Dbig.txt for n="+str(n),
                                  "DbigRegion"+str(n)+".png",
                                  xlim=(-1.5,1.5),
                                  ylim=(-1.5,1.5))
    fig,ax = plt.subplots()

    ax.plot(list(range(len(data_size))), evaluation, marker="o")

    ax.set_xticks(list(range(len(data_size))), data_size)

    plt.xlabel("Number of data points")
    plt.ylabel("Test Error")
    plt.title("Homework Decision Tree Learning Curve")
    plt.savefig("DbigError.png")
    plt.show()

def part3():
    data = parse_hw2('data/Dbig.txt')
    data = np.random.permutation(data)
    test_data = data[8192:]

    data_size = [32, 128, 512, 2048, 8192]
    evaluation = []
    for n in data_size:
        clf = tree.DecisionTreeClassifier()
        xs = [entry['x'] for entry in data[:n]]
        ys = [entry['y'] for entry in data[:n]]
        clf = clf.fit(xs, ys)

        correct = 0
        for entry in test_data:
            y1 = clf.predict([entry['x']])
            if y1 == entry['y']:
                correct += 1
        print(' n = '+str(n))
        print(' nodes: '+str(clf.tree_.node_count))
        print(' err_n: '+str(1 - correct/len(test_data)))
        evaluation.append(1 - correct/len(test_data))

    fig,ax = plt.subplots()

    ax.plot(list(range(len(data_size))), evaluation, marker="o")

    ax.set_xticks(list(range(len(data_size))), data_size)

    plt.xlabel("Number of data points")
    plt.ylabel("Test Error")
    plt.title("Scikit-learn Decision Tree Learning Curve")
    plt.savefig("SKLearnError.png")
    plt.show()

def part4():
    pass
