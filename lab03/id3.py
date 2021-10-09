import copy
import math
import pprint
import random

tree = {}

atrib_idx = {'buying': 0,
             'maint': 1,
             'doors': 2,
             'persons': 3,
             'lug_boot': 4,
             'safety': 5}

classes = ['acc', 'unacc', 'good', 'vgood']


def information_gain(X, values, idx):
    e_x = calc_entropy(X)
    sum = 0.0
    card_x = len(X)

    vals_counter = {v: 0 for v in values}
    for x in X:
        vals_counter[x[idx]] += 1

    for v in values:
        filtered_x = []
        for x in X:
            if x[idx] == v:
                filtered_x.append(x)
        sum += vals_counter[v] / card_x * calc_entropy(filtered_x)

    return e_x - sum


def calc_entropy(X):
    entropy = 0.0
    card_x = len(X)
    entropy_dict = {c: 0 for c in classes}
    for x in Xs:
        entropy_dict[x[6]] += 1
    for key, value in entropy_dict.items():
        val = value / card_x
        entropy += val * math.log2(val)
    return -entropy


def id3(d, Xs, atribs, path):
    c = Xs[0][6]
    ok = True
    for x in Xs[1:]:
        if c != x[6]:
            ok = False
            break
    if ok:
        current_dict = tree
        for key in path[:-1]:
            current_dict = current_dict[key]
        current_dict[path[-1]] = c
        return

    if len(atribs) == 0 or d == 0:
        current_dict = tree
        for key in path[:-1]:
            current_dict = current_dict[key]

        counters = [0, 0, 0, 0]
        for x in Xs:
            c = x[6]
            if c == classes[0]:
                counters[0] += 1
            elif c == classes[1]:
                counters[1] += 1
            elif c == classes[2]:
                counters[2] += 1
            elif c == classes[3]:
                counters[3] += 1

        result = classes[counters.index(max(counters))]
        current_dict[path[-1]] = result
        return

    p1 = copy.deepcopy(path)
    max_ig = 0
    a_star = None
    for atrib, atrib_values in atribs.items():
        ig = information_gain(Xs, atrib_values, atrib_idx[atrib])
        if ig > max_ig:
            max_ig = ig
            a_star = atrib
    ai = a_star
    p1.append(ai)

    current_dict = tree
    for atrib in path:
        current_dict = current_dict[atrib]
    current_dict[ai] = {}

    for v in atribs[ai]:
        X = []
        p2 = copy.deepcopy(p1)
        for x in Xs:
            if x[atrib_idx[ai]] == v:
                X.append(x)
        atr = copy.deepcopy(atribs)
        atr.pop(ai)
        current_dict[ai][v] = {}
        p2.append(v)
        id3(d - 1, X, atr, p2)


def random_tree(d, Xs, atrib, path):
    if d == 0:
        current_dict = tree
        for key in path[:-1]:
            current_dict = current_dict[key]

        counters = [0, 0, 0, 0]
        for x in Xs:
            c = x[6]
            if c == classes[0]:
                counters[0] += 1
            elif c == classes[1]:
                counters[1] += 1
            elif c == classes[2]:
                counters[2] += 1
            elif c == classes[3]:
                counters[3] += 1

        result = classes[counters.index(max(counters))]
        current_dict[path[-1]] = result
    else:
        p1 = copy.deepcopy(path)
        ai = random.choice(list(atrib.keys()))
        p1.append(ai)

        current_dict = tree
        for key in path:
            current_dict = current_dict[key]
        current_dict[ai] = {}
        for v in atrib[ai]:
            X = []
            p2 = copy.deepcopy(p1)
            for x in Xs:
                if x[atrib_idx[ai]] == v:
                    X.append(x)
            atr = copy.deepcopy(atrib)
            atr.pop(ai)
            current_dict[ai][v] = {}
            p2.append(v)
            random_tree(d - 1, X, atr, p2)


def random_forest(Xs, atribs, n, d):
    global tree
    trees = []
    for i in range(n):
        random_tree(d, Xs, atribs, [])
        trees.append(copy.deepcopy(tree))
        tree = {}
    return trees


def id3_forest(Xs, atribs, n, d):
    global tree
    trees = []
    for i in range(n):
        id3(d, Xs, atribs, [])
        trees.append(copy.deepcopy(tree))
        tree = {}
    return trees


if __name__ == '__main__':
    Xs = []
    atribs = {'buying': ['vhigh', 'high', 'med', 'low'],
              'maint': ['vhigh', 'high', 'med', 'low'],
              'doors': ['2', '3', '4', '5more'],
              'persons': ['2', '4', 'more'],
              'lug_boot': ['small', 'med', 'big'],
              'safety': ['low', 'med', 'high']}

    with open("car.data.txt", "r") as f:
        for line in f:
            Xs.append(line[:-1].split(','))
    # random_tree(2, Xs, atribs, [])
    # id3(Xs, atribs, [])
    # trees = random_forest(Xs, atribs, n=2, d=2)
    trees = id3_forest(Xs, atribs, n=2, d=2)
    pprint.pprint(trees[0])
    pprint.pprint(trees[1])
