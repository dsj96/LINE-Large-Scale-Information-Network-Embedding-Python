weighted_edge_list = []

with open('data/data02/karate.edgelist', "r") as graphfile:
    for l in graphfile:
        s = l.strip().split(" ")
        weighted_edge_list.append([s[0], s[1], '1'])

with open('data/data02/weighted.karate.edgelist', "w+") as w:
    for l in weighted_edge_list:
        w.writelines(" ".join([str(s) for s in l]) + '\n')
