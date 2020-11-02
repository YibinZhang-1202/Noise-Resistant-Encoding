from pyxdameraulevenshtein import damerau_levenshtein_distance
import distance
import copy


def dist(a, b):
    # return distance.jaccard(a, b)
    return damerau_levenshtein_distance(a, b)


class Token:
    def __init__(self, word, freq):
        self.word = word
        self.freq = freq
        self.hn_card = -1


class Node:
    def __init__(self, word, freq):
        word = Token(word, freq)
        self.cluster_list = [word]
        self.representative = word
        self.share_typo_neighbors = []
        self.cluster_fid_score = 0
        self.cluster_stab_score = 0

    def merge_node(self, self_index, to_merge, to_merge_index, nodes_list):
        self.cluster_list += to_merge.cluster_list

        if to_merge.representative.freq > self.representative.freq:
            self.representative = to_merge.representative

        to_merge.share_typo_neighbors.remove(self_index)

        for x in to_merge.share_typo_neighbors:
            nodes_list[x].share_typo_neighbors.remove(to_merge_index)
            if self_index not in nodes_list[x].share_typo_neighbors:
                nodes_list[x].share_typo_neighbors.append(self_index)

        self.share_typo_neighbors = list(set(self.share_typo_neighbors + to_merge.share_typo_neighbors))

        return self

    def print_node(self):
        print(self.representative.word, self.representative.freq, self.share_typo_neighbors)

    def belong_node(self, word):
        cluster_max_freq = -1
        for x in self.cluster_list:
            if dist(x.word, word) <= Graph.k and x.freq > cluster_max_freq:
                cluster_max_freq = x.freq
        return cluster_max_freq

    def contain(self, word):
        for x in self.cluster_list:
            if x.word == word:
                return True
        return False


class Graph:
    k = None

    def __init__(self, word_frequency_map, gamma, k):
        self.nodes_list = []
        self.word_frequency_map = word_frequency_map
        self.sum_freq = 0
        self.gamma = gamma
        self.ignore_indices = []
        Graph.k = k

        print("K = ", Graph.k)
        # Cost N
        for x in word_frequency_map:
            self.nodes_list.append(Node(x, self.word_frequency_map[x]))
            self.sum_freq = self.sum_freq + self.word_frequency_map[x]

        # Cost N^2
        print("\nBuilding share-typo edges done.")
        self.build_sharetypo_edge()
        print("Build share-typo edges done.\n")

        # Cost N
        self.total_fid_score = self.fid_score()
        self.total_stab_score = self.stab_score()

        self.objective_score = self.gamma * self.total_fid_score + (1 - self.gamma) * self.total_stab_score

        print("\nBegin Clustering.")
        print("Initial objective is ", self.objective_score)
        # Cost E
        if gamma != 1:
            self.agg_cluster()
        print("Clustering done.\n")


    def agg_cluster(self):
        outer_index = 0

        while outer_index < len(self.nodes_list):
            if outer_index in self.ignore_indices:
                outer_index += 1
                continue

            outer_cluster = self.nodes_list[outer_index]
            while len(outer_cluster.share_typo_neighbors) > 0:
                inner_index = outer_cluster.share_typo_neighbors.pop(0)

                if self.nodes_list[outer_index] is self.nodes_list[inner_index]:
                    continue

                # print(outer_index, len(outer_cluster.share_typo_neighbors))

                graph_copy = copy.deepcopy(self)
                graph_copy.merge_cluster(outer_index, inner_index)

                if graph_copy.objective_score > self.objective_score:
                    print('Merge, new objective is ', graph_copy.objective_score)
                    self.merge_cluster(outer_index, inner_index)
                else:
                    # Has checked AB already, don't have to check BA
                    if outer_index in self.nodes_list[inner_index].share_typo_neighbors:
                        self.nodes_list[inner_index].share_typo_neighbors.remove(outer_index)

            outer_index += 1

        self.ignore_indices.sort(reverse=True)
        for x in self.ignore_indices:
            self.nodes_list.pop(x)

        print(self.total_fid_score, self.total_stab_score, self.objective_score)

    def merge_cluster(self, index_1, index_2):
        cluster_1 = self.nodes_list[index_1]
        cluster_2 = self.nodes_list[index_2]

        self.total_fid_score = self.total_fid_score - cluster_1.cluster_fid_score - cluster_2.cluster_fid_score
        for c_index in set(cluster_1.share_typo_neighbors + cluster_2.share_typo_neighbors).union({index_1, index_2}):
            self.total_stab_score = self.total_stab_score - self.nodes_list[c_index].cluster_stab_score

        self.nodes_list[index_1] = cluster_1.merge_node(index_1, cluster_2, index_2, self.nodes_list)
        self.nodes_list[index_2] = self.nodes_list[index_1]

        self.total_fid_score = self.total_fid_score + self.update_cluster_fid_score(self.nodes_list[index_1])
        for c_index in set(self.nodes_list[index_1].share_typo_neighbors).union({index_1}):
            self.total_stab_score = self.total_stab_score + self.update_cluster_stab_score(self.nodes_list[c_index])

        self.ignore_indices.append(index_2)

        # self.objective_score = self.gamma * self.fid_score() + (1 - self.gamma) * self.stab_score()
        self.objective_score = self.gamma * self.total_fid_score + (1 - self.gamma) * self.total_stab_score

    def stab_score(self):
        total_stab_score = 0
        for index, the_cluster in enumerate(self.nodes_list):
            if index in self.ignore_indices:
                continue
            total_stab_score += self.update_cluster_stab_score(the_cluster)
        self.total_stab_score = total_stab_score
        return total_stab_score

    def fid_score(self):
        total_fid_score = 0
        for index, the_cluster in enumerate(self.nodes_list):
            if index in self.ignore_indices:
                continue
            total_fid_score += self.update_cluster_fid_score(the_cluster)
        self.total_fid_score = total_fid_score
        return total_fid_score

    def update_cluster_stab_score(self, the_cluster):
        cluster_stab_score = 0
        for word in the_cluster.cluster_list:
            count = 0
            for neighbor_c in the_cluster.share_typo_neighbors:
                for token in self.nodes_list[neighbor_c].cluster_list:
                    if dist(word.word, token.word) <= Graph.k and token.freq >= word.freq:
                        count = count + 1
                        break
            word.hn_card = count

            cluster_stab_score = cluster_stab_score + word.freq * word.hn_card
            # cluster_stab_score = cluster_stab_score + (word.freq / self.sum_freq) * word.hn_card
        the_cluster.cluster_stab_score = -1 * cluster_stab_score
        return -1 * cluster_stab_score

    def update_cluster_fid_score(self, the_cluster):
        cluster_fid_score = 0
        for word in the_cluster.cluster_list:
            cluster_fid_score = cluster_fid_score + word.freq * dist(word.word, the_cluster.representative.word)
            # cluster_fid_score = cluster_fid_score + (word.freq / self.sum_freq) * dist(word, the_cluster.representative)
        the_cluster.cluster_fid_score = -1 * cluster_fid_score
        return -1 * cluster_fid_score

    def build_sharetypo_edge(self):
        for outer_index, out_cluster in enumerate(self.nodes_list):
            for inner_index in range(outer_index+1, len(self.nodes_list)):
                inner_cluster = self.nodes_list[inner_index]
                if Graph.share_typos(out_cluster.cluster_list, inner_cluster.cluster_list):
                    out_cluster.share_typo_neighbors.append(inner_index)
                    inner_cluster.share_typo_neighbors.append(outer_index)

    def print_graph(self):
        for x in self.nodes_list:
            x.print_node()
        print("There are", len(self.nodes_list), "clusters.")
        print("Gamma =", self.gamma)

    def encode_word(self, word):
        for x in self.nodes_list:
            if x.contain(word):
                return x.representative.word

        # print("b")
        max_freq = 0
        encoded_word = None
        for x in self.nodes_list:
            if x.belong_node(word) > max_freq:
                encoded_word = x.representative.word

        return encoded_word

    @staticmethod
    def share_typos(c1, c2):
        for e_c1 in c1:
            for e_c2 in c2:
                if dist(e_c1.word, e_c2.word) <= Graph.k:
                    return True
        return False


def cluster(word_frequency_map, gamma, k):
    print("There are", len(word_frequency_map), "cluster originally.")

    cluster_graph = Graph(word_frequency_map, gamma, k)

    return cluster_graph
