import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
                in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose_hands':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (2, 1), (1, 0),
                             (8, 7), (7, 6), (6, 5), (5, 0),
                             (12, 11), (11, 10), (10, 9), (9, 0),
                             (16, 15), (15, 14), (14, 13), (13, 0),
                             (20, 19), (19, 18), (18, 17), (17, 0)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'openpose_with_hands':
            self.num_node = 18 + 20 * 2
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_body = [(4, 3), (3, 2), (7, 6), (6, 5),
                                  (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                                  (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                                  (17, 15), (16, 14)]
            neighbor_link_hand1 = [(4, 3), (3, 2), (2, 1), (1, -13),
                                   (8, 7), (7, 6), (6, 5), (5, -13),
                                   (12, 11), (11, 10), (10, 9), (9, -13),
                                   (16, 15), (15, 14), (14, 13), (13, -13),
                                   (20, 19), (19, 18), (18, 17), (17, -13)]
            neighbor_link_hand1 = [(x[0] + 17, x[1] + 17) for x in neighbor_link_hand1]
            neighbor_link_hand2 = [(4, 3), (3, 2), (2, 1), (1, -30),
                                   (8, 7), (7, 6), (6, 5), (5, -30),
                                   (12, 11), (11, 10), (10, 9), (9, -30),
                                   (16, 15), (15, 14), (14, 13), (13, -30),
                                   (20, 19), (19, 18), (18, 17), (17, -30)]
            neighbor_link_hand2 = [(x[0] + 17 + 20, x[1] + 17 + 20) for x in neighbor_link_hand2]
            self.edge = self_link + neighbor_link_body + neighbor_link_hand1 + neighbor_link_hand2
            self.center = 0
        elif layout == 'upperbody_with_hands':
            self.num_node = 7 + 20 * 2
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_body = [(3, 2), (2, 1), (1, 0),
                                  (6, 5), (5, 4), (4, 0)]
            neighbor_link_hand1 = [(4, 3), (3, 2), (2, 1), (1, -3),
                                   (8, 7), (7, 6), (6, 5), (5, -3),
                                   (12, 11), (11, 10), (10, 9), (9, -3),
                                   (16, 15), (15, 14), (14, 13), (13, -3),
                                   (20, 19), (19, 18), (18, 17), (17, -3)]
            neighbor_link_hand1 = [(x[0] + 6, x[1] + 6) for x in neighbor_link_hand1]
            neighbor_link_hand2 = [(4, 3), (3, 2), (2, 1), (1, -20),
                                   (8, 7), (7, 6), (6, 5), (5, -20),
                                   (12, 11), (11, 10), (10, 9), (9, -20),
                                   (16, 15), (15, 14), (14, 13), (13, -20),
                                   (20, 19), (19, 18), (18, 17), (17, -20)]
            neighbor_link_hand2 = [(x[0] + 6 + 20, x[1] + 6 + 20) for x in neighbor_link_hand2]
            self.edge = self_link + neighbor_link_body + neighbor_link_hand1 + neighbor_link_hand2
            self.center = 0
        elif layout == 'upperbody':
            self.num_node = 7
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link_body = [(3, 2), (2, 1), (1, 0),
                                  (6, 5), (5, 4), (4, 0)]
            self.edge = self_link + neighbor_link_body
            self.center = 0
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'acappella':
            self.num_node = 68
            self_link = [(i, i) for i in range(self.num_node)]
            all = [(i, i + 1) for i in range(68)]

            face = all[slice(0, 16)]
            eyebrown1 = all[slice(17, 21)]
            eyebrown2 = all[slice(22, 26)]
            nose = all[slice(27, 30)]
            nostril = all[slice(31, 35)]
            eye1 = all[slice(36, 41)]
            eye2 = all[slice(42, 47)]
            lips = all[slice(48, 59)]
            teeth = all[slice(60, 67)]
            self.edge = self_link + face + eye1 + eye2 + eyebrown1 + eyebrown2 + nose + nostril + lips + teeth
            self.center = 0
            # ORIGINAL SOURCECODE
            # https://github.com/1adrianb/face-alignment/blob/master/examples/detect_landmarks_in_image.py
            # pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
            #               'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
            #               'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
            #               'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
            #               'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
            #               'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
            #               'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
            #               'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
            #               'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
            #               }
        else:
            raise ValueError("Do Not Exist This Layout.")
        return self.edge, self.num_node

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
