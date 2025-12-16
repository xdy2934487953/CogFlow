import argparse
import itertools
import numpy as np
import tensorflow as tf
import time
from collections import defaultdict
from functools import partial
import os
import tensorflow.keras as keras
import copy
import shutil
import argparse

# FB237
def get_fb237_config(parser):
    parser.add_argument('--dataset', default='FB237')

    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=200)

    parser.add_argument('--test_batch_size', type=int, default=80)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=200)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=2)
    parser.add_argument('--con_steps', type=int, default=6)

    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=True)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--eval_valid', action='store_true', default=False)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    parser.add_argument('--test_output_attention', action='store_true', default=False)
    parser.add_argument('--test_analyze_attention', action='store_true', default=False)

    return parser


# FB15K
def get_fb15k_config(parser):
    parser.add_argument('--dataset', default='FB15K')

    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=200)

    parser.add_argument('--test_batch_size', type=int, default=80)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=200)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=1)
    parser.add_argument('--con_steps', type=int, default=6)

    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--eval_valid', action='store_true', default=False)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    parser.add_argument('--test_output_attention', action='store_true', default=False)
    parser.add_argument('--test_analyze_attention', action='store_true', default=False)

    return parser


# WN18RR
def get_wn18rr_config(parser):
    parser.add_argument('--dataset', default='WN18RR')

    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=200)

    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=200)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=2)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--eval_valid', action='store_true', default=False)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    parser.add_argument('--test_output_attention', action='store_true', default=False)
    parser.add_argument('--test_analyze_attention', action='store_true', default=False)

    return parser


# WN
def get_wn_config(parser):
    parser.add_argument('--dataset', default='WN')

    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=200)

    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=200)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=1)
    parser.add_argument('--con_steps', type=int, default=8)

    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--eval_valid', action='store_true', default=False)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    parser.add_argument('--test_output_attention', action='store_true', default=False)
    parser.add_argument('--test_analyze_attention', action='store_true', default=False)

    return parser


# YAGO310
def get_yago310_config(parser):
    parser.add_argument('--dataset', default='YAGO310')

    parser.add_argument('--n_dims_sm', type=int, default=50)
    parser.add_argument('--n_dims', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=200)
    parser.add_argument('--max_attended_nodes', type=int, default=20)
    parser.add_argument('--max_seen_nodes', type=int, default=200)

    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=200)
    parser.add_argument('--test_max_attended_nodes', type=int, default=20)
    parser.add_argument('--test_max_seen_nodes', type=int, default=200)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=1)
    parser.add_argument('--con_steps', type=int, default=6)

    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--clipnorm', type=float, default=1.)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--eval_valid', action='store_true', default=False)
    parser.add_argument('--moving_mean_decay', type=float, default=0.99)

    parser.add_argument('--test_output_attention', action='store_true', default=False)
    parser.add_argument('--test_analyze_attention', action='store_true', default=False)

    return parser


# Nell995: for separate learning per subset
def get_nell995_separate_config(parser):
    parser.add_argument('--dataset', default='NELL995')

    parser.add_argument('--n_dims_sm', type=int, default=200)
    parser.add_argument('--n_dims', type=int, default=200)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_edges_per_example', type=int, default=10000)
    parser.add_argument('--max_edges_per_node', type=int, default=1000)
    parser.add_argument('--max_attended_nodes', type=int, default=100)
    parser.add_argument('--max_seen_nodes', type=int, default=1000)

    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--test_max_edges_per_example', type=int, default=10000)
    parser.add_argument('--test_max_edges_per_node', type=int, default=1000)
    parser.add_argument('--test_max_attended_nodes', type=int, default=100)
    parser.add_argument('--test_max_seen_nodes', type=int, default=1000)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--aggregate_op', default='mean_v3')
    parser.add_argument('--uncon_steps', type=int, default=1)
    parser.add_argument('--con_steps', type=int, default=5)

    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipnorm', type=float, default=1.)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=False)

    parser.add_argument('--timer', action='store_true', default=False)
    parser.add_argument('--print_train', action='store_true', default=True)
    parser.add_argument('--print_train_metric', action='store_true', default=True)
    parser.add_argument('--print_train_freq', type=int, default=1)
    parser.add_argument('--eval_within_epoch', default=[])
    parser.add_argument('--eval_valid', action='store_true', default=False)
    parser.add_argument('--moving_mean_decay', type=float, default=0.9)

    parser.add_argument('--test_output_attention', action='store_true', default=False)
    parser.add_argument('--test_analyze_attention', action='store_true', default=False)

    return parser


def get_default_config(name):
    parser = argparse.ArgumentParser()
    if name == 'FB237' or name == 'FB237_v2':
        return get_fb237_config(parser)
    elif name == 'FB15K':
        return get_fb15k_config(parser)
    elif name == 'WN18RR' or name == 'WN18RR_v2':
        return get_wn18rr_config(parser)
    elif name == 'WN':
        return get_wn_config(parser)
    elif name == 'YAGO310':
        return get_yago310_config(parser)
    elif name == 'NELL995':
        return get_nell995_separate_config(parser)
    else:
        raise ValueError('Invalid `name`')
    



def get(dct, k):
    return dct.get(k, None) if isinstance(dct, dict) else None


def get_segment_ids(x):
    """ x: (np.array) d0 x 2, sorted
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    y = (x[1:] == x[:-1]).astype('uint8')
    return np.concatenate([np.array([0], dtype='int32'),
                           np.cumsum(1 - y[:, 0] * y[:, 1], dtype='int32')])


def get_unique(x):
    """ x: (np.array) d0 x 2, sorted
    """
    if len(x) == 0:
        return x

    y = (x[1:] == x[:-1]).astype('uint8')
    return x[np.concatenate([np.array([1], dtype='bool'),
                             (1 - y[:, 0] * y[:, 1]).astype('bool')])]


def groupby_2cols_nlargest(x, y, k):
    """ x: (np.array) d0 x 2, sorted
        y: (np.array) d1
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    mask = (x[1:] == x[:-1]).astype('uint8')
    mask = (1 - mask[:, 0] * mask[:, 1]).astype('bool')
    n = len(x)
    key_idx = np.concatenate([np.array([0], dtype='int32'),
                              np.arange(1, n).astype('int32')[mask],
                              np.array([n], dtype='int32')])
    res_idx = np.concatenate([np.sort(s + np.argpartition(-y[s:e], min(k - 1, e - s - 1))[:min(k, e - s)])
                              for s, e in zip(key_idx[:-1], key_idx[1:])])
    return res_idx.astype('int32')


def groupby_1cols_nlargest(x, y, k):
    """ x: (np.array) d0, sorted
        y: (np.array) d1
    """
    if len(x) == 0:
        return np.array([0], dtype='int32')

    mask = (x[1:] != x[:-1])
    n = len(x)
    key_idx = np.concatenate([np.array([0], dtype='int32'),
                              np.arange(1, n).astype('int32')[mask],
                              np.array([n], dtype='int32')])
    res_idx = np.concatenate([np.sort(s + np.argpartition(-y[s:e], min(k - 1, e - s - 1))[:min(k, e - s)])
                              for s, e in zip(key_idx[:-1], key_idx[1:])])
    return res_idx.astype('int32')


def groupby_1cols_merge(x, x_key, y_key, y_id):
    """ x (group by): (np.array) d0, sorted
        x_key (merge left key): (np.array): d0, unique in group
        y_key (merge right key): (np.array): d1
        y_id: (np.array): d1
    """
    mask = (x[1:] != x[:-1])
    n = len(x)
    key_idx = np.concatenate([np.array([0], dtype='int32'),
                              np.arange(1, n).astype('int32')[mask],
                              np.array([n], dtype='int32')])
    yid_li = [y_id[np.in1d(y_key, x_key[s:e])]
              for s, e in zip(key_idx[:-1], key_idx[1:])]
    res_idx = np.concatenate(yid_li)
    grp_idx = np.concatenate([np.repeat(np.array([i], dtype='int32'), len(yid)) for i, yid in enumerate(yid_li)])
    return res_idx, grp_idx


def groupby_1cols_cartesian(x, v1, y, v2):
    """ x: d0, sorted
        v1: d0
        y: d1, sorted
        v2: d1
    """
    mask = (x[1:] != x[:-1])
    n = len(x)
    x_key_idx = np.concatenate([np.array([0], dtype='int32'),
                                np.arange(1, n).astype('int32')[mask],
                                np.array([n], dtype='int32')])
    mask = (y[1:] != y[:-1])
    n = len(y)
    y_key_idx = np.concatenate([np.array([0], dtype='int32'),
                                np.arange(1, n).astype('int32')[mask],
                                np.array([n], dtype='int32')])
    batch_size = len(x_key_idx) - 1
    return np.array([(eg_idx, vi, vj)
                     for eg_idx, s1, e1, s2, e2 in zip(np.arange(batch_size),
                                                       x_key_idx[:-1],
                                                       x_key_idx[1:],
                                                       y_key_idx[:-1],
                                                       y_key_idx[1:])
                     for vi, vj in itertools.product(v1[s1:e1], v2[s2:e2])], dtype='int32')


def entropy(x):
    return tf.reduce_sum(- tf.math.log(tf.math.maximum(x, 1e-20)) * x, axis=-1)


def topk_occupy(x, k):
    values, _ = tf.math.top_k(x, k=k)
    return tf.reduce_sum(values, axis=-1) / tf.reduce_sum(x, axis=-1)

class Graph(object):
    def __init__(self, graph_triples, n_ents, n_rels, reversed_rel_dct):
        self.reversed_rel_dct = reversed_rel_dct

        full_edges = np.array(graph_triples.tolist(), dtype='int32').view('<i4,<i4,<i4')
        full_edges = np.sort(full_edges, axis=0, order=['f0', 'f1', 'f2']).view('<i4')
        # `full_edges`: use all train triples
        # full_edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending and consecutive `id`s
        self.full_edges = np.concatenate([np.expand_dims(np.arange(len(full_edges), dtype='int32'), 1),
                                          full_edges], axis=1)
        self.n_full_edges = len(self.full_edges)

        self.n_entities = n_ents
        self.selfloop = n_rels
        self.n_relations = n_rels + 1

        # `edges`: for current train batch
        # edges[i] = [id, head, tail, rel] sorted by head, tail, rel with ascending but not consecutive `id`s
        self.edges = None
        self.n_edges = 0

        # `memorized_nodes`: for current train batch
        self.memorized_nodes = None  # (np.array) (eg_idx, v) sorted by ed_idx, v

    def make_temp_edges(self, batch, remove_all_head_tail_edges=True):
        """ batch: (np.array) (head, tail, rel)
        """
        if remove_all_head_tail_edges:
            batch_set = set([(h, t) for h, t, r in batch])
            edges_idx = [i for i, (eid, h, t, r) in enumerate(self.full_edges)
                         if (h, t) not in batch_set and (t, h) not in batch_set]

        else:
            batch_set = set([(h, t, r) for h, t, r in batch])
            if self.reversed_rel_dct is None:
                edges_idx = [i for i, (eid, h, t, r) in enumerate(self.full_edges)
                             if (h, t, r) not in batch_set]
            else:
                edges_idx = [i for i, (eid, h, t, r) in enumerate(self.full_edges)
                             if (h, t, r) not in batch_set and (t, h, self.reversed_rel_dct.get(r, -1)) not in batch_set]
        self.edges = self.full_edges[edges_idx]
        self.n_edges = len(self.edges)

    def use_full_edges(self):
        self.edges = self.full_edges
        self.n_edges = len(self.edges)

    def get_candidate_edges(self, attended_nodes=None, tc=None):
        """ attended_nodes:
            (1) None: use all graph edges with batch_size=1
            (2) (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        """
        if tc is not None:
            t0 = time.time()

        if attended_nodes is None:
            candidate_edges = np.concatenate([np.zeros((self.n_edges, 1), dtype='int32'),
                                              self.edges], axis=1)  # (0, edge_id, vi, vj, rel) sorted by (0, edge_id)
        else:
            candidate_idx, new_eg_idx = groupby_1cols_merge(attended_nodes[:, 0], attended_nodes[:, 1],
                                                            self.edges[:, 1], self.edges[:, 0])
            if len(candidate_idx) == 0:
                return np.zeros((0, 5), dtype='int32')

            candidate_edges = np.concatenate([np.expand_dims(new_eg_idx, 1),
                                              self.full_edges[candidate_idx]], axis=1)  # (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)

        if tc is not None:
            tc['candi_e'] += time.time() - t0
        # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel)
        #   sorted by (eg_idx, edge_id) or (eg_idx, vi, vj, rel)
        return candidate_edges

    def get_sampled_edges(self, candidate_edges, mode=None, max_edges_per_eg=None, max_edges_per_vi=None, tc=None):
        """ candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
        """
        assert mode is not None
        if tc is not None:
            t0 = time.time()

        if len(candidate_edges) == 0:
            return np.zeros((0, 6), dtype='int32')

        logits = tf.random.uniform((len(candidate_edges),))
        if mode == 'by_eg':
            assert max_edges_per_eg is not None
            sampled_edges = candidate_edges[:, 0]  # n_candidate_edges
            sampled_idx = groupby_1cols_nlargest(sampled_edges, logits, max_edges_per_eg)  # n_sampled_edges
            sampled_edges = np.concatenate([candidate_edges[sampled_idx],
                                            np.expand_dims(sampled_idx, 1)], axis=1)  # n_sampled_edges x 6
        elif mode == 'by_vi':
            assert max_edges_per_vi is not None
            sampled_edges = candidate_edges[:, [0, 2]]  # n_candidate_edges x 2
            sampled_idx = groupby_2cols_nlargest(sampled_edges, logits, max_edges_per_vi)  # n_sampled_edges
            sampled_edges = np.concatenate([candidate_edges[sampled_idx],
                                            np.expand_dims(sampled_idx, 1)], axis=1)  # n_sampled_edges x 6
        else:
            raise ValueError('Invalid `mode`')

        if tc is not None:
            tc['sampl_e'] += time.time() - t0
        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx)
        #   sorted by (eg_idx, edge_id)
        return sampled_edges

    def get_selected_edges(self, sampled_edges, tc=None):
        """ sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        """
        if tc is not None:
            t0 = time.time()

        if len(sampled_edges) == 0:
           return np.zeros((0, 6), dtype='int32')

        idx_vi = get_segment_ids(sampled_edges[:, [0, 2]])
        _, idx_vj = np.unique(sampled_edges[:, [0, 3]], axis=0, return_inverse=True)

        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)

        selected_edges = np.concatenate([sampled_edges[:, [0, 2, 3, 4]], idx_vi, idx_vj], axis=1)

        if tc is not None:
            tc['sele_e'] += time.time() - t0
        # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj]
        #   sorted by (eg_idx, vi, vj)
        return selected_edges

    def set_init_memorized_nodes(self, heads, tc=None):
        """ heads: batch_size
        """
        if tc is not None:
            t0 = time.time()

        batch_size = heads.shape[0]
        eg_idx = np.array(np.arange(batch_size), dtype='int32')
        self.memorized_nodes = np.stack([eg_idx, heads], axis=1)

        if tc is not None:
            tc['i_memo_v'] += time.time() - t0
        # memorized_nodes: n_memorized_nodes (=batch_size) x 2, (eg_idx, v) sorted by (ed_idx, v)
        return self.memorized_nodes

    def get_topk_nodes(self, node_attention, max_nodes, tc=None):
        """ node_attention: (tf.Tensor) batch_size x n_nodes
        """
        if tc is not None:
            t0 = time.time()

        eps = 1e-20
        node_attention = node_attention.numpy()
        n_nodes = node_attention.shape[1]
        max_nodes = min(n_nodes, max_nodes)
        sorted_idx = np.argsort(-node_attention, axis=1)[:, :max_nodes]
        sorted_idx = np.sort(sorted_idx, axis=1)
        node_attention = np.take_along_axis(node_attention, sorted_idx, axis=1)  # sorted node attention
        mask = node_attention > eps
        eg_idx = np.repeat(np.expand_dims(np.arange(mask.shape[0]), 1), mask.shape[1], axis=1)[mask].astype('int32')
        vi = sorted_idx[mask].astype('int32')
        topk_nodes = np.stack([eg_idx, vi], axis=1)

        if tc is not None:
            tc['topk_v'] += time.time() - t0
        # topk_nodes: (np.array) n_topk_nodes x 2, (eg_idx, vi) sorted
        return topk_nodes

    def get_selfloop_edges(self, attended_nodes, tc=None):
        """ attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        """
        if tc is not None:
            t0 = time.time()

        eg_idx, vi = attended_nodes[:, 0], attended_nodes[:, 1]
        selfloop_edges = np.stack([eg_idx, vi, vi, np.repeat(np.array(self.selfloop, dtype='int32'), eg_idx.shape[0])],
                                  axis=1)  # (eg_idx, vi, vi, selfloop)

        if tc is not None:
            tc['sl_bt'] += time.time() - t0
        return selfloop_edges  # (eg_idx, vi, vi, selfloop)

    def get_union_edges(self, scanned_edges, selfloop_edges, tc=None):
        """ scanned_edges: (np.array) n_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
            selfloop_edges: (np.array) n_selfloop_edges x 4 (eg_idx, vi, vi, selfloop)
        """
        if tc is not None:
            t0 = time.time()

        scanned_edges = np.zeros((0, 4), dtype='int32') if len(scanned_edges) == 0 else scanned_edges[:, :4]  # (eg_idx, vi, vj, rel)
        all_edges = np.concatenate([scanned_edges, selfloop_edges], axis=0).copy()
        sorted_idx = np.squeeze(np.argsort(all_edges.view('<i4,<i4,<i4,<i4'),
                                           order=['f0', 'f1', 'f2'], axis=0), 1).astype('int32')
        aug_scanned_edges = all_edges[sorted_idx]  # sorted by (eg_idx, vi, vj)
        idx_vi = get_segment_ids(aug_scanned_edges[:, [0, 1]])
        _, idx_vj = np.unique(aug_scanned_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        aug_scanned_edges = np.concatenate([aug_scanned_edges, idx_vi, idx_vj], axis=1)

        if tc is not None:
            tc['union_e'] += time.time() - t0
        # aug_scanned_edges: n_aug_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        return aug_scanned_edges

    def add_nodes_to_memorized(self, selected_edges, inplace=False, tc=None):
        """ selected_edges: (np.array) n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        if len(selected_edges) > 0:
            selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)
            mask = np.in1d(selected_vj.view('<i4,<i4'), self.memorized_nodes.view('<i4,<i4'), assume_unique=True)
            mask = np.logical_not(mask)
            new_nodes = selected_vj[mask]  # n_new_nodes x 2

        if len(selected_edges) > 0 and len(new_nodes) > 0:
            memorized_and_new = np.concatenate([self.memorized_nodes, new_nodes], axis=0)  # n_memorized_and_new_nodes x 2
            sorted_idx = np.squeeze(np.argsort(memorized_and_new.view('<i4,<i4'),
                                               order=['f0', 'f1'], axis=0), 1).astype('int32')

            memorized_and_new = memorized_and_new[sorted_idx]
            n_memorized_and_new_nodes = len(memorized_and_new)

            new_idx = np.argsort(sorted_idx).astype('int32')
            n_memorized_nodes = self.memorized_nodes.shape[0]
            new_idx_for_memorized = np.expand_dims(new_idx[:n_memorized_nodes], 1)

            if inplace:
                self.memorized_nodes = memorized_and_new
        else:
            new_idx_for_memorized = None
            memorized_and_new = self.memorized_nodes
            n_memorized_and_new_nodes = len(memorized_and_new)

        if tc is not None:
            tc['add_scan'] += time.time() - t0
        # new_idx_for_memorized: n_memorized_nodes x 1
        # memorized_and_new: n_memorized_and_new_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        return new_idx_for_memorized, n_memorized_and_new_nodes, memorized_and_new

    def set_index_over_nodes(self, selected_edges, nodes, tc=None):
        """ selected_edges (or aug_selected_edges): n_selected_edges (or n_aug_selected_edges) x 6, sorted
            nodes: (eg_idx, v) unique and sorted
        """
        if tc is not None:
            t0 = time.time()

        if len(selected_edges) == 0:
            return np.zeros((0, 8), dtype='int32')

        selected_vi = get_unique(selected_edges[:, [0, 1]])  # n_selected_edges x 2
        selected_vj = np.unique(selected_edges[:, [0, 2]], axis=0)  # n_selected_edges x 2
        mask_vi = np.in1d(nodes.view('<i4,<i4'), selected_vi.view('<i4,<i4'), assume_unique=True)
        mask_vj = np.in1d(nodes.view('<i4,<i4'), selected_vj.view('<i4,<i4'), assume_unique=True)
        new_idx_e2vi = np.expand_dims(np.arange(mask_vi.shape[0])[mask_vi], 1).astype('int32')  # n_matched_by_idx_and_vi x 1
        new_idx_e2vj = np.expand_dims(np.arange(mask_vj.shape[0])[mask_vj], 1).astype('int32')  # n_matched_by_idx_and_vj x 1

        idx_vi = selected_edges[:, 4]
        idx_vj = selected_edges[:, 5]
        new_idx_e2vi = new_idx_e2vi[idx_vi]  # n_selected_edges x 1
        new_idx_e2vj = new_idx_e2vj[idx_vj]  # n_selected_edges x 1

        # selected_edges: n_selected_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        selected_edges = np.concatenate([selected_edges, new_idx_e2vi, new_idx_e2vj], axis=1)

        if tc is not None:
            tc['idx_v'] += time.time() - t0
        return selected_edges

    def get_seen_edges(self, seen_nodes, aug_scanned_edges, tc=None):
        """ seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique but not sorted
            aug_scanned_edges: (np.array) n_aug_scanned_edges x 8,
                (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        """
        if tc is not None:
            t0 = time.time()

        aug_scanned_vj = aug_scanned_edges[:, [0, 2]].copy()  # n_aug_scanned_edges x 2, (eg_idx, vj) not unique and not sorted
        mask_vj = np.in1d(aug_scanned_vj.view('<i4,<i4'), seen_nodes.view('<i4,<i4'))
        seen_edges = aug_scanned_edges[mask_vj][:, :4]  # n_seen_edges x 4, (eg_idx, vi, vj, rel) sorted by (eg_idx, vi, vj)

        idx_vi = get_segment_ids(seen_edges[:, [0, 1]])
        _, idx_vj = np.unique(seen_edges[:, [0, 2]], axis=0, return_inverse=True)
        idx_vi = np.expand_dims(np.array(idx_vi, dtype='int32'), 1)
        idx_vj = np.expand_dims(np.array(idx_vj, dtype='int32'), 1)
        seen_edges = np.concatenate((seen_edges, idx_vi, idx_vj), axis=1)

        if tc is not None:
            tc['seen_e'] += time.time() - t0
        # seen_edges: n_seen_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        return seen_edges

    def get_vivj_edges(self, vi_nodes, vj_nodes, with_eg_idx=True):
        """ vi_nodes: n_attended_vi_nodes x 2, (eg_idx, vi) or n_attended_vi_nodes, (vi)
            vj_nodes: n_attended_vj_nodes x 2, (eg_idx, vj) or n_attended_vj_nodes, (vj)
        """
        if with_eg_idx:
            candidate_idx, new_eg_idx = groupby_1cols_merge(vi_nodes[:, 0], vi_nodes[:, 1],
                                                            self.edges[:, 1], self.edges[:, 0])
            candidate_edges = np.concatenate([np.expand_dims(new_eg_idx, 1), self.full_edges[candidate_idx]], axis=1)  # (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
            candidate_vj = candidate_edges[:, [0, 3]].copy()  # n_candidate_edges x 2, (eg_idx, vj) not unique and not sorted
            mask_vj = np.in1d(candidate_vj.view('<i4,<i4'), vj_nodes.view('<i4,<i4'))
            vivj_edges = candidate_edges[mask_vj]  # n_vivj_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, vi, vj)
        else:
            candidate_idx = self.edges[:, 0][np.in1d(self.edges[:, 1], vi_nodes)]
            candidate_edges = self.full_edges[candidate_idx]  # (edge_id, vi, vj, rel) sorted by edge_id
            candidate_vj = candidate_edges[:, 2].copy()  # n_candidate_edges
            mask_vj = np.in1d(candidate_vj, vj_nodes)
            vivj_edges = candidate_edges[mask_vj]  # n_vivj_edges x 4, (edge_id, vi, vj, rel) sorted by edge_id

        return vivj_edges


class DataFeeder(object):
    def get_train_batch(self, train_data, graph, batch_size, shuffle=True, remove_all_head_tail_edges=True):
        n_train = len(train_data)
        rand_idx = np.random.permutation(n_train) if shuffle else np.arange(n_train)
        start = 0
        while start < n_train:
            end = min(start + batch_size, n_train)
            pad = max(start + batch_size - n_train, 0)
            batch = np.array([train_data[i] for i in np.concatenate([rand_idx[start:end], rand_idx[:pad]])], dtype='int32')
            graph.make_temp_edges(batch, remove_all_head_tail_edges=remove_all_head_tail_edges)
            yield batch, end - start
            start = end

    def get_eval_batch(self, eval_data, graph, batch_size, shuffle=False):
        n_eval = len(eval_data)
        rand_idx = np.random.permutation(n_eval) if shuffle else np.arange(n_eval)
        start = 0
        while start < n_eval:
            end = min(start + batch_size, n_eval)
            batch = np.array([eval_data[i] for i in rand_idx[start:end]], dtype='int32')
            graph.use_full_edges()
            yield batch, end - start
            start = end


class DataEnv(object):
    def __init__(self, dataset):
        self.data_feeder = DataFeeder()

        self.ds = dataset

        self.valid = dataset.valid
        self.test = dataset.test
        self.train = dataset.train
        self.test_candidates = dataset.test_candidates
        self.test_by_rel = dataset.test_by_rel

        self.graph = Graph(dataset.graph, dataset.n_entities, dataset.n_relations, dataset.reversed_rel_dct)

        self.filter_pool = defaultdict(set)
        for head, tail, rel in np.concatenate([self.train, self.valid, self.test], axis=0):
            self.filter_pool[(head, rel)].add(tail)

    def get_train_batcher(self, remove_all_head_tail_edges=True):
        return partial(self.data_feeder.get_train_batch, self.train, self.graph,
                       remove_all_head_tail_edges=remove_all_head_tail_edges)

    def get_valid_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.valid, self.graph)

    def get_test_batcher(self):
        return partial(self.data_feeder.get_eval_batch, self.test, self.graph)

    def get_test_relations(self):
        return self.test_by_rel.keys() if self.test_by_rel is not None else None

    def get_test_batcher_by_rel(self, rel):
        return partial(self.data_feeder.get_eval_batch, self.test_by_rel[rel], self.graph) \
            if self.test_by_rel is not None else None

    @property
    def n_train(self):
        return len(self.train)

    @property
    def n_valid(self):
        return len(self.valid)

    @property
    def n_test(self):
        return len(self.test)
class Dataset(object):
    def __init__(self, train_path, valid_path, test_path, graph_path=None, test_candidates_path=None,
                 do_reverse=False, do_reverse_on_graph=False, get_reverse=None, has_reverse=False,
                 test_paths=None, test_candidates_paths=None):
        train = self._load_triple_file(train_path)
        valid = self._load_triple_file(valid_path)
        test = self._load_triple_file(test_path)
        graph = self._load_triple_file(graph_path) if graph_path is not None else train

        if test_candidates_path is not None:
            test_candidates = self._load_test_candidates_file(test_candidates_path)
        elif test_candidates_paths is not None:
            test_candidates = {os.path.basename(path).split('_')[-1]: self._load_test_candidates_file(path)
                               for path in test_candidates_paths}
        else:
            test_candidates = None

        if test_paths is not None:
            test_by_rel = {os.path.basename(path).split('_')[-1]: self._load_triple_file(path)
                           for path in test_paths}
        else:
            test_by_rel = None


        if do_reverse:
            train = self._add_reverse_triples(train)
            valid = self._add_reverse_triples(valid)
            test = self._add_reverse_triples(test)
            graph = self._add_reverse_triples(graph)
        elif do_reverse_on_graph:
            graph = self._add_reverse_triples(graph)

        self.entity2id, self.id2entity, self.relation2id, self.id2relation = \
            self._make_dict(graph + train + valid + test)
        self.n_entities = len(self.entity2id)
        self.n_relations = len(self.relation2id)

        self.reversed_rel_dct = None
        if do_reverse or do_reverse_on_graph or has_reverse:
            self.reversed_rel_dct = self._get_reversed_relation_dict(self.relation2id, get_reverse=get_reverse)

        self.train = self._convert_to_id(train)
        self.valid = self._convert_to_id(valid)
        self.test = self._convert_to_id(test)
        self.graph = self._convert_to_id(graph)

        if test_candidates_path is not None:
            self.test_candidates = self._convert_to_id_v2(test_candidates)
        elif test_candidates_paths is not None:
            self.test_candidates = {rel: self._convert_to_id_v2(dct) for rel, dct in test_candidates.items()}
        else:
            self.test_candidates = None

        if test_paths is not None:
            self.test_by_rel = {rel: self._convert_to_id(triples) for rel, triples in test_by_rel.items()}
        else:
            self.test_by_rel = None

    def _load_triple_file(self, filepath):
        triples = []
        with open(filepath) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((h, t, r))
        return triples

    def _load_test_candidates_file(self, filepath):
        test_candidates = defaultdict(dict)
        with open(filepath) as fin:
            for line in fin:
                pair, ans = line.strip().split(': ')
                h, t = pair.split(',')
                h = h.replace('thing$', '')
                t = t.replace('thing$', '')
                test_candidates[h][t] = ans
        return test_candidates

    def _make_dict(self, triples):
        ent2id, rel2id = {}, {}
        id2ent, id2rel = {}, {}
        for h, t, r in triples:
            ent2id.setdefault(h, len(ent2id))
            id2ent[ent2id[h]] = h
            ent2id.setdefault(t, len(ent2id))
            id2ent[ent2id[t]] = t
            rel2id.setdefault(r, len(rel2id))
            id2rel[rel2id[r]] = r
        return ent2id, id2ent, rel2id, id2rel

    def _add_reverse_triples(self, triples):
        return triples + [(t, h, '_' + r) for h, t, r in triples]

    def _get_reversed_relation_dict(self, relation2id, get_reverse=None):
        if get_reverse is None:
            get_reverse = lambda r: '_' + r if r[0] != '_' else r[1:]
        return {id: relation2id[get_reverse(rel)] for rel, id in relation2id.items()
                if get_reverse(rel) in relation2id}

    def _convert_to_id(self, triples):
        return np.array([(self.entity2id[h], self.entity2id[t], self.relation2id[r])
                         for h, t, r in triples], dtype='int32')

    def _convert_to_id_v2(self, answers):
        return {self.entity2id[h]: {self.entity2id[t]: ans for t, ans in t_dct.items()}
                for h, t_dct in answers.items()}


class FB237(Dataset):
    path = '../data/KBC/FB237'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'FB237'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(FB237, self).__init__(train_path, valid_path, test_path,
                                    do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class FB237_v2(Dataset):
    path = '../data/MWalk/FB15K-237'

    def __init__(self):
        self.name = 'FB237_v2'
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        graph_path = os.path.join(self.path, 'graph.txt')
        super(FB237_v2, self).__init__(train_path, valid_path, test_path,
                                       graph_path=graph_path,
                                       has_reverse=True)


class FB15K(Dataset):
    path = '../data/KBC/FB15K'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'FB15K'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(FB15K, self).__init__(train_path, valid_path, test_path,
                                    do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class WN(Dataset):
    path = '../data/KBC/WN'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'WN'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(WN, self).__init__(train_path, valid_path, test_path,
                                 do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class WN18RR(Dataset):
    path = '../data/KBC/WN18RR'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'WN18RR'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(WN18RR, self).__init__(train_path, valid_path, test_path,
                                     do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class WN18RR_v2(Dataset):
    path = '../data/MWalk/WN18RR'

    def __init__(self):
        self.name = 'WN18RR_v2'
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        graph_path = os.path.join(self.path, 'graph.txt')
        super(WN18RR_v2, self).__init__(train_path, valid_path, test_path,
                                        graph_path=graph_path,
                                        has_reverse=True)


class YAGO310(Dataset):
    path = '../data/KBC/YAGO3-10'

    def __init__(self, do_reverse=True, do_reverse_on_graph=True):
        self.name = 'YAGO310'
        train_path = os.path.join(self.path, 'train')
        valid_path = os.path.join(self.path, 'valid')
        test_path = os.path.join(self.path, 'test')
        super(YAGO310, self).__init__(train_path, valid_path, test_path,
                                      do_reverse=do_reverse, do_reverse_on_graph=do_reverse_on_graph)


class NELL995(Dataset):
    path = '../data/MWalk'

    query_relations = ['athleteplaysforteam',
                       'athleteplaysinleague',
                       'athletehomestadium',
                       'athleteplayssport',
                       'teamplayssport',
                       'organizationheadquarteredincity',
                       'worksfor',
                       'personborninlocation',
                       'personleadsorganization',
                       'organizationhiredperson',
                       'agentbelongstoorganization',
                       'teamplaysinleague']

    def __init__(self, query_relation=None):
        self.name = 'NELL995' if query_relation is None else query_relation
        if query_relation is None:
            self.path = os.path.join(NELL995.path, 'nell')
            test_candidates_path = None
        else:
            assert query_relation in NELL995.query_relations
            self.path = os.path.join(NELL995.path, query_relation)
            test_candidates_path = os.path.join(self.path, 'sort_test.pairs')
        train_path = os.path.join(self.path, 'train.txt')
        valid_path = os.path.join(self.path, 'dev.txt')
        test_path = os.path.join(self.path, 'test.txt')
        graph_path = os.path.join(self.path, 'graph.txt')

        super(NELL995, self).__init__(train_path, valid_path, test_path,
                                      graph_path=graph_path,
                                      test_candidates_path=test_candidates_path,
                                      has_reverse=True,
                                      get_reverse=lambda r: r + '_inv' if r[-4:] != '_inv' else r[:-4])

    @classmethod
    def datasets(cls, include_whole=False):
        for rel in cls.query_relations:
            yield cls(query_relation=rel)
        if include_whole:
            yield cls()

class F(keras.layers.Layer):
    def __init__(self, n_dims, n_layers, name=None):
        super(F, self).__init__(name=name)
        self.n_dims = n_dims
        self.n_layers = n_layers

    def build(self, input_shape):
        if self.n_layers == 1:
            self.dense_1 = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='dense_1')
        elif self.n_layers == 2:
            self.dense_1 = keras.layers.Dense(self.n_dims, activation=tf.nn.leaky_relu, name='dense_1')
            self.dense_2 = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='dense_2')
        else:
            raise ValueError('Invalid `n_layers`')

    def call(self, inputs, training=None):
        """ inputs[i]: bs x ... x n_dims
        """
        x = tf.concat(inputs, axis=-1)
        if self.n_layers == 1:
            return self.dense_1(x)
        elif self.n_layers == 2:
            return self.dense_2(self.dense_1(x))


class G(keras.layers.Layer):
    def __init__(self, n_dims, name=None):
        super(G, self).__init__(name=name)
        self.n_dims = n_dims

    def build(self, input_shape):
        self.left_dense = keras.layers.Dense(self.n_dims, activation=tf.nn.leaky_relu, name='left_dense')
        self.right_dense = keras.layers.Dense(self.n_dims, activation=tf.nn.leaky_relu, name='right_dense')
        self.center_dense = keras.layers.Dense(self.n_dims, activation=None, name='center_dense')

    def call(self, inputs, training=None):
        """ inputs: (left, right)
                left[i]: bs x ... x n_dims
                right[i]: bs x ... x n_dims
        """
        left, right = inputs
        left_x = tf.concat(left, axis=-1)
        right_x = tf.concat(right, axis=-1)
        return tf.reduce_sum(self.left_dense(left_x) * self.center_dense(self.right_dense(right_x)), axis=-1)


def update_op(inputs, update):
    out = inputs + update
    return out


def node2edge_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): batch_size x n_nodes x n_dims
        selected_edges: n_selected_edges x 6 (or 8) ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
    """
    hidden = inputs
    batch_size = tf.shape(inputs)[0]
    n_selected_edges = len(selected_edges)
    idx = tf.cond(tf.equal(batch_size, 1), lambda: tf.zeros((n_selected_edges,), dtype='int32'), lambda: selected_edges[:, 0])
    result = []
    if return_vi:
        idx_and_vi = tf.stack([idx, selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        hidden_vi = tf.gather_nd(hidden, idx_and_vi)  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        idx_and_vj = tf.stack([idx, selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        hidden_vj = tf.gather_nd(hidden, idx_and_vj)  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def node2edge_v2_op(inputs, selected_edges, return_vi=True, return_vj=True):
    """ inputs (hidden): n_selected_nodes x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    assert return_vi or return_vj
    hidden = inputs
    result = []
    if return_vi:
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        hidden_vi = tf.gather(hidden, new_idx_e2vi)  # n_selected_edges x n_dims
        result.append(hidden_vi)
    if return_vj:
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        hidden_vj = tf.gather(hidden, new_idx_e2vj)  # n_selected_edges x n_dims
        result.append(hidden_vj)
    return result


def aggregate_op(inputs, selected_edges, output_shape, at='vj', aggr_op_name='mean_v3'):
    """ inputs (edge_vec): n_seleted_edges x n_dims
        selected_edges: n_selected_edges x 6 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj), sorted by idx, vi, vj )
        output_shape: (batch_size=1, n_nodes, n_dims)
    """
    assert selected_edges is not None
    assert output_shape is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        aggr_op = tf.math.segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.segment_sum if aggr_op_name == 'sum' else \
            tf.math.segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vi+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vi+1) x n_dims
        idx_and_vi = tf.stack([selected_edges[:, 0], selected_edges[:, 1]], axis=1)  # n_selected_edges x 2
        idx_and_vi = tf.cast(tf.math.segment_max(idx_and_vi, idx_vi), tf.int32)  # (max_id_vi+1) x 2
        edge_vec_aggr = tf.scatter_nd(idx_and_vi, edge_vec_aggr, output_shape)  # batch_size x n_nodes x n_dims
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        max_idx_vj = tf.reduce_max(idx_vj)
        aggr_op = tf.math.unsorted_segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.unsorted_segment_sum if aggr_op_name == 'sum' else \
            tf.math.unsorted_segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vj+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vj+1) x n_dims
        idx_and_vj = tf.stack([selected_edges[:, 0], selected_edges[:, 2]], axis=1)  # n_selected_edges x 2
        idx_and_vj = tf.cast(tf.math.unsorted_segment_max(idx_and_vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1) x 2
        edge_vec_aggr = tf.scatter_nd(idx_and_vj, edge_vec_aggr, output_shape)  # batch_size x n_nodes x n_dims
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_aggr


def aggregate_v2_op(inputs, selected_edges, output_shape, at='vj', aggr_op_name='mean_v3'):
    """ inputs (edge_vec): n_seleted_edges x n_dims
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
        output_shape: (n_visited_nodes, n_dims)
    """
    assert selected_edges is not None
    assert output_shape is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        aggr_op = tf.math.segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.segment_sum if aggr_op_name == 'sum' else \
            tf.math.segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vi)  # (max_idx_vi+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vi+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.segment_sum(tf.ones((len(idx_vi), 1)), idx_vi)  # (max_idx_vi+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vi+1) x n_dims
        new_idx_e2vi = selected_edges[:, 6]  # n_selected_edges
        reduced_idx_e2vi = tf.cast(tf.math.segment_max(new_idx_e2vi, idx_vi), tf.int32)  # (max_id_vi+1)
        reduced_idx_e2vi = tf.expand_dims(reduced_idx_e2vi, 1)  # (max_id_vi+1) x 1
        edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vi, edge_vec_aggr, output_shape)  # n_visited_nodes x n_dims
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        max_idx_vj = tf.reduce_max(idx_vj)
        aggr_op = tf.math.unsorted_segment_mean if aggr_op_name == 'mean' or \
                                          aggr_op_name == 'mean_v2' or \
                                          aggr_op_name == 'mean_v3' else \
            tf.math.unsorted_segment_sum if aggr_op_name == 'sum' else \
            tf.math.unsorted_segment_max if aggr_op_name == 'max' else None
        edge_vec_aggr = aggr_op(edge_vec, idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x n_dims
        if aggr_op_name == 'mean_v2':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.log(tf.math.exp(1.) - 1 + edge_count)  # (max_idx_vj+1) x n_dims
        elif aggr_op_name == 'mean_v3':
            edge_count = tf.math.unsorted_segment_sum(tf.ones((len(idx_vj), 1)), idx_vj, max_idx_vj + 1)  # (max_idx_vj+1) x 1
            edge_vec_aggr = edge_vec_aggr * tf.math.sqrt(edge_count)  # (max_idx_vj+1) x n_dims
        new_idx_e2vj = selected_edges[:, 7]  # n_selected_edges
        reduced_idx_e2vj = tf.cast(tf.math.unsorted_segment_max(new_idx_e2vj, idx_vj, max_idx_vj + 1), tf.int32)  # (max_idx_vj+1)
        reduced_idx_e2vj = tf.expand_dims(reduced_idx_e2vj, 1)  # (max_idx_vj+1) x 1
        edge_vec_aggr = tf.scatter_nd(reduced_idx_e2vj, edge_vec_aggr, output_shape)  # n_visited_nodes x n_dims
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_aggr


def sparse_softmax_op(logits, segment_ids, sort=True):
    if sort:
        logits_max = tf.math.segment_max(logits, segment_ids)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_diff = logits - logits_max
        logits_exp = tf.math.exp(logits_diff)
        logits_expsum = tf.math.segment_sum(logits_exp, segment_ids)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    else:
        num_segments = tf.reduce_max(segment_ids) + 1
        logits_max = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
        logits_max = tf.gather(logits_max, segment_ids)
        logits_diff = logits - logits_max
        logits_exp = tf.math.exp(logits_diff)
        logits_expsum = tf.math.unsorted_segment_sum(logits_exp, segment_ids, num_segments)
        logits_expsum = tf.gather(logits_expsum, segment_ids)
        logits_norm = logits_exp / logits_expsum
    return logits_norm


def neighbor_softmax_op(inputs, selected_edges, at='vi'):
    """ inputs (edge_vec): n_seleted_edges x ...
        selected_edges: n_selected_edges x 8 ( int32, selected_edges[i] = (idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by idx, vi, vj )
    """
    assert selected_edges is not None
    edge_vec = inputs
    if at == 'vi':
        idx_vi = selected_edges[:, 4]  # n_selected_edges
        edge_vec_norm = sparse_softmax_op(edge_vec, idx_vi)  # n_selected_edges x ...
    elif at == 'vj':
        idx_vj = selected_edges[:, 5]  # n_selected_edges
        edge_vec_norm = sparse_softmax_op(edge_vec, idx_vj, sort=False)  # n_selected_edges x ...
    else:
        raise ValueError('Invalid `at`')
    return edge_vec_norm


class SharedEmbedding(keras.Model):
    def __init__(self, n_entities, n_relations, n_dims):
        super(SharedEmbedding, self).__init__(name='shared_emb')
        self.n_dims = n_dims
        self.entity_embedding = keras.layers.Embedding(n_entities, self.n_dims, name='entities')  # n_nodes x n_dims
        self.relation_embedding = keras.layers.Embedding(n_relations, self.n_dims, name='relations')  # n_rels x n_dims

    def call(self, inputs, target=None, training=None):
        assert target is not None
        if target == 'entity':
            return self.entity_embedding(inputs)
        elif target == 'relation':
            return self.relation_embedding(inputs)
        else:
            raise ValueError('Invalid `target`')

    def get_query_context(self, heads, rels):
        with tf.name_scope(self.name):
            head_emb = self.entity_embedding(heads)  # batch_size x n_dims
            rel_emb = self.relation_embedding(rels)  # batch_size x n_dims
        return head_emb, rel_emb


class UnconsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_dims, n_layers, aggr_op_name):
        super(UnconsciousnessFlow, self).__init__(name='uncon_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.aggr_op_name = aggr_op_name

        # fn(hidden_vi, rel_emb, hidden_vj)
        self.message_fn = F(self.n_dims, self.n_layers, name='message_fn')

        # fn(message_aggr, hidden, ent_emb)
        self.hidden_fn = F(self.n_dims, self.n_layers, name='hidden_fn')

    def call(self, inputs, selected_edges=None, shared_embedding=None, training=None, tc=None):
        """ inputs (hidden): 1 x n_nodes x n_dims
            selected_edges: n_selected_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)

            Here: batch_size = 1
        """
        assert selected_edges is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        # compute unconscious messages
        hidden = inputs
        hidden_vi, hidden_vj = node2edge_op(hidden, selected_edges)  # n_selected_edges x n_dims
        rel_idx = selected_edges[:, 3]  # n_selected_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_selected_edges x n_dims
        message = self.message_fn((hidden_vi, rel_emb, hidden_vj))  # n_selected_edges x n_dims

        # aggregate unconscious messages
        message_aggr = aggregate_op(message, selected_edges, (1, self.n_nodes, self.n_dims),
                                    aggr_op_name=self.aggr_op_name)  # 1 x n_nodes

        # update unconscious states
        ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
        ent_emb = shared_embedding(ent_idx, target='entity')  # 1 x n_nodes x n_dims
        update = self.hidden_fn((message_aggr, hidden, ent_emb))  # 1 x n_nodes x n_dims
        hidden = update_op(hidden, update)  # 1 x n_nodes x n_dims

        if tc is not None:
            tc['u.call'] += time.time() - t0
        return hidden  # 1 x n_nodes x n_dims

    def get_init_hidden(self, shared_embedding):
        with tf.name_scope(self.name):
            ent_idx = tf.expand_dims(tf.range(0, self.n_nodes), axis=0)  # 1 x n_nodes
            ent_emb = shared_embedding(ent_idx, target='entity')  # 1 x n_nodes x n_dims
            hidden = ent_emb
        return hidden


class ConsciousnessFlow(keras.Model):
    def __init__(self, n_entities, n_dims, n_layers, aggr_op_name):
        super(ConsciousnessFlow, self).__init__(name='con_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.aggr_op_name = aggr_op_name

        # fn(hidden_vi, rel_emb, hidden_vj, query_head_emb, query_rel_emb)
        self.message_fn = F(self.n_dims, self.n_layers, name='message_fn')

        # fn(message_aggr, hidden, hidden_uncon, query_head_emb, query_rel_emb)
        self.hidden_fn = F(self.n_dims, self.n_layers, name='hidden_fn')

        self.intervention_fn = keras.layers.Dense(self.n_dims, activation=None, use_bias=False, name='intervention_fn')

    def call(self, inputs, seen_edges=None, memorized_nodes=None, node_attention=None, hidden_uncon=None,
             query_head_emb=None, query_rel_emb=None, shared_embedding=None, training=None, tc=None):
        """ inputs (hidden): n_memorized_nodes x n_dims
            seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj), sorted by (idx, vi, vj)
                (1) including selfloop edges and backtrace edges
                (2) batch_size >= 1
            memorized_nodes: n_memorized_nodes x 2, (eg_idx, v)
            node_attention: batch_size x n_nodes
            hidden_uncon: 1 x n_nodes x n_dims
            query_head_emb: batch_size x n_dims
            query_rel_emb: batch_size x n_dims
        """
        assert seen_edges is not None
        assert node_attention is not None
        assert hidden_uncon is not None
        assert query_head_emb is not None
        assert query_rel_emb is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        hidden = inputs

        # compute conscious messages
        hidden_vi, hidden_vj = node2edge_v2_op(hidden, seen_edges)  # n_seen_edges x n_dims
        rel_idx = seen_edges[:, 3]  # n_seen_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_seen_edges x n_dims
        eg_idx = seen_edges[:, 0]  # n_seen_edges
        query_head_vec = tf.gather(query_head_emb, eg_idx)  # n_seen_edges x n_dims
        query_rel_vec = tf.gather(query_rel_emb, eg_idx)  # n_seen_edges x n_dims

        message = self.message_fn((hidden_vi, rel_emb, hidden_vj, query_head_vec, query_rel_vec))  # n_seen_edges x n_dims

        # aggregate conscious messages
        n_memorized_nodes = tf.shape(hidden)[0]
        message_aggr = aggregate_v2_op(message, seen_edges, (n_memorized_nodes, self.n_dims),
                                       aggr_op_name=self.aggr_op_name)  # n_memorized_nodes x n_dims

        # get unconscious states
        eg_idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
        hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims
        hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims
        query_head_vec = tf.gather(query_head_emb, eg_idx)  # n_memorized_nodes x n_dims
        query_rel_vec = tf.gather(query_rel_emb, eg_idx)  # n_memorized_nodes x n_dims

        # attend unconscious states
        idx_and_v = tf.stack([eg_idx, v], axis=1)  # n_memorized_nodes x 2
        node_attention = tf.gather_nd(node_attention, idx_and_v)  # n_memorized_nodes
        hidden_uncon = tf.expand_dims(node_attention, 1) * hidden_uncon  # n_memorized_nodes x n_dims
        hidden_uncon = self.intervention_fn(hidden_uncon)  # n_memorized_nodes x n_dims

        # update conscious state
        update = self.hidden_fn((message_aggr, hidden, hidden_uncon, query_head_vec, query_rel_vec))  # n_memorized_nodes x n_dims
        hidden = update_op(hidden, update)  # n_memorized_nodes x n_dims

        if tc is not None:
            tc['c.call'] += time.time() - t0
        return hidden  # n_memorized_nodes x n_dims

    def get_init_hidden(self, hidden_uncon, memorized_nodes):
        """ hidden_uncon: 1 x n_nodes x n_dims
            memorized_nodes: n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        """
        with tf.name_scope(self.name):
            idx, v = memorized_nodes[:, 0], memorized_nodes[:, 1]  # n_memorized_nodes, n_memorized_nodes
            hidden_uncon = tf.squeeze(hidden_uncon, axis=0)  # n_nodes x n_dims
            hidden_uncon = tf.gather(hidden_uncon, v)  # n_memorized_nodes x n_dims
            hidden_init = hidden_uncon
        return hidden_init  # n_memorized_nodes x n_dims


class AttentionFlow(keras.Model):
    def __init__(self, n_entities, n_dims_sm):
        super(AttentionFlow, self).__init__(name='att_flow')
        self.n_nodes = n_entities
        self.n_dims = n_dims_sm

        # fn((hidden_con_vi, rel_emb, query_head_emb, query_rel_emb),
        #    (hidden_con_vj, rel_emb, query_head_emb, query_rel_emb))
        self.transition_fn_1 = G(self.n_dims, name='transition_fn')

        # fn((hidden_con_vi, rel_emb, query_head_emb, query_rel_emb),
        #    (hidden_uncon_vj, rel_emb, query_head_emb, query_rel_emb))
        self.transition_fn_2 = G(self.n_dims, name='transition_fn')

        self.proj = keras.layers.Dense(self.n_dims, activation=tf.tanh, name='proj')

    def call(self, inputs, scanned_edges=None, hidden_uncon=None, hidden_con=None, shared_embedding=None,
             new_idx_for_memorized=None, n_memorized_and_scanned_nodes=None, query_head_emb=None, query_rel_emb=None,
             training=None, tc=None):
        """ inputs (node_attention): batch_size x n_nodes
            scanned_edges (aug_scanned_edges): n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
              (1) including selfloop edges
              (2) batch_size >= 1
            hidden_uncon: 1 x n_nodes x n_dims
            hidden_con: n_memorized_nodes x n_dims
            query_head_emb: batch_size x n_dims
            query_rel_emb: batch_size x n_dims
        """
        assert scanned_edges is not None
        assert hidden_con is not None
        assert hidden_uncon is not None
        assert query_head_emb is not None
        assert query_rel_emb is not None
        assert n_memorized_and_scanned_nodes is not None
        assert shared_embedding is not None
        if tc is not None:
            t0 = time.time()

        hidden_con = self.proj(hidden_con)  # n_memorized_nodes x n_dims_sm
        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_scanned_nodes, self.n_dims)))  # n_memorized_and_scanned_nodes x n_dims_sm
        hidden_uncon = self.proj(hidden_uncon)  # 1 x n_nodes x n_dims_sm
        query_head_vec = self.proj(query_head_emb)  # batch_size x n_dims_sm
        query_rel_vec = self.proj(query_rel_emb)  # batch_size x n_dims_sm

        # compute transition
        hidden_con_vi, hidden_con_vj = node2edge_v2_op(hidden_con, scanned_edges)  # n_aug_scanned_edges x n_dims_sm
        hidden_uncon_vj, = node2edge_op(hidden_uncon, scanned_edges, return_vi=False)  # n_aug_scanned_edges x n_dims_sm

        rel_idx = scanned_edges[:, 3]  # n_aug_scanned_edges
        rel_emb = shared_embedding(rel_idx, target='relation')  # n_aug_scanned_edges x n_dims
        rel_emb = self.proj(rel_emb)  # n_aug_scanned_edges x n_dims_sm

        eg_idx = scanned_edges[:, 0]  # n_aug_scanned_edges
        q_head_vec = tf.gather(query_head_vec, eg_idx)  # n_seen_edges x n_dims
        q_rel_vec = tf.gather(query_rel_vec, eg_idx)  # n_seen_edges x n_dims

        transition_logits = self.transition_fn_1(((hidden_con_vi, rel_emb, q_head_vec, q_rel_vec),
                                                  (hidden_con_vj, rel_emb, q_head_vec, q_rel_vec)))
        transition_logits += self.transition_fn_2(((hidden_con_vi, rel_emb, q_head_vec, q_rel_vec),
                                                   (hidden_uncon_vj, rel_emb, q_head_vec, q_rel_vec)))  # n_aug_scanned_edges
        transition = neighbor_softmax_op(transition_logits, scanned_edges)  # n_aug_scanned_edges

        # compute transition attention
        node_attention = inputs  # batch_size x n_nodes
        idx_and_vi = tf.stack([scanned_edges[:, 0], scanned_edges[:, 1]], axis=1)  # n_aug_scanned_edges x 2
        gathered_node_attention = tf.gather_nd(node_attention, idx_and_vi)  # n_aug_scanned_edges
        trans_attention = gathered_node_attention * transition  # n_aug_scanned_edges

        # compute new node attention
        batch_size = tf.shape(inputs)[0]
        new_node_attention = aggregate_op(trans_attention, scanned_edges, (batch_size, self.n_nodes),
                                          aggr_op_name='sum')  # batch_size x n_nodes

        new_node_attention_sum = tf.reduce_sum(new_node_attention, axis=1, keepdims=True)  # batch_size x 1
        new_node_attention = new_node_attention / new_node_attention_sum  # batch_size x n_nodes

        if tc is not None:
            tc['a.call'] += time.time() - t0
        # new_node_attention: batch_size x n_nodes
        return new_node_attention

    def get_init_node_attention(self, heads):
        with tf.name_scope(self.name):
            node_attention = tf.one_hot(heads, self.n_nodes)  # batch_size x n_nodes
        return node_attention


class Model(object):
    def __init__(self, graph, hparams):
        self.graph = graph
        self.hparams = hparams

        self.shared_embedding = SharedEmbedding(graph.n_entities, graph.n_relations, hparams.n_dims)

        self.uncon_flow = UnconsciousnessFlow(graph.n_entities, hparams.n_dims, hparams.n_layers, hparams.aggregate_op)

        self.con_flow = ConsciousnessFlow(graph.n_entities, hparams.n_dims, hparams.n_layers, hparams.aggregate_op)

        self.att_flow = AttentionFlow(graph.n_entities, hparams.n_dims_sm)

        self.heads, self.rels = None, None

        # for visualization
        self.attended_nodes = None

        # for analysis
        self.entropy_along_steps = None
        self.top1_occupy_along_steps = None
        self.top3_occupy_along_steps = None
        self.top5_occupy_along_steps = None

    def set_init(self, heads, rels, tails, batch_i, epoch):
        """ heads: batch_size
            rels: batch_size
        """
        self.heads = heads
        self.rels = rels
        self.tails = tails
        self.batch_i = batch_i
        self.epoch = epoch

    def initialize(self, training=True, output_attention=False, analyze_attention=False, tc=None):
        query_head_emb, query_rel_emb = self.shared_embedding.get_query_context(self.heads, self.rels)  # batch_size x n_dims

        ''' initialize unconsciousness flow'''
        hidden_uncon = self.uncon_flow.get_init_hidden(self.shared_embedding)  # 1 x n_nodes x n_dims

        ''' run unconsciousness flow before running consciousness flow '''
        if self.hparams.uncon_steps is not None:
            for _ in range(self.hparams.uncon_steps):
                # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
                candidate_edges = self.graph.get_candidate_edges(tc=get(tc, 'graph'))

                # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
                max_edges_per_eg = self.hparams.max_edges_per_example if training else self.hparams.test_max_edges_per_example
                sampled_edges = self.graph.get_sampled_edges(candidate_edges, mode='by_eg',
                                                             max_edges_per_eg=max_edges_per_eg,
                                                             tc=get(tc, 'graph'))

                # selected_edges: (np.array) n_selected_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj] sorted by (eg_idx, vi, vj)
                selected_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

                hidden_uncon = self.uncon_flow(hidden_uncon, selected_edges=selected_edges,
                                               shared_embedding=self.shared_embedding,
                                               tc=get(tc, 'model'))  # 1 x n_nodes x n_dims

        ''' initialize attention flow '''
        node_attention = self.att_flow.get_init_node_attention(self.heads)  # batch_size x n_nodes

        ''' initialize consciousness flow '''
        memorized_v = self.graph.set_init_memorized_nodes(self.heads)  # n_memorized_nodes (=batch_size) x 2, (eg_idx, v)
        hidden_con = self.con_flow.get_init_hidden(hidden_uncon, memorized_v)  # n_memorized_nodes x n_dims

        if output_attention and not training:
            self.attended_nodes = []

        if analyze_attention and not training:
            self.entropy_along_steps = [tf.reduce_mean(entropy(node_attention))]
            self.top1_occupy_along_steps = [tf.reduce_mean(topk_occupy(node_attention, 1))]
            self.top3_occupy_along_steps = [tf.reduce_mean(topk_occupy(node_attention, 3))]
            self.top5_occupy_along_steps = [tf.reduce_mean(topk_occupy(node_attention, 5))]

        # hidden_uncon: 1 x n_nodes x n_dims
        # hidden_con: n_memorized_nodes x n_dims
        # node_attention: batch_size x n_nodes
        # query_head_emb, query_rel_emb: batch_size x n_dims
        return hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb

    def flow(self, hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb,
             training=True, output_attention=False, analyze_attention=False, tc=None):
        """ hidden_uncon: 1 x n_nodes x n_dims
            hidden_con: n_memorized_nodes x n_dims
            node_attention: batch_size x n_nodes
        """
        ''' get scanned edges '''
        # attended_nodes: (np.array) n_attended_nodes x 2, (eg_idx, vi) sorted
        max_attended_nodes = self.hparams.max_attended_nodes if training else self.hparams.test_max_attended_nodes
        attended_nodes = self.graph.get_topk_nodes(node_attention, max_attended_nodes, tc=get(tc, 'graph'))  # n_attended_nodes x 2

        if output_attention and not training:
            self.attended_nodes.append((attended_nodes, tf.gather_nd(node_attention, attended_nodes)))

        # candidate_edges: (np.array) n_candidate_edges x 5, (eg_idx, edge_id, vi, vj, rel) sorted by (eg_idx, edge_id)
        candidate_edges = self.graph.get_candidate_edges(attended_nodes=attended_nodes,
                                                         tc=get(tc, 'graph'))  # n_candidate_edges x 2

        # sampled_edges: (np.array) n_sampled_edges x 6, (eg_idx, edge_id, vi, vj, rel, ca_idx) sorted by (eg_idx, edge_id)
        max_edges_per_vi = self.hparams.max_edges_per_node if training else self.hparams.test_max_edges_per_node
        sampled_edges = self.graph.get_sampled_edges(candidate_edges,
                                                     mode='by_vi',
                                                     max_edges_per_vi=max_edges_per_vi,
                                                     tc=get(tc, 'graph'))

        # scanned_edges: (np.array) n_scanned_edges (=n_sampled_edges) x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        scanned_edges = self.graph.get_selected_edges(sampled_edges, tc=get(tc, 'graph'))

        ''' add selfloop edges '''
        selfloop_edges = self.graph.get_selfloop_edges(attended_nodes, tc=get(tc, 'graph'))

        # aug_scanned_edges: n_aug_scanned_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        aug_scanned_edges = self.graph.get_union_edges(scanned_edges, selfloop_edges, tc=get(tc, 'graph'))

        ''' run attention flow (over memorized and scanned nodes) '''
        # new_idx_for_memorized: n_memorized_nodes x 1 or None
        # memorized_and_scanned: n_memorized_and_scanned_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        new_idx_for_memorized, n_memorized_and_scanned_nodes, memorized_and_scanned = \
            self.graph.add_nodes_to_memorized(scanned_edges, tc=get(tc, 'graph'))

        # aug_scanned_edges: n_aug_scanned_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        aug_scanned_edges = self.graph.set_index_over_nodes(aug_scanned_edges, memorized_and_scanned, tc=get(tc, 'graph'))

        new_node_attention = self.att_flow(node_attention,
                                           scanned_edges=aug_scanned_edges,
                                           hidden_uncon=hidden_uncon,
                                           hidden_con=hidden_con,
                                           shared_embedding=self.shared_embedding,
                                           new_idx_for_memorized=new_idx_for_memorized,
                                           n_memorized_and_scanned_nodes=n_memorized_and_scanned_nodes,
                                           query_head_emb=query_head_emb,
                                           query_rel_emb=query_rel_emb,
                                           tc=get(tc, 'model'))  # n_aug_scanned_edges, batch_size x n_nodes

        ''' get seen edges '''
        # seen_nodes: (np.array) n_seen_nodes x 2, (eg_idx, vj) unique and sorted
        max_seen_nodes = self.hparams.max_seen_nodes if training else self.hparams.test_max_seen_nodes
        seen_nodes = self.graph.get_topk_nodes(new_node_attention, max_seen_nodes, tc=get(tc, 'graph'))  # n_seen_nodes x 2

        # seen_edges: (np.array) n_seen_edges x 6, (eg_idx, vi, vj, rel, idx_vi, idx_vj) sorted by (eg_idx, vi, vj)
        seen_edges = self.graph.get_seen_edges(seen_nodes, aug_scanned_edges, tc=get(tc, 'graph'))

        ''' run consciousness flow (over memorized and seen nodes) '''
        # new_idx_for_memorized: n_memorized_nodes x 1 or None
        # memorized_and_seen: _memorized_and_seen_nodes x 2, (eg_idx, v) sorted by (eg_idx, v)
        new_idx_for_memorized, n_memorized_and_seen_nodes, memorized_and_seen = \
            self.graph.add_nodes_to_memorized(seen_edges, inplace=True, tc=get(tc, 'graph'))

        if new_idx_for_memorized is not None:
            hidden_con = tf.scatter_nd(new_idx_for_memorized, hidden_con,
                                       tf.TensorShape((n_memorized_and_seen_nodes, self.hparams.n_dims)))  # n_memorized_nodes (new) x n_dims

        # seen_edges: n_seen_edges x 8, (eg_idx, vi, vj, rel, idx_vi, idx_vj, new_idx_e2vi, new_idx_e2vj) sorted by (eg_idx, vi, vj)
        seen_edges = self.graph.set_index_over_nodes(seen_edges, memorized_and_seen, tc=get(tc, 'graph'))

        new_hidden_con = self.con_flow(hidden_con,
                                       seen_edges=seen_edges,
                                       memorized_nodes=self.graph.memorized_nodes,
                                       node_attention=new_node_attention,
                                       hidden_uncon=hidden_uncon,
                                       query_head_emb=query_head_emb,
                                       query_rel_emb=query_rel_emb,
                                       shared_embedding=self.shared_embedding,
                                       tc=get(tc, 'model'))  # n_memorized_nodes (new) x n_dims

        if analyze_attention and not training:
            self.entropy_along_steps.append(tf.reduce_mean(entropy(new_node_attention)))
            self.top1_occupy_along_steps.append(tf.reduce_mean(topk_occupy(new_node_attention, 1)))
            self.top3_occupy_along_steps.append(tf.reduce_mean(topk_occupy(new_node_attention, 3)))
            self.top5_occupy_along_steps.append(tf.reduce_mean(topk_occupy(new_node_attention, 5)))

        # hidden_uncon: 1 x n_nodes x n_dims,
        # new_hidden_con: n_memorized_nodes x n_dims,
        # new_node_attention: batch_size x n_nodes
        return hidden_uncon, new_hidden_con, new_node_attention

    def save_attention_to_file(self, final_node_attention, id2entity, id2relation, epoch, dir_name, training=True):
        max_attended_nodes = self.hparams.max_attended_nodes if training else self.hparams.test_max_attended_nodes
        attended_nodes = self.graph.get_topk_nodes(final_node_attention, max_attended_nodes)  # n_attended_nodes x 2
        self.attended_nodes.append((attended_nodes, tf.gather_nd(final_node_attention, attended_nodes)))

        batch_size = len(self.heads)
        for batch_i in range(batch_size):
            head, rel, tail = self.heads[batch_i], self.rels[batch_i], self.tails[batch_i]
            filename = '{:d}->{:d}->{:d}.txt'.format(head, rel, tail)
            filename = 'train_epoch{:d}_'.format(epoch) + filename if training \
                else 'test_epoch{:d}_'.format(epoch) + filename

            with open(os.path.join(dir_name, filename), 'w') as fout:
                fout.write('nodes:\n')
                nodes = []
                for att_nodes, att_scores in self.attended_nodes:
                    mask = (att_nodes[:, 0] == batch_i)
                    att_nds = att_nodes[:, 1][mask]
                    att_scs = att_scores[mask]
                    nd_idx = np.argsort(-att_scs)
                    node_line = '\t'.join(['{:d}({}):{:f}'.format(att_nds[nd_i],
                                                                  id2entity[att_nds[nd_i]],
                                                                  att_scs[nd_i])
                                           for nd_i in nd_idx])
                    fout.write(node_line + '\n')
                    nodes.append(att_nds[nd_idx])

                fout.write('edges:\n')
                for i, nds_vi in enumerate(nodes[:-1]):
                    nds_vj = nodes[i+1]
                    edges = self.graph.get_vivj_edges(nds_vi, nds_vj, with_eg_idx=False)  # n_vivj_edges x 4, (edge_id, vi, vj, rel) sorted by (vi, vj)
                    edge_line = '\t'.join(['{:d}({})->{:d}({})->{:d}({})'.format(vi, id2entity[vi],
                                                                                 rel, id2relation[rel],
                                                                                 vj, id2entity[vj])
                                           for _, vi, vj, rel in edges])
                    fout.write(edge_line + '\n')

    @property
    def trainable_variables(self):
        return self.shared_embedding.trainable_variables + \
               self.uncon_flow.trainable_variables + \
               self.con_flow.trainable_variables + \
               self.att_flow.trainable_variables
# 获取所有 GPU 设备
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # 选择第一个 GPU 设备并启用内存增长
    tf.config.experimental.set_memory_growth(gpus[0], True)

from dataenv import DataEnv
from model import Model
import config
import datasets


def loss_fn(prediction, tails):
    """ predictions: (tf.Tensor) batch_size x n_nodes
        tails: (np.array) batch_size
    """
    pred_idx = tf.stack([tf.range(0, len(tails)), tails], axis=1)  # batch_size x 2
    pred_prob = tf.gather_nd(prediction, pred_idx)  # batch_size
    pred_loss = tf.reduce_mean(- tf.math.log(pred_prob + 1e-20))
    return pred_loss


def calc_metric(heads, relations, prediction, targets, filter_pool):
    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = 0., 0., 0., 0., 0., 0., 0.

    n_preds = prediction.shape[0]
    for i in range(n_preds):
        head = heads[i]
        rel = relations[i]
        tar = targets[i]
        pred = prediction[i]
        fil = list(filter_pool[(head, rel)] - {tar})

        sorted_idx = np.argsort(-pred)
        mask = np.logical_not(np.isin(sorted_idx, fil))
        sorted_idx = sorted_idx[mask]

        rank = np.where(sorted_idx == tar)[0].item() + 1

        if rank <= 1:
            hit_1 += 1
        if rank <= 3:
            hit_3 += 1
        if rank <= 5:
            hit_5 += 1
        if rank <= 10:
            hit_10 += 1
        mr += rank
        mrr += 1. / rank
        max_r = max(max_r, rank)

    hit_1 /= n_preds
    hit_3 /= n_preds
    hit_5 /= n_preds
    hit_10 /= n_preds
    mr /= n_preds
    mrr /= n_preds

    return hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r


def calc_metric_v2(heads, relations, prediction, candidates):
    aps = []
    n_preds = prediction.shape[0]
    for i in range(n_preds):
        head = heads[i]
        pred = prediction[i]
        tail_dct = candidates[head]

        score_ans = [(pred[tail], ans) for tail, ans in tail_dct.items()]
        score_ans.sort(key=lambda x: x[0], reverse=True)

        ranks = []
        correct = 0
        for idx, item in enumerate(score_ans):
            if item[1] == '+':
                correct += 1
                ranks.append(correct / (1. + idx))
        if len(ranks) == 0:
            ranks.append(0)
        aps.append(np.mean(ranks))
    mean_ap = np.mean(aps)
    n_queries = len(aps)
    return mean_ap, n_queries


class Trainer(object):
    def __init__(self, model, data_env, hparams):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams

        if hparams.clipnorm is None:
            self.optimizer = keras.optimizers.Adam(learning_rate=self.get_lr())
        else:
            self.optimizer = keras.optimizers.Adam(learning_rate=self.get_lr(), clipnorm=hparams.clipnorm)

        self.train_loss = None
        self.train_accuracy = None

    def get_lr(self, epoch=None):
        if isinstance(hparams.learning_rate, float):
            return hparams.learning_rate
        elif isinstance(hparams.learning_rate, (tuple, list)):
            if epoch is None:
                return hparams.learning_rate[-1]
            else:
                return hparams.learning_rate[epoch-1]
        else:
            raise ValueError('Invalid `learning_rate`')

    def train_step(self, heads, tails, rels, epoch, batch_i, tc=None):
        self.optimizer.learning_rate = self.get_lr(epoch)
        n_splits = 1
        heads_li, tails_li, rels_li = [heads], [tails], [rels]
        while True:
            try:
                train_loss = 0.
                accuracy = 0.
                prediction_li = []
                for k in range(n_splits):
                    heads,tails, rels = heads_li[k], tails_li[k], rels_li[k]
                    self.model.set_init(heads, rels, tails, batch_i, epoch)

                    with tf.GradientTape() as tape:
                        hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
                            self.model.initialize(tc=tc)

                        for step in range(1, self.hparams.con_steps + 1):
                            hidden_uncon, hidden_con, node_attention = \
                                self.model.flow(hidden_uncon, hidden_con, node_attention,
                                                query_head_emb, query_rel_emb, tc=tc)
                        prediction = node_attention

                        #pred_tails = tf.argmax(node_attention, axis=1)
                        #self.model.graph.add_inputs(np.stack([heads, rels, tails, pred_tails], axis=1), epoch, batch_i)

                        loss = loss_fn(prediction, tails)

                    if tc is not None:
                        t0 = time.time()
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    if tc is not None:
                        tc['grad']['comp'] += time.time() - t0

                    if tc is not None:
                        t0 = time.time()
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    if tc is not None:
                        tc['grad']['apply'] += time.time() - t0

                    train_loss += loss
                    accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tails), tf.float32))
                    prediction_li.append(prediction.numpy())

                train_loss /= n_splits
                accuracy /= n_splits

                decay = self.hparams.moving_mean_decay
                self.train_loss = train_loss if self.train_loss is None else self.train_loss * decay + train_loss * (1 - decay)
                self.train_accuracy = accuracy if self.train_accuracy is None else self.train_accuracy * decay + accuracy * (1 - decay)

                train_metric = None
                if self.hparams.print_train_metric:
                    train_metric = calc_metric(np.concatenate(heads_li, axis=0),
                                               np.concatenate(rels_li, axis=0),
                                               np.concatenate(prediction_li, axis=0),
                                               np.concatenate(tails_li, axis=0),
                                               self.data_env.filter_pool)

                return train_loss.numpy(), accuracy.numpy(), self.train_loss.numpy(), self.train_accuracy.numpy(), train_metric

            except tf.errors.ResourceExhaustedError:
                print('Meet `tf.errors.ResourceExhaustedError`')
                n_splits += 1
                print('split into %d batches' % n_splits)
                heads_li = np.array_split(heads, n_splits, axis=0)
                tails_li = np.array_split(tails, n_splits, axis=0)
                rels_li = np.array_split(rels, n_splits, axis=0)


class Evaluator(object):
    def __init__(self, model, data_env, hparams, mode='test', rel=None):
        self.model = model
        self.data_env = data_env
        self.hparams = hparams

        if mode == 'test':
            self.test_candidates = data_env.test_candidates if rel is None else data_env.test_candidates[rel]
        else:
            self.test_candidates = None

        self.heads = []
        self.relations = []
        self.prediction = []
        self.targets = []

        self.eval_loss = None
        self.eval_accuracy = None

        if hparams.test_analyze_attention:
            self.entropy_along_steps = [0.] * hparams.con_steps
            self.top1_occupy_along_steps = [0.] * hparams.con_steps
            self.top3_occupy_along_steps = [0.] * hparams.con_steps
            self.top5_occupy_along_steps = [0.] * hparams.con_steps
            self.count = 0

    def eval_step(self, heads, tails, rels, epoch, batch_i,
                  disable_output_attention=True, disable_analyze_attention=True):
        n_splits = 1
        heads_li, tails_li, rels_li = [heads], [tails], [rels]
        while True:
            try:
                eval_loss = 0.
                accuracy = 0.
                prediction_li = []
                for k in range(n_splits):
                    heads, tails, rels = heads_li[k], tails_li[k], rels_li[k]
                    self.model.set_init(heads, rels, tails, batch_i, epoch)

                    hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb = \
                        self.model.initialize(training=False,
                                              output_attention=not disable_output_attention,
                                              analyze_attention=not disable_analyze_attention)

                    for step in range(1, self.hparams.con_steps + 1):
                        hidden_uncon, hidden_con, node_attention = \
                            self.model.flow(hidden_uncon, hidden_con, node_attention, query_head_emb, query_rel_emb,
                                            training=False,
                                            output_attention=not disable_output_attention,
                                            analyze_attention=not disable_analyze_attention)
                    prediction = node_attention  # batch_size x n_nodes

                    self.heads.append(heads)
                    self.relations.append(rels)
                    self.prediction.append(prediction.numpy())
                    self.targets.append(tails)

                    loss = loss_fn(prediction, tails)

                    eval_loss += loss
                    accuracy += tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, axis=1), tails), tf.float32))
                    prediction_li.append(prediction.numpy())

                    if not disable_output_attention:
                        self.model.save_attention_to_file(node_attention,
                                                          self.data_env.ds.id2entity,
                                                          self.data_env.ds.id2relation,
                                                          epoch, self.hparams.dir_name,
                                                          training=False)

                    if not disable_analyze_attention:
                        self.entropy_along_steps = [a + b for a, b in zip(self.entropy_along_steps,
                                                                          self.model.entropy_along_steps)]
                        self.top1_occupy_along_steps = [a + b for a, b in zip(self.top1_occupy_along_steps,
                                                                          self.model.top1_occupy_along_steps)]
                        self.top3_occupy_along_steps = [a + b for a, b in zip(self.top3_occupy_along_steps,
                                                                              self.model.top3_occupy_along_steps)]
                        self.top5_occupy_along_steps = [a + b for a, b in zip(self.top5_occupy_along_steps,
                                                                              self.model.top5_occupy_along_steps)]
                        self.count += 1

                eval_loss /= n_splits
                accuracy /= n_splits

                decay = self.hparams.moving_mean_decay
                self.eval_loss = eval_loss if self.eval_loss is None else self.eval_loss * decay + eval_loss * (1 - decay)
                self.eval_accuracy = accuracy if self.eval_accuracy is None else self.eval_accuracy * decay + accuracy * (1 - decay)

                return eval_loss.numpy(), accuracy.numpy(), self.eval_loss.numpy(), self.eval_accuracy.numpy()

            except (tf.errors.InternalError, tf.errors.ResourceExhaustedError, SystemError):
                print('Meet `tf.errors.InternalError` or `tf.errors.ResourceExhaustedError` or `SystemError`')
                n_splits += 1
                print('split into %d batches' % n_splits)
                heads_li = np.array_split(heads, n_splits, axis=0)
                tails_li = np.array_split(tails, n_splits, axis=0)
                rels_li = np.array_split(rels, n_splits, axis=0)

    def reset_metric(self):
        self.heads = []
        self.relations = []
        self.prediction = []
        self.targets = []

        self.eval_loss = None
        self.eval_accuracy = None

    def metric_result(self):
        heads = np.concatenate(self.heads, axis=0)
        relations = np.concatenate(self.relations, axis=0)
        prediction = np.concatenate(self.prediction, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        return calc_metric(heads, relations, prediction, targets, self.data_env.filter_pool)

    def metric_result_v2(self):
        if self.test_candidates is None:
            return None
        else:
            heads = np.concatenate(self.heads, axis=0)
            relations = np.concatenate(self.relations, axis=0)
            prediction = np.concatenate(self.prediction, axis=0)
            return calc_metric_v2(heads, relations, prediction, self.test_candidates)

    def metric_for_analysis(self):
        if self.hparams.test_analyze_attention:
            entropy_along_steps = [a / self.count for a in self.entropy_along_steps]
            top1_occupy_along_steps = [a / self.count for a in self.top1_occupy_along_steps]
            top3_occupy_along_steps = [a / self.count for a in self.top3_occupy_along_steps]
            top5_occupy_along_steps = [a / self.count for a in self.top5_occupy_along_steps]
            return entropy_along_steps, top1_occupy_along_steps, top3_occupy_along_steps, top5_occupy_along_steps
        else:
            return None

def reset_time_cost(hparams):
    if hparams.timer:
        return {'model': defaultdict(float), 'graph': defaultdict(float), 'grad': defaultdict(float)}
    else:
        return None


def str_time_cost(tc):
    if tc is not None:
        model_tc = ', '.join('m.{} {:3f}'.format(k, v) for k, v in tc['model'].items())
        graph_tc = ', '.join('g.{} {:3f}'.format(k, v) for k, v in tc['graph'].items())
        grad_tc = ', '.join('d.{} {:3f}'.format(k, v) for k, v in tc['grad'].items())
        return model_tc + ', ' + graph_tc + ', ' + grad_tc
    else:
        return ''


def run_eval(data_env, model, hparams, epoch, batch_i,
             enable_test_bs=False, disable_output_attention=True, disable_analyze_attention=True):
    if hparams.eval_valid:
        valid_evaluator = Evaluator(model, data_env, hparams, mode='valid')
        valid_evaluator.reset_metric()
        valid_batcher = data_env.get_valid_batcher()
        b_i = 1
        n_b = data_env.n_valid / hparams.test_batch_size
        p = 10
        batch_size = hparams.test_batch_size if enable_test_bs else hparams.batch_size
        for valid_batch, bs in valid_batcher(batch_size):
            heads, tails, rels = valid_batch[:, 0], valid_batch[:, 1], valid_batch[:, 2]
            valid_evaluator.eval_step(heads, tails, rels, epoch, b_i,
                                      disable_output_attention=disable_output_attention,
                                      disable_analyze_attention=disable_analyze_attention)
            if b_i > n_b * p / 100:
                print('[VALID] {:d}%'.format(p))
                p += 10
            b_i += 1

        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = valid_evaluator.metric_result()
        print('[REPORT VALID] {:d}, {:d} | loss: {:.4f} | acc: {:.4f} | '
              'hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | '
              'mr: {:.1f} | mmr: {:.6f} | max_r: {:.1f} * * * * *'.format(
            epoch, batch_i, valid_evaluator.eval_loss, valid_evaluator.eval_accuracy,
            hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))

        if not disable_analyze_attention:
            entropy, top1_occupy, top3_occupy, top5_occupy = valid_evaluator.metric_for_analysis()
            print('[ANALYSIS VALID] {:d}, {:d} | entropy: {:.6f} | top1_occupy: {:.6f} | '
                  'top3_occupy: {:.6f} | top5_occupy: {:.6f} * * * * *'.format(
                epoch, batch_i, entropy, top1_occupy, top3_occupy, top5_occupy))

    if data_env.test_by_rel is not None:
        for rel in data_env.get_test_relations():
            test_evaluator = Evaluator(model, data_env, hparams, mode='test', rel=rel)
            test_evaluator.reset_metric()
            test_batcher = data_env.get_test_batcher_by_rel(rel)

            b_i = 1
            n_b = data_env.n_test / hparams.test_batch_size
            p = 10
            batch_size = hparams.test_batch_size if enable_test_bs else hparams.batch_size
            for test_batch, bs in test_batcher(batch_size):
                heads, tails, rels = test_batch[:, 0], test_batch[:, 1], test_batch[:, 2]
                test_evaluator.eval_step(heads, tails, rels, epoch, b_i,
                                         disable_output_attention=disable_output_attention,
                                         disable_analyze_attention=disable_analyze_attention)
                if b_i > n_b * p / 100:
                    print('[TEST] {:d}%'.format(p))
                    p += 10
                b_i += 1

            hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = test_evaluator.metric_result()
            mean_ap, n_queries = test_evaluator.metric_result_v2()
            print('[REPORT TEST] {} {:d}, {:d} | loss: {:.4f} | acc: {:.4f} | '
                  'hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | '
                  'mr: {:.1f} | mmr: {:.6f} | max_r: {:.1f} | map: {:.6f} | n_queries: {:.1f} * * * * *'.format(
                rel, epoch, batch_i, test_evaluator.eval_loss, test_evaluator.eval_accuracy,
                hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r, mean_ap, n_queries))
    else:
        test_evaluator = Evaluator(model, data_env, hparams, mode='test')
        test_evaluator.reset_metric()
        test_batcher = data_env.get_test_batcher()

        b_i = 1
        n_b = data_env.n_test / hparams.test_batch_size
        p = 10
        batch_size = hparams.test_batch_size if enable_test_bs else hparams.batch_size
        for test_batch, bs in test_batcher(batch_size):
            heads, tails, rels = test_batch[:, 0], test_batch[:, 1], test_batch[:, 2]
            test_evaluator.eval_step(heads, tails, rels, epoch, b_i,
                                     disable_output_attention=disable_output_attention,
                                     disable_analyze_attention=disable_analyze_attention)
            if b_i > n_b * p / 100:
                print('[TEST] {:d}%'.format(p))
                p += 10
            b_i += 1

        hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = test_evaluator.metric_result()
        if test_evaluator.test_candidates is None:
            print('[REPORT TEST] {:d}, {:d} | loss: {:.4f} | acc: {:.4f} | '
                  'hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | '
                  'mr: {:.1f} | mmr: {:.6f} | max_r: {:.1f} * * * * *'.format(
                epoch, batch_i, test_evaluator.eval_loss, test_evaluator.eval_accuracy,
                hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
        else:
            mean_ap, n_queries = test_evaluator.metric_result_v2()
            print('[REPORT TEST] {:d}, {:d} | loss: {:.4f} | acc: {:.4f} | '
                  'hit_1: {:.6f} | hit_3: {:.6f} | hit_5: {:.6f} | hit_10: {:.6f} | '
                  'mr: {:.1f} | mmr: {:.6f} | max_r: {:.1f} | map: {:.6f} | n_queries: {:.1f} * * * * *'.format(
                epoch, batch_i, test_evaluator.eval_loss, test_evaluator.eval_accuracy,
                hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r, mean_ap, n_queries))

        if not disable_analyze_attention:
            entropy, top1_occupy, top3_occupy, top5_occupy = test_evaluator.metric_for_analysis()
            print('[ANALYSIS TEST] {:d}, {:d} | entropy: {} | top1_occupy: {} | top3_occupy: {} | top5_occupy: {} * * * * *'.format(
                epoch, batch_i,
                ', '.join(['{:.6f}'.format(e) for e in entropy]),
                ', '.join(['{:.6f}'.format(o) for o in top1_occupy]),
                ', '.join(['{:.6f}'.format(o) for o in top3_occupy]),
                ', '.join(['{:.6f}'.format(o) for o in top5_occupy])))


def run(dataset, hparams):
    data_env = DataEnv(dataset)
    model = Model(data_env.graph, hparams)

    trainer = Trainer(model, data_env, hparams)
    train_batcher = data_env.get_train_batcher(remove_all_head_tail_edges=hparams.remove_all_head_tail_edges)

    t0_tr = time.time()
    n_batches = int(np.ceil(data_env.n_train / hparams.batch_size))
    for epoch in range(1, hparams.max_epochs + 1):
        batch_i = 1

        for train_batch, batch_size in train_batcher(hparams.batch_size):
            time_cost = reset_time_cost(hparams)

            heads, tails, rels = train_batch[:, 0], train_batch[:, 1], train_batch[:, 2]
            cur_train_loss, cur_accuracy, train_loss, accuracy, train_metric = trainer.train_step(
                heads, tails, rels, epoch, batch_i, tc=time_cost)

            t1_tr = time.time()
            dt_tr = t1_tr - t0_tr
            t0_tr = t1_tr

            if hparams.print_train and batch_i % hparams.print_train_freq == 0:
                if hparams.print_train_metric:
                    hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r = train_metric
                    print('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {} | '
                          'hit_1: {:.4f} | hit_3: {:.4f} | hit_5: {:.4f} | hit_10: {:.4f} | '
                          'mr: {:.1f} | mmr: {:.4f} | max_r: {:.1f}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy,
                        dt_tr, str_time_cost(time_cost), hit_1, hit_3, hit_5, hit_10, mr, mrr, max_r))
                else:
                    print('[TRAIN] {:d}, {:d} | loss: {:.4f} ({:.4f}) | acc: {:.4f} ({:.4f}) | time: {:.4f} {}'.format(
                        epoch, batch_i, train_loss, cur_train_loss, accuracy, cur_accuracy, dt_tr, str_time_cost(time_cost)))

            batch_i += 1

            if epoch == 1 and batch_i in set([int(p * n_batches) for p in hparams.eval_within_epoch]):
                t0 = time.time()
                print('[EVAL] disable test_batch_size')
                run_eval(data_env, model, hparams, epoch, batch_i,
                         enable_test_bs=False,
                         disable_output_attention=not hparams.test_output_attention,
                         disable_analyze_attention=not hparams.test_analyze_attention)
                print('[EVAL] enable test_batch_size')
                run_eval(data_env, model, hparams, epoch, batch_i,
                         enable_test_bs=True,
                         disable_output_attention=True,
                         disable_analyze_attention=True)
                print('[EVAL] {:d}, {:d} | time: {:.4f}'.format(epoch, batch_i, time.time() - t0))

        t0 = time.time()
        print('[EVAL] disable test_batch_size')
        run_eval(data_env, model, hparams, epoch, batch_i,
                 enable_test_bs=False,
                 disable_output_attention=not hparams.test_output_attention,
                 disable_analyze_attention=not hparams.test_analyze_attention)
        print('[EVAL] enable test_batch_size')
        run_eval(data_env, model, hparams, epoch, batch_i,
                 enable_test_bs=True,
                 disable_output_attention=True,
                 disable_analyze_attention=True)
        print('[EVAL] {:d}, {:d} | time: {:.4f}'.format(epoch, batch_i, time.time()-t0))

    print('DONE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, choices=['FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995'])
    parser.add_argument('--n_dims_sm', type=int, default=None)
    parser.add_argument('--n_dims', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_edges_per_example', type=int, default=None)
    parser.add_argument('--max_edges_per_node', type=int, default=None)
    parser.add_argument('--max_attended_nodes', type=int, default=None)
    parser.add_argument('--max_seen_nodes', type=int, default=None)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--test_max_edges_per_example', type=int, default=None)
    parser.add_argument('--test_max_edges_per_node', type=int, default=None)
    parser.add_argument('--test_max_attended_nodes', type=int, default=None)
    parser.add_argument('--test_max_seen_nodes', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--aggregate_op', default=None)
    parser.add_argument('--uncon_steps', type=int, default=None)
    parser.add_argument('--con_steps', type=int, default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--clipnorm', type=float, default=None)
    parser.add_argument('--remove_all_head_tail_edges', action='store_true', default=None)
    parser.add_argument('--timer', action='store_true', default=None)
    parser.add_argument('--print_train', action='store_true', default=None)
    parser.add_argument('--print_train_metric', action='store_true', default=None)
    parser.add_argument('--print_train_freq', type=int, default=None)
    parser.add_argument('--eval_within_epoch', default=None)
    parser.add_argument('--eval_valid', action='store_true', default=None)
    parser.add_argument('--moving_mean_decay', type=float, default=None)
    parser.add_argument('--test_output_attention', action='store_true', default=None)
    parser.add_argument('--test_analyze_attention', action='store_true', default=None)
    args = parser.parse_args()

    default_parser = config.get_default_config(args.dataset)
    hparams = copy.deepcopy(default_parser.parse_args())
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr is not None:
            setattr(hparams, arg, attr)
    print(hparams)

    if hparams.dataset == 'NELL995':
        nell995_cls = getattr(datasets, hparams.dataset)
        for ds in nell995_cls.datasets():
            print('nell > ' + ds.name)
            if hparams.test_output_attention:
                dir_name = '../output/NELL995_subgraph/' + ds.name
                hparams.dir_name = dir_name
                if os.path.exists(dir_name):
                    shutil.rmtree(dir_name)
                os.makedirs(dir_name)
            run(ds, hparams)
    else:
        ds = getattr(datasets, hparams.dataset)()
        print(ds.name)
        if hparams.test_output_attention:
            dir_name = '../output/' + ds.name + '_subgraph'
            hparams.dir_name = dir_name
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.makedirs(dir_name)
        run(ds, hparams)
