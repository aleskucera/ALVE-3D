import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../cut-pursuit/build/src'))

import libcp
import numpy as np
from jakteristics import compute_features, FEATURE_NAMES

CP_CUTOFF = 25
REG_STRENGTH = 10
EDGE_WEIGHT_THRESHOLD = -0.5
FEATURES = ['anisotropy', 'planarity', 'linearity', 'PCA2',
            'sphericity', 'verticality', 'surface_variation']


def partition_cloud(points: np.ndarray, edge_sources: np.ndarray, edge_targets: np.ndarray, colors: np.ndarray = None):
    embeddings = compute_features(points.astype(np.double), search_radius=2,
                                  max_k_neighbors=1000, feature_names=FEATURES)
    embeddings[np.isnan(embeddings)] = -1
    if colors is not None:
        embeddings = np.concatenate([embeddings, colors], axis=-1)
    embeddings = np.concatenate([embeddings, points], axis=-1)

    dist = ((embeddings[edge_sources, :] - embeddings[edge_targets, :]) ** 2).sum(1)
    edge_weight = np.exp(dist * EDGE_WEIGHT_THRESHOLD) / np.exp(EDGE_WEIGHT_THRESHOLD)

    components, component_map = libcp.cutpursuit(embeddings.astype(np.float32),
                                                 edge_sources.astype(np.uint32),
                                                 edge_targets.astype(np.uint32),
                                                 edge_weight.astype(np.float32),
                                                 REG_STRENGTH,
                                                 cutoff=CP_CUTOFF, spatial=1,
                                                 weight_decay=0.2)
    return np.array(components, dtype=np.int64), np.array(component_map, dtype=np.int64)


def calculate_features(points: np.ndarray) -> dict:
    geometric_features = compute_features(points.astype(np.double), search_radius=2,
                                          max_k_neighbors=1000, feature_names=FEATURE_NAMES)
    geometric_features[np.isnan(geometric_features)] = -1
    ret = {}
    for feature in FEATURE_NAMES:
        ret[feature] = geometric_features[..., FEATURE_NAMES.index(feature)]
    return ret
