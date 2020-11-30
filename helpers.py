import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import networkx as nx

import logging


def do_single_shuffle(sub_raster, voxel_bin_pairs):
    """
    Attempts one shuffle with retries (with inf loop check) in sub_raster

    :param sub_raster: a matrix keyed [voxel ID, (sliced) time bin ID] to make
                       the shuffle in
    :param voxel_bin_pairs: a set of (voxel ID, active time bin ID) pairs
                            (shuffle also performed here for consistency)
    :return: None, performs shuffle in references
    """

    def attempt_shuffle():
        # Pick 2 indexes to try a swap of
        templist = list(voxel_bin_pairs)
        chosen_indices = np.random.choice(range(len(templist)), size=2, replace=False)

        tup0 = templist[chosen_indices[0]]
        tup1 = templist[chosen_indices[1]]

        voxel0 = tup0[0]
        bin0 = tup0[1]

        voxel1 = tup1[0]
        bin1 = tup1[1]

        # The condition to reject a proposed shuffle
        if (voxel0, bin1) in voxel_bin_pairs or (voxel1, bin0) in voxel_bin_pairs:
            return False

        # This is a sanity check to make sure the list and the raster are staying
        # consistent. Not necessary for the functioning of the algorithm.
        if sub_raster[voxel0, bin0] != 1:
            raise Exception("sub raster does not have 1s correctly according to list: voxel0")
        if sub_raster[voxel1, bin1] != 1:
            raise Exception("sub raster does not have 1s correctly according to list: voxel1")


        # Swap voxel0's bin
        sub_raster[voxel0, bin0] = 0
        sub_raster[voxel0, bin1] = 1

        # Swap voxel1's bin
        sub_raster[voxel1, bin1] = 0
        sub_raster[voxel1, bin0] = 1

        newtup0 = (voxel0, bin1)
        newtup1 = (voxel1, bin0)
        voxel_bin_pairs.remove(tup0)
        voxel_bin_pairs.remove(tup1)
        voxel_bin_pairs.add(newtup0)
        voxel_bin_pairs.add(newtup1)

        return True

    attempts = 0
    max_attempts = 50
    # Try a single shuffle, returns false if the proposed shuffle has equal time bins
    # (the condition for a retry). This loop is to avoid an infinite loop in an edge
    # case in which, for some reason, the only possible shuffles all have the same   time
    # bin (like a sub-raster consisting only of a single time bin).
    while attempts < max_attempts:
        if attempt_shuffle():
            return
        attempts += 1

    raise Exception("exceeded maximum shuffle attempts")


def shuffle_cascade(new_raster, min_bin, max_bin):
    """
    Does n_s shuffles of activation time bins in new_raster.

    :param new_raster: matrix keyed [voxel ID, time bin ID]
    :param min_bin: the bin to start shuffling at (inclusive)
    :param max_bin: the bin to stop shuffling at (inclusive)
    :return: None, performs shuffles in new_raster
    """

    # this is a VIEW, so edited the subraster will cause an edit
    # to new_raster
    # see: https://stackoverflow.com/questions/30917753/subsetting-a-2d-numpy-array
    sub_raster = new_raster[:, min_bin:max_bin + 1]

    # a set of tuples (voxel ID, (sliced) bin ID (so bin ID + min_bin is actual bin ID))
    voxel_bin_pairs = set()

    for voxel_id in range(sub_raster.shape[0]):
        for sliced_bin_id in range(sub_raster.shape[1]):
            if sub_raster[voxel_id, sliced_bin_id] == 1:
                voxel_bin_pairs.add((voxel_id, sliced_bin_id))

    # According to the paper, n_s ~= num active nodes in the cascade
    num_active_nodes = len(set(map(lambda x: x[0], voxel_bin_pairs)))
    n_s = num_active_nodes
    logging.debug("num active: %s" % num_active_nodes)
    for i in range(n_s):
        lenbefore = len(voxel_bin_pairs)
        do_single_shuffle(sub_raster, voxel_bin_pairs)
        lenafter = len(voxel_bin_pairs)

        if lenbefore != lenafter:
            raise Exception("broken invariant: reduced total activations")

    return sub_raster


def nc_shuffler(raster, clustered_timebins):
    """
    Takes the original raster and shuffles each cascade according to constrained
    pairwise shuffling, in which two voxels (active at some point during the cascade)
    are chosen and random and their time bins are swapped if they are not equal. If they
    are equal, a new pair is chosen.

    This swapping is performed n_s times per cascade, where n_s is the numbers of nodes
    active during that cascade.

    A new raster is returned based on the shuffled time bins.

    :param raster: 2d matrix, keyed [voxel ID, time bin ID]
    :param clustered_timebins: List of (time bin ID, cascade ID) pairs
    :return: new raster with shuffled time bins per cascade
    """

    new_raster = np.copy(raster)

    # build a dataframe of (time bin ID, cascade ID) for easier filtering to determine
    # min/max bin for each cascade
    cluster_bin_df = pd.DataFrame(clustered_timebins, columns=["binID", "cascadeID"])
    cascadeIDs = list(cluster_bin_df["cascadeID"].unique())
    for cascade_id in cascadeIDs:
        # determine min/max bin for the cascade
        filtered = cluster_bin_df.loc[cluster_bin_df["cascadeID"] == cascade_id]
        min_bin = filtered["binID"].min()
        max_bin = filtered["binID"].max()

        # sanity check
        if min_bin == max_bin:
            print("min and max bins are equal: %s, this shouldn't happen" % min_bin)
            raise Exception()

        shuffle_cascade(new_raster, min_bin, max_bin)

    return new_raster


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.
    from https://stackoverflow.com/questions/53958700/plotting-the-degree-distribution-of-a-graph-using-nx-degree-histogram
    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k, 0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k, 0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax = max(degseq)+1
    freq = [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq


def plot_directed_degree_dist(G):
    in_degree_freq = degree_histogram_directed(G, in_degree=True)
    out_degree_freq = degree_histogram_directed(G, out_degree=True)
    plt.figure(figsize=(12, 8))
    plt.loglog(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree')
    plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()


def plot_cc(G):
    gc = G.subgraph(max(nx.weakly_connected_components(G)))
    lcc = nx.clustering(gc)

    cmap = plt.get_cmap('autumn')
    norm = plt.Normalize(0, max(lcc.values()))
    node_colors = [cmap(norm(lcc[node])) for node in gc.nodes]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    nx.draw_spring(gc, node_color=node_colors, with_labels=True, ax=ax1)
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), label='Clustering', shrink=0.95, ax=ax1)

    ax2.hist(lcc.values(), bins=10)
    ax2.set_xlabel('Clustering')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()
