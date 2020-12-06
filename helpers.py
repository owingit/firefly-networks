import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

import networkx as nx

import copy

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
    max_attempts = 500
    # Try a single shuffle, returns false if the proposed shuffle has equal time bins
    # (the condition for a retry). This loop is to avoid an infinite loop in an edge
    # case in which, for some reason, the only possible shuffles all have the same time
    # bin (like a sub-raster consisting only of a single time bin).
    try:
        while attempts < max_attempts:
            if attempt_shuffle():
                return
            attempts += 1

        raise BaseException("exceeded maximum shuffle attempts")
    except BaseException:
        pass


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
    num_cascades = 0
    num_length_one_cascades = 0
    for cascade_id in cascadeIDs:
        num_cascades += 1
        # determine min/max bin for the cascade
        filtered = cluster_bin_df.loc[cluster_bin_df["cascadeID"] == cascade_id]
        min_bin = filtered["binID"].min()
        max_bin = filtered["binID"].max()
        try:
            # sanity check
            if min_bin == max_bin:
                print("min and max bins are equal: %s, this shouldn't happen" % min_bin)
                raise Exception()
        except Exception:
            num_length_one_cascades += 1

        shuffle_cascade(new_raster, min_bin, max_bin)

    return new_raster, num_cascades, num_length_one_cascades


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


def plot_directed_degree_dist(list_of_Gs):
    """Plots in-degree and out-degree distribution across all graphs in list

    :param list_of_Gs: list of nx graphs made from np a_ijs
    """
    in_degree_freq = {}
    out_degree_freq = {}
    for G in list_of_Gs:
        in_degrees = degree_histogram_directed(G, in_degree=True)
        for degree in range(len(in_degrees)):
            if in_degree_freq.get(degree):
                in_degree_freq[degree] += in_degrees[degree]
            else:
                in_degree_freq[degree] = in_degrees[degree]
        out_degrees = degree_histogram_directed(G, out_degree=True)
        for o_degree in range(len(out_degrees)):
            if out_degree_freq.get(o_degree):
                out_degree_freq[o_degree] += out_degrees[o_degree]
            else:
                out_degree_freq[o_degree] = out_degrees[o_degree]

    plt.figure(figsize=(12, 6))
    plt.loglog(list(in_degree_freq.keys()), list(in_degree_freq.values()), 'go-', label='in-degree dist')
    plt.loglog(list(out_degree_freq.keys()), list(out_degree_freq.values()), 'bo-', label='out-degree dist')
    plt.legend()
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.show()


def plot_cc(list_of_Gs):
    """Plots the clustering coefficient for graphs in the list passed.
    Currently only plots the first one; easily extensible to be avgs, all graphs, etc

    :param list_of_Gs: list of nx graphs made from np a_ij
    """
    for G in [list_of_Gs[0]]:
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


def plot_flash_emergence(list_of_Gs, list_of_cascade_startpoints, list_of_cascade_endpoints, do_networks=False):
    """Plots nodes and edges of voxels / labeled agents in cascades

    :param list_of_Gs: list of nx graphs made from np a_ij
    """
    seed_set_sizes = []
    final_set_sizes = []
    for index in range(len(list_of_cascade_startpoints)):
        cascade_startpoint = list_of_cascade_startpoints[index]
        cascade_endpoint = list_of_cascade_endpoints[index]
        G = copy.deepcopy(list_of_Gs[0])
        cmap = plt.get_cmap('autumn')

        norm = plt.Normalize(cascade_startpoint, cascade_endpoint)
        nodes_in_cascade = [node for node in G.nodes if (
                 any([cascade_endpoint >= time >= cascade_startpoint for time in G.nodes[node]['times']]))]
        node_effective_times = {node: G.nodes[node]['times'] for node in nodes_in_cascade}
        for node in node_effective_times.keys():
            times_to_remove = []
            for time in node_effective_times[node]:
                if time < cascade_startpoint:
                    times_to_remove.append(time)
                elif time > cascade_endpoint:
                    times_to_remove.append(time)
            for time in times_to_remove:
                node_effective_times[node].remove(time)
        node_colors = cmap(norm([min(node_effective_times[node]) for node in nodes_in_cascade]))
        nodes_to_remove = []

        for n in G.nodes():
            if n not in nodes_in_cascade:
                nodes_to_remove.append(n)
        for node in nodes_to_remove:
            G.remove_node(node)

        first_node_time = min([time for x in node_effective_times.keys() for time in node_effective_times[x]])
        seed_set = [node for node in nodes_in_cascade if first_node_time in G.nodes[node]['times']]
        len_seed_set = len(seed_set)
        seed_set_sizes.append(len_seed_set)
        num_nodes_in_cascade = len(G.nodes())
        final_set_sizes.append(num_nodes_in_cascade)
        edges_to_remove = []
        for u, v in G.edges():
            if min(node_effective_times[v]) < min(node_effective_times[u]):
                edges_to_remove.append((u, v))
        degrees = dict(G.degree())

        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])
        if do_networks:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            xs = [G.nodes[node]['coords'][0] for node in G.nodes()]
            ys = [G.nodes[node]['coords'][1] for node in G.nodes()]
            node_indices = list(G.nodes())
            graph_layout = dict(zip(node_indices, zip(xs, ys)))

            nx.draw(G, pos=graph_layout, ax=ax1, node_color=node_colors, node_size=[(v+1)*10 for v in degrees.values()])
            ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            limits = plt.axis('on')
            ax1.set_xlabel('X embedded position')
            ax1.set_ylabel('Y embedded position')
            ax1.set_xlim(-15, 15)
            ax1.set_ylim(-15, 15)
            cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), label='Time of flash in cascade', shrink=0.95, ax=ax1)
            cblocations = [cascade_startpoint + 2, cascade_endpoint - 2]
            cblabels = ['Earlier', 'Later']
            cbar.set_ticks(cblocations)
            cbar.set_ticklabels(cblabels)
            plt.title('Cascade {}: {} nodes -> {} nodes'.format(index, len_seed_set, num_nodes_in_cascade))
            plt.show()
    return seed_set_sizes, final_set_sizes


def plot_size_distributions(initial_set_distributions, final_set_distributions, start_of_list):
    fig, ((ax1), (ax2), (ax3)) = plt.subplots(ncols=3, nrows=1)
    ax1.set_xlim([-1, max([item for d in initial_set_distributions for item in d]) + 5])
    ax1.set_title('Seed node set size')
    ax1.set_xlabel('Num flashers at beginning of cascades')
    ax1.set_ylabel('Count')
    for e, initial_set_distribution in enumerate(initial_set_distributions):
        cascade_length = e + start_of_list
        ax1.hist(initial_set_distribution, label='Cascade size 0.{}s'.format(cascade_length),
                 bins=10, alpha=0.5, rwidth=0.5, align='left', density=True)
        ax1.legend()

    ax2.set_xlim([-1, max([item for d in final_set_distributions for item in d]) + 5])
    ax2.set_title('Cascade size dict')
    ax2.set_xlabel('Num flashers at end of cascades')
    ax2.set_ylabel('Count')

    ax3.set_xlabel('Num flashers at end of cascades')
    ax3.set_ylabel('Count')
    ax3.set_title('Log-log of cascade size')

    for i, final_set_distribution in enumerate(final_set_distributions):
        cascade_length = i + start_of_list
        n, x, _ = ax2.hist(final_set_distribution, label='Cascade size 0.{}s'.format(cascade_length),
                           bins=10, alpha=0.5, rwidth=0.5, align='left', density=True)
        ax2.legend()
        bin_centers = 0.5 * (x[1:] + x[:-1])
        ax3.loglog(bin_centers, n, label='Cascade size 0.{}s'.format(cascade_length))
        ax3.legend()

    plt.show()
