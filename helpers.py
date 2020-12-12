import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

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


def plot_flash_emergence(list_of_Gs, list_of_cascade_startpoints, list_of_cascade_endpoints,
                         do_networks=False, do_3d=False, do_betweenness=False):
    """Plots nodes and edges of voxels / labeled agents in cascades

    :param list_of_Gs: list of nx graphs made from np a_ij
    :param list_of_cascade_startpoints
    :param list_of_cascade_endpoints
    :param do_networks: whether to plot nets
    :param do_3d: whether to calc and plot in 3d
    :param do_betweenness: whether to calc and plot betweenness centrality
    """
    do_3d = False
    seed_set_sizes = []
    final_set_sizes = []
    blank = 0
    high_centrality_positions = []
    for index in range(len(list_of_cascade_startpoints)):
        cascade_startpoint = list_of_cascade_startpoints[index]
        cascade_endpoint = list_of_cascade_endpoints[index]

        G = copy.deepcopy(list_of_Gs[0])
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

        cmap = plt.get_cmap('autumn')
        norm = plt.Normalize(cascade_startpoint, cascade_endpoint)
        node_colors = cmap(norm([min(node_effective_times[node]) for node in nodes_in_cascade]))

        nodes_to_remove = []
        for n in G.nodes():
            if n not in nodes_in_cascade:
                nodes_to_remove.append(n)
        for node in nodes_to_remove:
            G.remove_node(node)

        if len(node_effective_times.keys()) == 0:
            blank += 1
            len_seed_set = 0
            num_nodes_in_cascade = 0
        else:
            blank = 0
            first_node_time = min([time for x in node_effective_times.keys() for time in node_effective_times[x]])
            seed_set = [node for node in nodes_in_cascade if first_node_time in G.nodes[node]['times']]
            len_seed_set = len(seed_set)
            num_nodes_in_cascade = len(G.nodes())

        if blank >= 5:
            #  skip gaps
            continue

        seed_set_sizes.append(len_seed_set)
        final_set_sizes.append(num_nodes_in_cascade)
        edges_to_remove = []
        for u, v in G.edges():
            if min(node_effective_times[v]) < min(node_effective_times[u]):
                edges_to_remove.append((u, v))
        degrees = dict(G.degree())
        for edge in edges_to_remove:
            G.remove_edge(edge[0], edge[1])

        if do_betweenness:
            _centralities = nx.betweenness_centrality(G)
            max_key = max(_centralities, key=_centralities.get)
            max_key_pos = G.nodes[max_key]['coords']
            high_centrality_positions.append(max_key_pos)
            _centralities_norm = plt.Normalize(0, max(_centralities.values()))
            node_colors_ = cmap(_centralities_norm([_centralities[node] for node in G.nodes]))
            node_alphas_ = _centralities_norm([_centralities[node] for node in G.nodes])
            node_color_alphas_ = []
            edge_color_alphas = []
            for i, node_color in enumerate(node_colors_):
                r, g, b = node_color[0], node_color[1], node_color[2]
                a = node_alphas_[i]
                if a < 0.1:
                    a = 0.1
                edge_color_alphas.append([0, 0, 0, a])
                node_color_alphas_.append([r, g, b, a])
        else:
            _centralities_norm = None
            node_color_alphas_ = None
            edge_color_alphas = None
            node_colors_ = None

        if do_networks:
            _x_ = 0
            _y_ = 1
            _z_ = 2
            fig = plt.figure(figsize=(10, 7))
            if do_3d:
                ax1 = Axes3D(fig)
            else:
                ax1 = fig.add_subplot(111)
            xs = [G.nodes[node]['coords'][_x_] for node in G.nodes()]
            ys = [G.nodes[node]['coords'][_y_] for node in G.nodes()]
            if do_3d:
                zs = [G.nodes[node]['coords'][_z_] for node in G.nodes()]
                node_indices = list(G.nodes())
                graph_layout = dict(zip(node_indices, zip(xs, ys, zs)))
            else:
                node_indices = list(G.nodes())
                graph_layout = dict(zip(node_indices, zip(xs, ys)))
            if do_3d:
                for i, (key, value) in enumerate(graph_layout.items()):
                    xi = value[_x_]
                    yi = value[_y_]
                    zi = value[_z_]
                    if do_betweenness:
                        # Scatter plot
                        ax1.scatter(xi, yi, zi, c=node_color_alphas_[i], s=20 + 20 * G.degree(key),
                                    edgecolors=edge_color_alphas[i])
                    else:
                        ax1.scatter(xi, yi, zi, c=node_colors[i], s=20 + 20 * G.degree(key), edgecolors='k')
                for i, j in enumerate(G.edges()):
                    x = np.array((graph_layout[j[0]][_x_], graph_layout[j[1]][_x_]))
                    y = np.array((graph_layout[j[0]][_y_], graph_layout[j[1]][_y_]))
                    z = np.array((graph_layout[j[0]][_z_], graph_layout[j[1]][_z_]))
                    # Plot the connecting lines
                    arw = Arrow3D([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], arrowstyle="->",
                                  color="black",
                                  lw=1, mutation_scale=25)
                    ax1.add_artist(arw)
                ax1.tick_params(left=True, bottom=True, right=True, labelright=True,
                                labelleft=True, labelbottom=True)
            else:
                if do_betweenness:
                    nx.draw(G, pos=graph_layout, ax=ax1, node_color=node_colors_,
                            node_size=[(v + 1) * 10 for v in degrees.values()])
                else:
                    nx.draw(G, pos=graph_layout, ax=ax1, node_color=node_colors,
                            node_size=[(v + 1) * 10 for v in degrees.values()])
                ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            limits = plt.axis('on')
            ax1.set_xlabel('X embedded position')
            ax1.set_ylabel('Y embedded position')
            if do_3d:
                ax1.set_zlabel('Z embedded position')
            ax1.set_xlim(-20, 20)
            ax1.set_ylim(-20, 20)
            if do_3d:
                ax1.set_zlim(-20, 20)
            if do_betweenness:
                label = 'Betweenness centrality'
                cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=_centralities_norm), label=label, shrink=0.95,
                                    ax=ax1, )
            else:
                label = 'Time of flash in cascade'
                cbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), label=label, shrink=0.95,
                                    ax=ax1, )
            cblocations = [cascade_startpoint, 10]
            if do_betweenness:
                cblabels = ['Lower', 'Higher']
            else:
                cblabels = ['Earlier', 'Later']
            cbar.set_ticks(cblocations)
            cbar.set_ticklabels(cblabels)
            plt.title('Steps {}-{}: {} nodes -> {} nodes'.format(
                cascade_startpoint, cascade_endpoint, len_seed_set, num_nodes_in_cascade))
            plt.show()
            if do_3d:
                stri_augment = '3d_'
            else:
                stri_augment = '_'
            if do_betweenness:
                stri = 'a_ij_data_smaller_cascades/nets/{}btw_steps{}to{}_{}_nodes_to_{}_nodes.png'.format(
                    stri_augment, cascade_startpoint, cascade_endpoint, len_seed_set, num_nodes_in_cascade)
            else:
                stri = 'a_ij_data_smaller_cascades/nets/{}flashplot_steps{}to{}_{}_nodes_to_{}_nodes.png'.format(
                    stri_augment, cascade_startpoint, cascade_endpoint, len_seed_set, num_nodes_in_cascade)
            plt.savefig(stri)
            plt.close()
    return seed_set_sizes, final_set_sizes, high_centrality_positions


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


def plot_high_centrality_positions(list_of_position_lists, do_3d=False):
    for position_list in list_of_position_lists:
        if not do_3d:
            xs = [p[0] for p in position_list]
            ys = [p[1] for p in position_list]
            H, x, y = np.histogram2d(xs, ys, bins=10, density=True)
            fig = plt.figure(figsize=(7, 3))
            ax = fig.add_subplot(131, title='Dist of high betweenness locations')
            plt.imshow(H, interpolation='nearest', origin='lower',
                       extent=[x[0], x[-1], y[0], y[-1]])
            plt.show()
        else:
            print('unsupported: 3d histogram of betweenness')
            # this doesn't work, oh well
            # xyzs = [(p[0], p[1], p[2]) for p in position_list]
            #
            # H, edges = np.histogramdd(xyzs, bins=10, density=True)
            # fig = plt.figure(figsize=(7, 3))
            # ax = Axes3D(fig, title='Dist of high betweenness locations')
            # plt.imshow(np.reshape(H, (-1, 2)), interpolation='nearest', origin='lower', aspect='auto',
            #            extent=[edges[0][0],
            #                    edges[0][-1],
            #                    edges[0][0],
            #                    edges[0][-1]
            #                    ])
            # plt.show()


class Arrow3D(FancyArrowPatch):
    # https://stackoverflow.com/questions/38194247/how-can-i-connect-two-points-in-3d-scatter-plot-with-arrow
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
