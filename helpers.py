import numpy as np
import pandas as pd

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
    max_attempts = 10
    # Try a single shuffle, returns false if the proposed shuffle has equal time bins
    # (the condition for a retry). This loop is to avoid an infinite loop in an edge
    # case in which, for some reason, the only possible shuffles all have the same time
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
