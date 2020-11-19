import networkx as nx
from sklearn.cluster import KMeans

import math
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_FILE = "0.390625density0.1betadistributionTb_obstacles1600_steps_experiment_results_2020-11-09_11:05:25.960570_csv.csv"


class NormalizedCount:
    # Input:
    # (i)  A raster of activation data, recorded as (time,node) pairs,i.e. (te, se);
    # (ii) p-value for the reconstruction, p.

    # Output: The reconstructed weighted directed network
    def __init__(self, voxel_coords_to_ts, time_bin_length=1, do_3d=False):
        self.time_bin_length = time_bin_length
        self.raster = self.build_raster(voxel_coords_to_ts, do_3d, self.time_bin_length)
        self.num_propagation_steps, self.ijs_at_each_timebin, ones_indices = self.count_coincident_components()
        self.num_cascades, self.clustered_timesteps = self.count_cascades(ones_indices)
        self.nc_ij = {}
        for cascade in range(self.num_cascades):
            timesteps_in_cascade = [cts[0] for cts in self.clustered_timesteps if cts[1] == cascade]
            min_t = min(timesteps_in_cascade)
            max_t = max(timesteps_in_cascade)
            self.nc_ij[cascade] = self.normalized_count(min_t=min_t, max_t=max_t)
        print(self.nc_ij)

    @staticmethod
    def count_cascades(flash_occurrences):
        """ Count the number of cascades and label timesteps to their cascade using kmeans

        :param flash_occurrences: Timesteps of flashes
        :return: number of clusters, and list of flash -> cascade label pairs
        """
        list_of_flash_timesteps = [fo[1] for fo in flash_occurrences]
        loft = np.array(list(set(list_of_flash_timesteps)))
        thresh = 9
        num_clusters = 0
        for i in range(len(loft) - 1):
            j = i + 1
            if loft[j] - loft[i] > thresh:
                num_clusters += 1
        km = KMeans(n_clusters=num_clusters)

        km.fit(loft.reshape(-1, 1))
        q = list(zip(loft, km.labels_))

        return num_clusters, q

    @staticmethod
    def build_raster(voxel_coords_to_ts, do_3d=False, time_bin_length=1):
        """Convert to raster for analysis

        :param voxel_coords_to_ts: voxel cooordinates and timesteps at which they are active
        :param do_3d: whether to do zcoord
        :param time_bin_length: Time delta
        :returns: raster of 1s and 0s for active time bins
        """
        raster_dict = OrderedDict()
        max_t = 0
        num_voxels = len(voxel_coords_to_ts.keys())
        for voxel, timesteps in voxel_coords_to_ts.items():
            if not do_3d:
                vox = (voxel[0], voxel[1])
            else:
                vox = voxel
            if max(timesteps) > max_t:
                max_t = max(timesteps) + 1
            key = min(timesteps)
            raster_dict[(int(key), vox[0], vox[1])] = sorted(timesteps)

        raster = np.zeros((num_voxels, int(max_t / time_bin_length)))
        if time_bin_length > 1:
            for voxel, timesteps in raster_dict.items():
                effective_time_bins = np.zeros(int(max_t / time_bin_length))
                for timestep in timesteps:
                    effective_time_bins[int(timestep / time_bin_length)] = 1
                raster_dict[voxel] = [float(time_bin) for time_bin in range(len(effective_time_bins))
                                      if effective_time_bins[time_bin] > 0]

        for i, voxel in enumerate(sorted(raster_dict.keys(), key=lambda x: x[0])):
            # i is the index in the Nt x N sparse matrix of the current voxel
            # 1s should indicate source nodes
            for timestep in raster_dict[voxel]:
                raster[i][int(timestep)] = 1
        return raster

    def count_coincident_components(self):
        """

        :return: number of total propagations (flash moments crossed with coincident (flash at t+x) moments)
        :return: dict of potential propagation steps per timestep
        :return: list of (node, timestep) flash points (1s in the raster)
        """
        rows = self.raster.shape[0]
        cols = self.raster.shape[1]

        num_propagation_steps = 0
        ijs_at_each_timebin = {i: [] for i in range(cols)}
        ones_indices = []
        for timebin in range(0, cols):
            for voxel in range(0, rows):
                if self.raster[voxel, timebin] == 1:
                    ones_indices.append((voxel, timebin))

        for node_time_pair in ones_indices:
            timebin = node_time_pair[1]
            nxt_timebin = timebin + 1
            partner_list = [p for p in ones_indices if timebin < p[1] <= nxt_timebin]
            coincident_partnerships = [(node_time_pair[0], possible_partner[0]) for possible_partner in partner_list]
            ijs_at_each_timebin[timebin].extend(coincident_partnerships)

        for key in ijs_at_each_timebin.keys():
            if ijs_at_each_timebin.get(key+1) is not None:
                num_propagation_steps += (len(ijs_at_each_timebin[key]))

        return num_propagation_steps, ijs_at_each_timebin, ones_indices

    def normalized_count(self, min_t, max_t):
        """Not so inefficient now! Uses a mapping of t -> coincident t, t+1 flashers to calculate NC_ij

        NC[i,j] = NC_ij

        :param min_t: min timestep of current cascade
        :param max_t: max timestep of current cascade
        :return: 2d nc_ij array
        """
        nc = np.zeros((self.raster.shape[0], self.raster.shape[0]))
        for t in self.ijs_at_each_timebin.keys():
            if len(self.ijs_at_each_timebin.get(t)) == 0:
                continue
            elif min_t > t or t > max_t:
                continue
            else:
                for (i, j) in self.ijs_at_each_timebin[t]:
                    nodes_active_at_time_t = set([x for (x, y) in self.ijs_at_each_timebin[t]])
                    nc_ij = (1.0 / len(nodes_active_at_time_t))
                    nc[i, j] += nc_ij
        normalized_counts = nc / self.num_propagation_steps
        return normalized_counts


class DataWrangler:
    def __init__(self, file, do_3d=False):
        filename = file
        if do_3d:
            self.df = pd.read_csv(filename, names=["x", "y", "z", "t"])
        else:
            self.df = pd.read_csv(filename, names=["x", "y", "t"])
        self.df["x_round"] = self.df["x"].apply(lambda x: round(x, 2))
        self.df["y_round"] = self.df["y"].apply(lambda x: round(x, 2))
        if do_3d:
            self.df["z_round"] = self.df["z"].apply(lambda x: round(x, 2))
        print("unique x:", len(self.df["x"].unique()))
        print("unique x rounded:", len(self.df["x_round"].unique()))
        print("unique y:", len(self.df["y"].unique()))
        print("unique y rounded:", len(self.df["y_round"].unique()))
        if do_3d:
            print("unique z:", len(self.df["z"].unique()))
            print("unique z rounded:", len(self.df["z_round"].unique()))

        self.min_x = self.df["x"].min()
        self.min_y = self.df["y"].min()
        if do_3d:
            self.min_z = self.df["z"].min()
        else:
            self.min_z = None
        self.voxel_length = 0.5

        voxels_to_activation_times = self.pair_voxels_with_activation_times(do_3d=do_3d)
        self.active_voxel_coords = list(map(lambda x: self.voxel_to_positions(x[0], x[1]),
                                            voxels_to_activation_times.keys()))

        self.real_voxels_to_activation_times = {self.voxel_to_positions(key[0], key[1]): ts
                                                for key, ts in voxels_to_activation_times.items()
                                                }

    @staticmethod
    def adjust_start_0(val, overall_min):
        """Adjusts a value's range to start at 0 so it can be cleanly divided for voxel number

        :param val: value to adjust
        :param overall_min: min value with which to adjust
        :return: adjusted value
        """
        if overall_min < 0:
            return val + (-overall_min)
        else:
            return val - overall_min

    def xyz_to_voxel_xyz(self, x, y, z=None):
        """Convert coords to voxel coords

        :param x: x coord
        :param y: y coord
        :param z: z coord (optional)
        :return:
        """
        x_adj = self.adjust_start_0(x, self.min_x)
        y_adj = self.adjust_start_0(y, self.min_y)
        z_adj = None
        if z is not None and self.min_z is not None:
            z_adj = self.adjust_start_0(z, self.min_z)

        if z_adj is not None:
            return x_adj // self.voxel_length, y_adj // self.voxel_length, z_adj // self.voxel_length
        else:
            return x_adj // self.voxel_length, y_adj // self.voxel_length, None

    def pair_voxels_with_activation_times(self, do_3d=False):
        """Do what it says

        :param do_3d: whether to use the z dimension
        :return: Dictionary, keys = voxels, values = time of the voxel's appearance
        """
        voxels_to_activation_times = {}

        for index, row in self.df.iterrows():
            x = row["x"]
            y = row["y"]
            t = row["t"]
            if do_3d:
                z = row["z"]
            else:
                z = None

            voxel = self.xyz_to_voxel_xyz(x, y, z)
            if voxel in voxels_to_activation_times:
                voxels_to_activation_times[voxel].append(t)
            else:
                voxels_to_activation_times[voxel] = [t]

        return voxels_to_activation_times

    def voxel_to_positions(self, vx, vy, vz=None):
        x = (vx * self.voxel_length) + self.min_x
        y = (vy * self.voxel_length) + self.min_y
        z = None
        if vz is not None and self.min_z is not None:
            z = (vz * self.voxel_length) + self.min_z

        return x, y, z


def visualize_voxels_and_points(voxeled, df, voxel_length):
    """Plots voxels

    :param voxeled: voxelified data
    :param df: actual data
    :param voxel_length: voxel length
    :return: figure and axis for plotting
    """
    figur, axys = plt.subplots(figsize=(12, 12))

    axys.scatter(df["x"], df["y"], s=1.5, alpha=1, color="blue")
    axys.set_xlabel("x")
    axys.set_ylabel("y")

    for x, y, z in voxeled:
        # https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
        rect = patches.Rectangle((x, y),
                                 voxel_length,
                                 voxel_length,
                                 facecolor=None,
                                 edgecolor="r",
                                 linewidth=1,
                                 fill=False)
        axys.add_patch(rect)
    return figur, axys


do_3d = False
dw_test = DataWrangler(_FILE, do_3d=do_3d)
normalized_count = NormalizedCount(dw_test.real_voxels_to_activation_times, time_bin_length=2, do_3d=do_3d)
# fig, ax = visualize_voxels_and_points(dw_test.real_voxels_to_activation_times, dw_test.df, dw_test.voxel_length)
# ax.set_title("Simulated with Voxel Length 0.5")
# plt.show()
