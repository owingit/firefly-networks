import networkx as nx
from sklearn.cluster import KMeans

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
    def __init__(self, data_file, do_3d=False):
        self.raster_dict, self.raster = self.clean_zs_from_data_file(data_file, do_3d)
        self.num_propagation_steps, self.ijs_at_each_timestep, ones_indices = self.count_coincident_components()
        self.num_cascades, self.clustered_timesteps = self.count_cascades(ones_indices)
        self.nc_ij = {}
        print(self.clustered_timesteps)
        for cascade in range(self.num_cascades):
            timesteps_in_cascade = [cts[0] for cts in self.clustered_timesteps if cts[1] == cascade]
            min_t = min(timesteps_in_cascade)
            max_t = max(timesteps_in_cascade)
            self.nc_ij[cascade] = self.normalized_count(min_t=min_t, max_t=max_t)
        print(self.nc_ij)

    def count_cascades(self, flash_occurrences):
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
    def clean_zs_from_data_file(data_file, do_3d=False):
        """Remove zs and convert to raster for analysis"""
        raster_dict = OrderedDict()
        max_t = 0
        num_voxels = len(data_file.keys())
        for voxel, timesteps in data_file.items():
            if not do_3d:
                vox = (voxel[0], voxel[1])
            else:
                vox = voxel
            if max(timesteps) > max_t:
                max_t = max(timesteps) + 1
            key = min(timesteps)
            raster_dict[(int(key), vox[0], vox[1])] = sorted(timesteps)
        raster = np.zeros((num_voxels, int(max_t)))
        for i, voxel in enumerate(sorted(raster_dict.keys(), key=lambda x: x[0])):
            # i is the index in the Nt x N sparse matrix of the current voxel
            # 1s should indicate source nodes
            for timestep in raster_dict[voxel]:
                raster[i][int(timestep)] = 1
        return raster_dict, raster

    def count_coincident_components(self):
        """Use image labeling to count coincident components"""
        rows = self.raster.shape[0]
        cols = self.raster.shape[1]

        num_propagation_steps = 0
        ijs_at_each_timestep = {i: [] for i in range(cols)}
        ones_indices = []
        for timestep in range(0, cols):
            for voxel in range(0, rows):
                if self.raster[voxel, timestep] == 1:
                    ones_indices.append((voxel, timestep))
        for pair in ones_indices:
            timestep = pair[1]
            nxt = timestep + 1
            partner_list = [p for p in ones_indices if nxt == p[1]]
            coincident_partnerships = [(pair[0], pl[0]) for pl in partner_list]
            ijs_at_each_timestep[timestep].extend(coincident_partnerships)
        for key in ijs_at_each_timestep.keys():
            if ijs_at_each_timestep.get(key+1) is not None:
                num_propagation_steps += (len(ijs_at_each_timestep[key]))
        return num_propagation_steps, ijs_at_each_timestep, ones_indices

    def normalized_count(self, min_t, max_t):
        """Super inefficient, but the outcome should be a 2d array of nc_ij values.

        NC[i,j] = NC_ij

        :param min_t: min timestep of current cascade
        :param max_t: max timestep of current cascade
        :return: 2d nc_ij array
        """
        nc = np.zeros((self.raster.shape[0], self.raster.shape[0]))
        for propagation_index in self.ijs_at_each_timestep:
            if len(self.ijs_at_each_timestep.get(propagation_index)) == 0:
                continue
            elif min_t > propagation_index or propagation_index > max_t:
                continue
            else:
                for (i, j) in self.ijs_at_each_timestep[propagation_index]:
                    if i == 0:
                        print(i, j)
                    nc_ij = (1.0 / len(self.ijs_at_each_timestep[propagation_index]))
                    nc[i, j] += nc_ij
        normalized_counts = normalize(nc, axis=1, norm='l2')
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
normalized_count = NormalizedCount(dw_test.real_voxels_to_activation_times, do_3d=do_3d)
# fig, ax = visualize_voxels_and_points(dw_test.real_voxels_to_activation_times, dw_test.df, dw_test.voxel_length)
# ax.set_title("Simulated with Voxel Length 0.5")
# plt.show()
