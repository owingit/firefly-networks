import networkx as nx
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import ndimage
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
        self.num_propagation_steps, self.nodes_active_at_time, self.map = self.count_coincident_components()
        self.nc_ij = self.normalized_count()

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
        choices_at_each_timestep = {i: 0 for i in range(cols)}
        timestep_propagation_map = []
        ones_indices = []
        for timestep in range(0, cols):
            for voxel in range(0, rows):
                if self.raster[voxel, timestep] == 1:
                    ones_indices.append([voxel, timestep])
        for pair in ones_indices:
            timestep = pair[1]
            choices_at_each_timestep[timestep] += 1
        for key in choices_at_each_timestep.keys():
            if choices_at_each_timestep.get(key+1) is not None:
                if (choices_at_each_timestep[key] * choices_at_each_timestep[key+1]) > 0:
                    timestep_propagation_map.append((key,
                                                     choices_at_each_timestep[key] * choices_at_each_timestep[key+1]))
                num_propagation_steps += (choices_at_each_timestep[key] * choices_at_each_timestep[key+1])
        return num_propagation_steps, choices_at_each_timestep, timestep_propagation_map

    def normalized_count(self):
        """Super inefficient, but the outcome should be a dict of nc_ij values.

        NC[i,j] = NC_ij

        :return: 2d nc_ij array
        """
        nc = np.zeros((self.raster.shape[0], self.raster.shape[0]))
        for i in range(self.raster.shape[0]):
            for j in range(self.raster.shape[0]):
                nc_ij = 0
                prop_index = 0
                running_sum = self.map[prop_index][1]
                for prop_step in range(1, self.num_propagation_steps-1):
                    if prop_step < running_sum:
                        t = self.map[prop_index][0]
                    else:
                        prop_index += 1
                        t = self.map[prop_index][0]
                        running_sum += self.map[prop_index][1]
                    t_plus_1 = t+1
                    if self.nodes_active_at_time.get(t):
                        coincident_flashers = self.raster[i, t] * self.raster[j, t_plus_1]
                        if coincident_flashers > 0 and self.nodes_active_at_time[t] > 0:
                            nc_ij += (coincident_flashers / self.nodes_active_at_time[t])

                nc[i, j] = nc_ij
        print(nc)
        return nc


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
