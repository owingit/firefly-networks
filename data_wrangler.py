import networkx as nx
import os
import pandas as pd


class DataWrangler:
    def __init__(self, file, do_3d=False, is_labeled=False):
        filename = file
        self.dfs = []
        self.real_voxels_to_activation_times = []
        if not is_labeled:
            for _f in os.listdir(filename):
                if do_3d:
                    self.df = pd.read_csv(filename+'/'+_f, names=["x", "y", "z", "t"])
                else:
                    self.df = pd.read_csv(filename+'/'+_f, names=["x", "y", "label", "t"])
                self.df["x_round"] = self.df["x"].apply(lambda x: round(x, 2))
                self.df["y_round"] = self.df["y"].apply(lambda x: round(x, 2))
                if do_3d:
                    self.df["z_round"] = self.df["z"].apply(lambda x: round(x, 2))

                self.min_x = self.df["x"].min()
                self.min_y = self.df["y"].min()
                if do_3d:
                    self.min_z = self.df["z"].min()
                else:
                    self.min_z = None
                self.voxel_length = 0.75

                voxels_to_activation_times = self.pair_voxels_with_activation_times(do_3d=do_3d)
                self.active_voxel_coords = list(map(lambda x: self.voxel_to_positions(x[0], x[1], x[2]),
                                                    voxels_to_activation_times.keys()))

                real_voxels_to_activation_times = {self.voxel_to_positions(key[0], key[1], key[2]): ts
                                                   for key, ts in voxels_to_activation_times.items()
                                                   }
                self.dfs.append(self.df)
                self.real_voxels_to_activation_times.append(real_voxels_to_activation_times)
        else:
            if not os.path.isdir(file):
                self.read_from_csv_into_df(filename)
            else:
                for _f in os.listdir(file):
                    self.read_from_csv_into_df(file+'/'+_f)

    def read_from_csv_into_df(self, filename):
        df = pd.read_csv(filename, names=["x", "y", "label", "t"])
        labels_and_times = {}
        for index, row in df.iterrows():
            label = row["label"]
            t = row["t"]

            if label in labels_and_times:
                labels_and_times[label].append(t)
            else:
                labels_and_times[label] = [t]

        # make a voxel = (label, label, label) in the labeled case
        real_voxels_to_activation_times = {(key, key, key): ts
                                           for key, ts in labels_and_times.items()
                                           }
        self.dfs.append(df)
        self.real_voxels_to_activation_times.append(real_voxels_to_activation_times)

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