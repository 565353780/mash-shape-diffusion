import os
import torch
import numpy as np
from torch.utils.data import Dataset

from mash_shape_diffusion.Config.shapenet import CATEGORY_IDS


class EncodedMashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        preload: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.preload = preload

        self.encoded_mash_folder_path = self.dataset_root_folder_path + "EncodedMash/"

        assert os.path.exists(self.encoded_mash_folder_path)

        self.path_dict_list = []

        dataset_name_list = os.listdir(self.encoded_mash_folder_path)

        for dataset_name in dataset_name_list:
            dataset_folder_path = self.encoded_mash_folder_path + dataset_name + "/"

            categories = os.listdir(dataset_folder_path)

            for i, category in enumerate(categories):
                class_folder_path = dataset_folder_path + category + "/"

                encoded_mash_filename_list = os.listdir(class_folder_path)

                print("[INFO][EncodedMashDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(i + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for encoded_mash_filename in encoded_mash_filename_list:
                    path_dict = {}
                    encoded_mash_file_path = class_folder_path + encoded_mash_filename

                    if not os.path.exists(encoded_mash_file_path):
                        continue

                    if self.preload:
                        encoded_mash = np.load(encoded_mash_file_path)
                        path_dict['encoded_mash'] = encoded_mash
                    else:
                        path_dict['encoded_mash'] = encoded_mash_file_path

                    path_dict['category_id'] = CATEGORY_IDS[category]

                    self.path_dict_list.append(path_dict)
        return

    def __len__(self):
        return len(self.path_dict_list)

    def __getitem__(self, index):
        index = index % len(self.path_dict_list)

        data = {}

        path_dict = self.path_dict_list[index]

        if self.preload:
            encoded_mash = path_dict['encoded_mash']
        else:
            encoded_mash_file_path = path_dict['encoded_mash']

            encoded_mash = np.load(encoded_mash_file_path)

        encoded_mash = torch.from_numpy(encoded_mash)

        data['encoded_mash'] = encoded_mash.float()

        data['category_id'] = path_dict['category_id']
        return data
