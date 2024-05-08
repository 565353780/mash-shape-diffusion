import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ImageEmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        preload: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.preload = preload

        self.encoded_mash_folder_path = self.dataset_root_folder_path + "EncodedMash/"
        self.image_embedding_folder_path = self.dataset_root_folder_path + "ImageEmbedding/"

        assert os.path.exists(self.encoded_mash_folder_path)
        assert os.path.exists(self.image_embedding_folder_path)

        self.path_dict_list = []

        dataset_name_list = os.listdir(self.encoded_mash_folder_path)

        for dataset_name in dataset_name_list:
            dataset_folder_path = self.encoded_mash_folder_path + dataset_name + "/"

            categories = os.listdir(dataset_folder_path)

            for i, category in enumerate(categories):
                class_folder_path = dataset_folder_path + category + "/"

                encoded_mash_filename_list = os.listdir(class_folder_path)

                print("[INFO][ImageEmbeddingDataset::__init__]")
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

                    image_embedding_file_path = self.image_embedding_folder_path + dataset_name + '/' + \
                        category + '/' + encoded_mash_filename

                    if not os.path.exists(image_embedding_file_path):
                        continue

                    if self.preload:
                        encoded_mash = np.load(encoded_mash_file_path)
                        image_embedding = np.load(image_embedding_file_path, allow_pickle=True).item()
                        path_dict['encoded_mash'] = encoded_mash
                        path_dict['image_embedding'] = image_embedding
                    else:
                        path_dict['encoded_mash'] = encoded_mash_file_path
                        path_dict['image_embedding'] = image_embedding_file_path

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
            image_embedding = path_dict['image_embedding']
        else:
            encoded_mash_file_path = path_dict['encoded_mash']
            image_embedding_file_path = path_dict['image_embedding']
            encoded_mash = np.load(encoded_mash_file_path)
            image_embedding = np.load(image_embedding_file_path, allow_pickle=True).item()

        encoded_mash_tensor = torch.from_numpy(encoded_mash).float()

        data['encoded_mash'] = encoded_mash_tensor

        image_embedding_tensor = {}

        for key, item in image_embedding.items():
            image_embedding_tensor[key] = torch.from_numpy(item).float()

        data['image_embedding'] = image_embedding_tensor
        return data
