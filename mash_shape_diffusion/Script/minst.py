import os
from tqdm import tqdm
from torchvision import datasets

if __name__ == "__main__":
    dataset_folder_path = '/home/chli/chLi/Dataset/'
    os.makedirs(dataset_folder_path, exist_ok=True)

    train_data = datasets.MNIST(root=dataset_folder_path, train=True, download=True)
    test_data = datasets.MNIST(root=dataset_folder_path, train=False, download=True)

    print('[INFO][minst::__main__]')
    print('\t start save train data as images...')
    for i in tqdm(range(len(train_data))):
        img, label = train_data[i]

        save_folder_path = dataset_folder_path + 'MNIST/train/' + str(label) + '/'
        os.makedirs(save_folder_path, exist_ok=True)

        img.save(save_folder_path + str(i) + '.png')

    print('[INFO][minst::__main__]')
    print('\t start save test data as images...')
    for i in tqdm(range(len(test_data))):
        img, label = test_data[i]

        save_folder_path = dataset_folder_path + 'MNIST/test/' + str(label) + '/'
        os.makedirs(save_folder_path, exist_ok=True)

        img.save(save_folder_path + str(i) + '.png')
