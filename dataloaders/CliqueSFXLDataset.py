import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import tqdm
import os

import concurrent.futures
from scipy.spatial.distance import cdist, pdist, squareform
import networkx

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = 'datasets/SF_XL/train/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

def load_city_df():
    # Load cities
    city_df = {}
    cluster_info = np.load("cache/datasets/SF_XL/clusters.npy", allow_pickle=True).flat[0]
    data = [(key, cluster) for cluster, keys in cluster_info.items() for key in keys]
    df = pd.DataFrame(data, columns=["key", "class"])
    easting = df["key"].apply(lambda x: float(x.split('/')[-1].split('@')[1]))
    northing = df["key"].apply(lambda x: float(x.split('/')[-1].split('@')[2]))
    group_id = df["key"].apply(lambda x: x.split('/')[-2])
    df.insert(1, 'easting', easting)
    df.insert(1, 'northing', northing)
    df.insert(1, 'group_id', group_id)
    cluster = df.groupby('class').ngroup()
    df.insert(1, 'unique_cluster', cluster)
    for city in df['group_id'].unique():
        # Database
        single_df = df[df['group_id'] == city]
        city_df[city] = df.reset_index()
        average_count = df.groupby('unique_cluster').size().mean()
        print(f"Average number of samples for each cluster for city {city}: {average_count}")

    return city_df

def compute_cluster_descriptors(city_df, model, descriptor_size=8192 + 256, batch_size=64):

    class SFXLDataset(torch.utils.data.Dataset):
        def __init__(self, rows, city_path):
            self.rows = rows
            self.city_path = city_path

            self.valid_transform = T.Compose([
                T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.rows)
        
        def __getitem__(self, idx):
            row = self.rows.iloc[idx]
            path = Path(BASE_PATH) / row['key']
            try:
                img = Image.open(path)
            except:
                print(f'Image {path} could not be loaded')
                img = Image.new('RGB', (322, 322))
            img = self.valid_transform(img)
            return img, row['unique_cluster']

    
    cluster_descriptors_dict = {}
    for city, df in tqdm.tqdm(city_df.items(), desc='Computing cluster descriptors'):

        # Create dataloader with one sample per cluster
        msls = SFXLDataset(df.groupby('unique_cluster').sample(1), city)
        dataloader = torch.utils.data.DataLoader(
            dataset=msls, 
            batch_size=batch_size,
            num_workers=12,
            drop_last=False,
            pin_memory=True,
            shuffle=False
        )

        cluster_descriptors = torch.zeros((df.unique_cluster.max() + 1, descriptor_size)).cuda()

        # Compute descriptors for each cluster
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                img, clusters = batch
                img = img.cuda()
                descriptors = model(img)
                cluster_descriptors[clusters] = descriptors

        cluster_descriptors_dict[city] = cluster_descriptors.cpu().numpy()

    return cluster_descriptors_dict


def create_dataset_part(
        cluster_descriptors_dict,
        city_df,
        num_batches=100,
        batch_size=60,
        num_images_per_place=4,
        sampled_similar_places=15,
        same_place_threshold=20.0,
        only_top_k=False,
    ):

    import os
    import time
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    images = np.zeros((num_batches, batch_size, num_images_per_place), dtype=object)

    for i in tqdm.tqdm(range(num_batches)):

        cities_this_batch = []

        batch_idx = 0
        while batch_idx < batch_size:

            cities_to_sample = [c for c in cluster_descriptors_dict.keys()]
            num_clusters = np.array([d.shape[0] for c, d in cluster_descriptors_dict.items()])

            city = np.random.choice(cities_to_sample, p=num_clusters/num_clusters.sum())

            # Don't sample already done in this batch
            while city in cities_this_batch:
                city = np.random.choice(cities_to_sample, p=num_clusters/num_clusters.sum())
            cities_this_batch.append(city)


            df = city_df[city]
            descriptor = cluster_descriptors_dict[city]
            
            # Sample a random cluster
            place_id = np.random.choice(df.unique_cluster.unique())

            # Compute similarity between the selected cluster and all the others
            distances = cdist(descriptor[place_id, None, :], descriptor)[0]
            # Normalize distances as probabilities (where min distance is max probability)
            if only_top_k:
                distances = np.delete(distances, place_id)
                other_places = np.delete(np.arange(df.unique_cluster.max() + 1), place_id)
                
                # Sample similar places
                topk = np.argsort(distances)[:sampled_similar_places]
                other_places = other_places[topk]
            else:
                distances[distances != 0] = distances.max() - distances[distances != 0]
                distances = distances / distances.sum()

                # Sample similar places
                other_places = np.random.choice(np.arange(df.unique_cluster.max() + 1), size=sampled_similar_places, p=distances, replace=False)
            other_places = np.concatenate([np.array([place_id]), other_places])

            df = df[df['unique_cluster'].isin(other_places)]

            # Create adjacency matrix from UTM coordinates (two places are connected if they are closer than same_place_threshold)
            utms = squareform(pdist(df[['easting', 'northing']].values)) < same_place_threshold

            while batch_idx < batch_size:

                # Find a clique of at least num_images_per_place
                for c in networkx.find_cliques(networkx.Graph(utms)):
                    if len(c) >= num_images_per_place:
                        clique = np.random.choice(c, num_images_per_place, replace=False)
                        break
                else:
                    break

                neighbors = np.unique(np.where(utms[clique, :])[1])

                # Append place to batch
                rows = df.iloc[list(clique)]
                images[i, batch_idx] = np.char.add(np.char.add(np.where(rows['query'].values, f'{city}/query/images/', f'{city}/database/images/').astype('<U100'), rows['key'].values.astype('<U100')), '.jpg')
                batch_idx += 1

                # Remove selected place and its neighbors from the graph
                # (just removing the edges is enough)
                utms[:, clique] = False
                utms[clique, :] = False
                utms[neighbors, :] = False
                utms[:, neighbors] = False

    return images


class CliqueSFXLDataset(Dataset):
    def __init__(
            self,
            split="train",
            transform=default_transform,
            base_path=BASE_PATH,
            num_batches=4000,
            num_processes=4,
            batch_size=30,
            num_images_per_place=4,
            sampled_similar_places=15,
            same_place_threshold=20.0,
            only_top_k=False,
            recompute_clusters=False,
    ):
        super(CliqueSFXLDataset, self).__init__()
        self.base_path = base_path
        self.transform = transform
        self.split = split

        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.num_images_per_place = num_images_per_place
        self.sampled_similar_places = sampled_similar_places
        self.same_place_threshold = same_place_threshold
        self.recompute_clusters = recompute_clusters
        self.only_top_k = only_top_k

        self.create_dataset(
            num_batches=num_batches,
            num_processes=num_processes,
            batch_size=batch_size,
            num_images_per_place=num_images_per_place,
            sampled_similar_places=sampled_similar_places,
            same_place_threshold=same_place_threshold,
            only_top_k=only_top_k,
        )
        
        
    def __getitem__(self, index):
        
        batch_idx = index // self.batch_size
        img_idx = index % self.batch_size
         
        imgs = []
        for img_name in self.data[batch_idx, img_idx]:
            img_path = self.base_path + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        # NOTE: contrary to image classification where __getitem__ returns only one image 
        # in GSVCities, we return a place, which is a Tesor of K images (K=self.img_per_place)
        # this will return a Tensor of shape [K, channels, height, width]. This needs to be taken into account 
        # in the Dataloader (which will yield batches of shape [BS, K, channels, height, width])
        return torch.stack(imgs), torch.tensor(img_idx).repeat(len(imgs))

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return self.batch_size * self.num_batches

    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))
        

    def reload(self, model=None):
        if self.recompute_clusters:
            self.create_dataset(
                model=model,
                num_batches=self.num_batches,
                num_processes=self.num_processes,
                batch_size=self.batch_size,
                num_images_per_place=self.num_images_per_place,
                sampled_similar_places=self.sampled_similar_places,
                same_place_threshold=self.same_place_threshold,
                only_top_k=self.only_top_k,
            )
        else:
            self.data = self.data[np.random.permutation(self.data.shape[0])]
        

    def create_dataset(
        self,
        model=None,
        num_batches=1000,
        num_processes=4,
        batch_size=30,
        num_images_per_place=4,
        sampled_similar_places=15,
        same_place_threshold=20.0,
        only_top_k=False,
    ):

        city_df = load_city_df()

        cluster_descriptors_path = 'cache/datasets/SF_XL/cluster_descriptors.npy'

        # Compute cluster descriptors if model is provided
        if model is not None:
            cluster_descriptors_dict = compute_cluster_descriptors(city_df, model)
            np.save(cluster_descriptors_path, cluster_descriptors_dict)
        elif os.path.isfile(cluster_descriptors_path):
            cluster_descriptors_dict = np.load(cluster_descriptors_path, allow_pickle=True).item()
        else:
            print('Model must be provided to compute cluster descriptors')
            print('- Computing descriptors using torch.hub DINOv2 SALAD')
            model = torch.hub.load("serizba/salad", "dinov2_salad").eval().cuda()
            cluster_descriptors_dict = compute_cluster_descriptors(city_df, model)
            np.save(cluster_descriptors_path, cluster_descriptors_dict)

        # Create dataset in parallel
        all_images = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            tasks = [executor.submit(
                create_dataset_part,
                cluster_descriptors_dict,
                city_df,
                num_batches // num_processes,
                batch_size,
                num_images_per_place,
                sampled_similar_places,
                same_place_threshold,
                only_top_k,
            ) for _ in range(num_processes)]
            
            # Collect results in all_images
            for task in concurrent.futures.as_completed(tasks):
                all_images.append(task.result())

        self.data = np.concatenate(all_images)