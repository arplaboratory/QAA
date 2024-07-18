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
from sklearn.neighbors import NearestNeighbors

import concurrent.futures
from scipy.spatial.distance import cdist, pdist, squareform
import networkx

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
NPY_ROOT = 'cache/datasets/'

def construct_df(dataset_name, split, same_place_threshold):
    city_df = {}
    db = np.load(os.path.join(NPY_ROOT, dataset_name, f"{dataset_name}_{split}_dbImages.npy"))
    db = pd.DataFrame(db, columns=["key"])
    db.insert(0, 'query', False)
    easting = db["key"].apply(lambda x: float(x.split('/')[-1].split('@')[1]))
    northing = db["key"].apply(lambda x: float(x.split('/')[-1].split('@')[2]))
    db.insert(0, 'easting', easting)
    db.insert(0, 'northing', northing)
    q = np.load(os.path.join(NPY_ROOT, dataset_name, f"{dataset_name}_{split}_qImages.npy"))
    q = pd.DataFrame(q, columns=["key"])
    q.insert(0, 'query', True)
    easting = q["key"].apply(lambda x: float(x.split('/')[-1].split('@')[1]))
    northing = q["key"].apply(lambda x: float(x.split('/')[-1].split('@')[2]))
    q.insert(0, 'easting', easting)
    q.insert(0, 'northing', northing)
    df = pd.concat([db, q], ignore_index=True)
    df.insert(0, 'unique_cluster', -1)
    if os.path.isfile("cache/datasets/" + dataset_name + "/cluster_id.npy"):
        df['unique_cluster'] = np.load("cache/datasets/" + dataset_name + "/cluster_id.npy")
    city_df[dataset_name] = df
    return city_df

def compute_cluster_descriptors(city_df, model, dataset_name, same_place_threshold, cluster_desc_threshold_percentage, descriptor_size=8192 + 256, batch_size=64):

    class DenseDataset(torch.utils.data.Dataset):
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
            path = f'{row["key"]}'
            try:
                img = Image.open(path)
            except:
                print(f'Image {path} could not be loaded')
                img = Image.new('RGB', (322, 322))
            img = self.valid_transform(img)
            return img, idx, row['unique_cluster']

    
    cluster_descriptors_dict = {}
    cluster_id = 0
    for city, df in tqdm.tqdm(city_df.items(), desc='Computing cluster descriptors'):

        if not os.path.isfile("cache/datasets/" + dataset_name + "/cluster_id.npy"):
            densedataset = DenseDataset(df, city)
            dataloader = torch.utils.data.DataLoader(
                dataset=densedataset, 
                batch_size=batch_size,
                num_workers=8,
                drop_last=False,
                pin_memory=True,
                shuffle=False
            )
            instance_descriptors = torch.zeros((len(df), descriptor_size)).cuda() # global mining or partial mining
            # Compute descriptors for each instance
            with torch.no_grad():
                for batch in tqdm.tqdm(dataloader):
                    img, idxs, _ = batch
                    img = img.cuda()
                    descriptors = model(img)
                    instance_descriptors[idxs] = descriptors

            image_utms = df[['easting', 'northing']] # Include database and query images
            # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(image_utms)
            soft_positives_per_query = knn.radius_neighbors(image_utms,
                                                            radius=same_place_threshold,
                                                            return_distance=False)
            distances_positive = []
            for i, soft_positives in enumerate(soft_positives_per_query):
                # calculate the feature distance between query and soft postive
                distance = torch.cdist(instance_descriptors[i:i+1], instance_descriptors[soft_positives])
                distances_positive.append(distance.squeeze(0))
            distances_positive = torch.cat(distances_positive)
            # Find cluster_desc_threshold_percentage * len(distances_positive) nearest neighbors
            distances_threshold = torch.topk(distances_positive, round(cluster_desc_threshold_percentage * len(distances_positive)), largest=False)[0][-1]
            print(f"Found distance threshold of {cluster_desc_threshold_percentage * 100}% of the positives: {distances_threshold}")
            available_keys = np.array(df['key'])
            print("Allocating clusters")
            while len(available_keys)>0:
                query_key = np.random.choice(available_keys, 1, replace=False)
                for l in range(3): # Extend 2 levels
                    query_idx_in_df = df[df['key'].isin(query_key)].index
                    df.loc[query_idx_in_df, 'unique_cluster'] = cluster_id
                    available_keys = np.setdiff1d(available_keys, query_key)
                    query_desc = instance_descriptors[query_idx_in_df]
                    positives_idxs_in_df = soft_positives_per_query[query_idx_in_df]
                    for i in range(len(query_desc)):
                        positives_desc = instance_descriptors[positives_idxs_in_df[i]]
                        distances = torch.cdist(query_desc, positives_desc)
                        df.loc[positives_idxs_in_df[i][(distances[i] <= distances_threshold).cpu()], 'unique_cluster'] = cluster_id
                    query_key = df[df['unique_cluster'] == cluster_id]['key'].values
                cluster_id += 1
                print("Unassigned keys: ", len(available_keys))
            print("Done")
            # Remove clusters with too few samples
            np.save("cache/datasets/" + dataset_name + "/cluster_id.npy", df['unique_cluster'].values)
            average_count = df.groupby('unique_cluster').size().mean()
            cluster_count = len(np.unique(df['unique_cluster']))
            print(f'Creating {cluster_count} unique clusters')
            print(f"Average number of samples for each cluster: {average_count}")

        densedataset = DenseDataset(df.groupby('unique_cluster').sample(1), city)
        dataloader = torch.utils.data.DataLoader(
            dataset=densedataset, 
            batch_size=batch_size,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
            shuffle=False
        )

        cluster_descriptors = torch.zeros((df.unique_cluster.max() + 1, descriptor_size)).cuda()

        # Compute descriptors for each cluster
        with torch.no_grad():
            for batch in dataloader:
                img, _, clusters = batch
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
    cities_to_sample = [c for c in cluster_descriptors_dict.keys()]
    city = cities_to_sample[0]
    df = city_df[city]
    descriptor = cluster_descriptors_dict[city]

    for i in tqdm.tqdm(range(num_batches)):

        batch_idx = 0
        valid = np.ones(len(df['unique_cluster'].unique()))
        while batch_idx < batch_size:
            
            # Sample a random cluster
            available_clusters = df['unique_cluster'].unique()[valid == 1]
            available_descriptor = descriptor[available_clusters]
            place_id_idx = np.random.choice(len(available_clusters))
            place_id = available_clusters[place_id_idx]

            # Compute similarity between the selected cluster and all the others
            distances = cdist(available_descriptor[place_id_idx, None, :], available_descriptor)[0]
            # Normalize distances as probabilities (where min distance is max probability)
            if only_top_k:
                distances = np.delete(distances, place_id_idx)
                other_places = np.delete(available_clusters, place_id_idx)

                # Sample similar places
                topk = np.argsort(distances)[:sampled_similar_places]
                other_places = other_places[topk]
            else:
                distances[distances != 0] = distances.max() - distances[distances != 0]
                distances = distances / distances.sum()

                # Sample similar places
                sample_idx = np.random.choice(len(available_clusters) - 1, sampled_similar_places, p=distances, replace=False)
                other_places = other_places[sample_idx]
            other_places = np.concatenate([np.array([place_id]), other_places])

            invalid_idx = np.where(np.isin(df['unique_cluster'].unique(), other_places, assume_unique=True))[0]
            valid[invalid_idx] = 0

            current_df = df[df['unique_cluster'].isin(other_places)]

            # Create adjacency matrix from UTM coordinates (two places are connected if they are closer than same_place_threshold)
            utms = squareform(pdist(current_df[['easting', 'northing']].values)) < same_place_threshold

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
                rows = current_df.iloc[list(clique)]
                images[i, batch_idx] = rows['key'].values
                batch_idx += 1

                # Remove selected place and its neighbors from the graph
                # (just removing the edges is enough)
                utms[:, clique] = False
                utms[clique, :] = False
                utms[neighbors, :] = False
                utms[:, neighbors] = False

    return images


class CliqueGenericDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            split="train",
            transform=default_transform,
            num_batches=4000,
            num_processes=4,
            batch_size=30,
            num_images_per_place=4,
            sampled_similar_places=15,
            same_place_threshold=20.0,
            cluster_desc_threshold_percentage=0.1,
            only_top_k=False,
            recompute_clusters=False,
            shuffle_method="global",
            prefetch_factor=1,
    ):
        super(CliqueGenericDataset, self).__init__()
        self.dataset_name = dataset_name
        self.transform = transform
        self.split = split

        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.num_images_per_place = num_images_per_place
        self.sampled_similar_places = sampled_similar_places
        self.same_place_threshold = same_place_threshold
        self.cluster_desc_threshold_percentage = cluster_desc_threshold_percentage
        self.recompute_clusters = recompute_clusters
        self.only_top_k = only_top_k
        self.shuffle_method = shuffle_method
        self.prefetch_factor = prefetch_factor

        self.create_dataset(
            num_batches=num_batches,
            num_processes=num_processes,
            batch_size=batch_size,
            num_images_per_place=num_images_per_place,
            sampled_similar_places=sampled_similar_places,
            same_place_threshold=same_place_threshold,
            cluster_desc_threshold_percentage=cluster_desc_threshold_percentage,
            only_top_k=only_top_k,
            prefetch_factor=prefetch_factor,
        )
        
        
    def __getitem__(self, index):
        
        batch_idx = index // self.batch_size
        img_idx = index % self.batch_size
         
        imgs = []
        for img_name in self.data[batch_idx, img_idx]:
            img_path = img_name
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
        return self.batch_size * 2000 # GSV has 2084 batches

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
                cluster_desc_threshold_percentage=self.cluster_desc_threshold_percentage,
                only_top_k=self.only_top_k,
                prefetch_factor=self.prefetch_factor,
            )
        elif self.shuffle_method =="global":
            self.data = self.data[np.random.permutation(self.data.shape[0])]
        elif self.shuffle_method =="batch":
            for i in range(self.data.shape[0]):
                self.data[i] = self.data[i][np.random.permutation(self.data[i].shape[0])]
            self.data = self.data[np.random.permutation(self.data.shape[0])]
        elif self.shuffle_method =="image":
            for i in range(self.data.shape[0]):
                for j in  range(self.data[i].shape[0]):
                    self.data[i][j] = self.data[i][j][np.random.permutation(self.data[i][j].shape[0])]
            for i in range(self.data.shape[0]):
                self.data[i] = self.data[i][np.random.permutation(self.data[i].shape[0])]
            self.data = self.data[np.random.permutation(self.data.shape[0])]
        else:
            raise ValueError("Invalid shuffle method")
        

    def create_dataset(
        self,
        model=None,
        num_batches=1000,
        num_processes=4,
        batch_size=30,
        num_images_per_place=4,
        sampled_similar_places=15,
        same_place_threshold=20.0,
        cluster_desc_threshold_percentage=0.1,
        only_top_k=False,
    ):

        city_df = construct_df(self.dataset_name, self.split, same_place_threshold)

        cluster_descriptors_path = f'cache/datasets/{self.dataset_name}/cluster_descriptors.npy'

        # Compute cluster descriptors if model is provided
        if model is not None:
            cluster_descriptors_dict = compute_cluster_descriptors(city_df, model, self.dataset_name, same_place_threshold, cluster_desc_threshold_percentage)
            np.save(cluster_descriptors_path, cluster_descriptors_dict)
        elif os.path.isfile(cluster_descriptors_path):
            cluster_descriptors_dict = np.load(cluster_descriptors_path, allow_pickle=True).item()
        else:
            print('Model must be provided to compute cluster descriptors')
            print('- Computing descriptors using torch.hub DINOv2 SALAD')
            model = torch.hub.load("serizba/salad", "dinov2_salad").eval().cuda()
            cluster_descriptors_dict = compute_cluster_descriptors(city_df, model, self.dataset_name, same_place_threshold, cluster_desc_threshold_percentage)
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