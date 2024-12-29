import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T
from torchvision.transforms import v2
import numpy as np
import tqdm
import os
import concurrent.futures
from scipy.spatial.distance import cdist, pdist, squareform
import networkx
import faiss

default_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = 'datasets/SF_XL/train/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

def init_pool(cluster_descriptors_dict_input, city_df_input):
    global cluster_descriptors_dict, city_df
    cluster_descriptors_dict = cluster_descriptors_dict_input
    city_df = city_df_input

def load_city_df():
    # Load cities
    city_df = {}
    cluster_info = np.load("cache/datasets/SF_XL/clusters.npy", allow_pickle=True).flat[0] # From preprocess_dataset_cluster.sh
    data = [(key, cluster) for cluster, keys in cluster_info.items() for key in keys]
    df = pd.DataFrame(data, columns=["key", "class"])
    easting = df["key"].apply(lambda x: float(x.split('/')[-1].split('@')[1]))
    northing = df["key"].apply(lambda x: float(x.split('/')[-1].split('@')[2]))
    group_id = df["key"].apply(lambda x: x.split('/')[-2])
    df.insert(1, 'easting', easting)
    df.insert(1, 'northing', northing)
    df.insert(1, 'group_id', group_id)
    for city in df['group_id'].unique():
        # Database
        subset_df = df.loc[df['group_id'] == city]
        subset_df = subset_df.reset_index()
        cluster = subset_df.groupby('class').ngroup()
        subset_df.insert(1, 'unique_cluster', cluster)
        city_df[city] = subset_df
        average_count = subset_df.groupby('unique_cluster').size().mean()
        print(f"Average number of samples for each cluster for city {city}: {average_count}")

    return city_df

def compute_cluster_descriptors(city_df, model, batch_size=32):

    class SFXLDataset(torch.utils.data.Dataset):
        def __init__(self, rows, city_path):
            self.rows = rows
            self.city_path = city_path

            self.valid_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return len(self.rows)
        
        def __getitem__(self, idx):
            row = self.rows.iloc[idx]
            path = Path(BASE_PATH) / row['key']
            try:
                img = torchvision.io.read_image(str(path))
            except:
                print(f'Image {path} could not be loaded')
                img = Image.new('RGB', (322, 322))
            img = self.valid_transform(img)
            return img, row['unique_cluster']

    
    cluster_descriptors_dict = {}
    model.eval()
    for city, df in tqdm.tqdm(city_df.items(), desc='Computing cluster descriptors'):

        # Create dataloader with one sample per cluster
        msls = SFXLDataset(df.groupby('unique_cluster').sample(1), city)
        dataloader = torch.utils.data.DataLoader(
            dataset=msls, 
            batch_size=batch_size,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
            shuffle=False
        )

        descriptor_size = None
        res = faiss.StandardGpuResources()

        # Compute descriptors for each cluster
        invalid_clusters = []
        print("Computing descriptors for city", city)
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                img, clusters = batch
                img = img.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if model.backbone.domain_prompt != "none":
                        descriptors, domain_desc = model(img)
                    else:
                        descriptors = model(img)
                if descriptor_size is None:
                    descriptor_size = descriptors.shape[1]
                    cluster_descriptors = [[] for _ in range(df.unique_cluster.max() + 1)]
                for i in range(len(descriptors)):
                    cluster_descriptors[clusters[i]] = descriptors[i].cpu()
            for i in range(len(cluster_descriptors)):
                if cluster_descriptors[i] == []:
                    cluster_descriptors[i] = torch.ones(descriptor_size) * 1e6
                    invalid_clusters.append(i)
            cluster_descriptors = torch.stack(cluster_descriptors, dim=0)
        print("Sorting cluster indices for city", city)
        index = faiss.IndexFlatL2(descriptor_size)
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(cluster_descriptors)
        _, I = index.search(cluster_descriptors, 64)
        for i in invalid_clusters:
            I[i] = torch.ones_like(I[i]) * -1
        cluster_descriptors_dict[city] = I.cpu().numpy()
        print("Finish for city", city)
    model.train()

    return cluster_descriptors_dict


def create_dataset_part(
        num_batches=100,
        batch_size=60,
        num_images_per_place=4,
        sampled_similar_places=15,
        same_place_threshold=20.0,
    ):

    import os
    import time
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    images = np.zeros((num_batches, batch_size, num_images_per_place), dtype=object)
    # Pre-compute city weights based on dataset size
    city_weights = {city: len(df) for city, df in city_df.items()}
    total_samples = sum(city_weights.values())
    city_weights = {city: count/total_samples for city, count in city_weights.items()}
    cities = list(city_weights.keys())
    weights = [city_weights[city] for city in cities]

    for i in tqdm.tqdm(range(num_batches)):
        batch_idx = 0
        cities_this_batch = []
        while batch_idx < batch_size:
            # Sample city based on size-weighted probability
            city = np.random.choice(cities, p=weights)
            # Don't sample already done in this batch
            while city in cities_this_batch:
                city = np.random.choice(cities, p=weights)
            cities_this_batch.append(city)
            
            df = city_df[city]
            topk = cluster_descriptors_dict[city]
            
            # Sample valid clusters only
            valid_clusters = df.unique_cluster.unique()
            valid_clusters = valid_clusters[topk[valid_clusters][:, 0] != -1]
            if len(valid_clusters) == 0:
                continue

            place_id = np.random.choice(valid_clusters)
            topk_subset = topk[place_id]
            other_places = topk_subset[:sampled_similar_places]
            other_places = other_places[other_places != -1]  # Remove invalid

            # Get subset of dataframe for selected clusters
            df_subset = df[df['unique_cluster'].isin(other_places)]

            # Create adjacency matrix from UTM coordinates
            utms = squareform(pdist(df_subset[['easting', 'northing']].values)) < same_place_threshold
            while batch_idx < batch_size:
                degrees = np.sum(utms, axis=0)
                # if there is no node with enough degree, return None
                if np.sum(degrees >= num_images_per_place) == 0:
                    break
                
                # Get indices of nodes with enough degree
                high_degree_indices = np.where(degrees >= num_images_per_place)[0]
                
                # Remove low-degree nodes from adjacency matrix
                mask = np.zeros(len(utms), dtype=bool)
                mask[high_degree_indices] = True
                utms_filtered = utms[mask][:, mask]
                # Find cliques in filtered graph
                for c in networkx.find_cliques(networkx.Graph(utms_filtered)):
                    if len(c) >= num_images_per_place:
                        # Map filtered indices back to original indices
                        original_indices = np.arange(len(utms))[mask][c]
                        clique = np.random.choice(original_indices, num_images_per_place, replace=False)
                        break
                else:
                    break
                
                neighbors = np.unique(np.where(utms[clique, :])[1])

                # Append place to batch
                rows = df_subset.iloc[list(clique)]
                images[i, batch_idx] = rows['key'].values
                batch_idx += 1

                # Remove selected place and its neighbors from the graph
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
            prefetch_factor=1,
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
        self.prefetch_factor = prefetch_factor

        self.create_dataset(
            num_batches=num_batches,
            num_processes=num_processes,
            batch_size=batch_size,
            num_images_per_place=num_images_per_place,
            sampled_similar_places=sampled_similar_places,
            same_place_threshold=same_place_threshold,
            prefetch_factor=prefetch_factor,
        )
        
        
    def __getitem__(self, index):
        
        batch_idx = index // self.batch_size
        img_idx = index % self.batch_size
         
        imgs = []
        for index in range(self.num_images_per_place):
            img_name = self.data[batch_idx, img_idx, index]
            img_path = self.base_path + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)
            img = v2.ToDtype(torch.float16, scale=True)(img)
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
            return torchvision.io.read_image(path, mode=torchvision.io.image.ImageReadMode.RGB)
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))
        

    def reload(self, model=None, recompute=False):
        if recompute:
            self.create_dataset(
                model=model,
                num_batches=self.num_batches,
                num_processes=self.num_processes,
                batch_size=self.batch_size,
                num_images_per_place=self.num_images_per_place,
                sampled_similar_places=self.sampled_similar_places,
                same_place_threshold=self.same_place_threshold,
                prefetch_factor=self.prefetch_factor,
                recompute=recompute,
            )
        else:
            self.data = self.data[np.random.permutation(self.data.shape[0])]
            if self.prefetch_factor > 1:
                for i in range(self.data.shape[0]):
                    for j in  range(self.data[i].shape[0]):
                        self.data[i][j] = self.data[i][j][np.random.permutation(self.data[i][j].shape[0])]
        

    def create_dataset(
        self,
        model=None,
        num_batches=1000,
        num_processes=4,
        batch_size=30,
        num_images_per_place=4,
        sampled_similar_places=15,
        same_place_threshold=20.0,
        prefetch_factor=1,
        recompute=False,
    ):

        city_df = load_city_df()

        cluster_descriptors_path = 'cache/datasets/SF_XL/cluster_descriptors.npy'

        # Compute cluster descriptors if model is provided
        if model is not None:
            cluster_descriptors_dict = compute_cluster_descriptors(city_df, model)
            if not recompute: # recompute does not save
                np.save(cluster_descriptors_path, cluster_descriptors_dict)
        elif os.path.isfile(cluster_descriptors_path):
            cluster_descriptors_dict = np.load(cluster_descriptors_path, allow_pickle=True).item()
        else:
            print('Model must be provided to compute cluster descriptors')
            print('- Computing descriptors using torch.hub DINOv2 SALAD')
            model = torch.hub.load("serizba/salad", "dinov2_salad").cuda()
            cluster_descriptors_dict = compute_cluster_descriptors(city_df, model)
            if not recompute: # recompute does not save
                np.save(cluster_descriptors_path, cluster_descriptors_dict)

        # Create dataset in parallel
        all_images = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes, initializer=init_pool, initargs=(cluster_descriptors_dict, city_df)) as executor:
            tasks = [executor.submit(
                create_dataset_part,
                num_batches // num_processes,
                batch_size,
                num_images_per_place * prefetch_factor,
                sampled_similar_places,
                same_place_threshold,
            ) for _ in range(num_processes)]
            
            # Collect results in all_images
            for task in concurrent.futures.as_completed(tasks):
                all_images.append(task.result())

        self.data = np.concatenate(all_images)