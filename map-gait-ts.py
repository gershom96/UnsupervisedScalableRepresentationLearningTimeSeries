import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import wandb

from networks.causal_cnn import CausalCNNEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import utils
import losses

class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        # sample is expected to be of shape (18, 100)
        # mean and std are broadcasted along the time dimension
        return (sample - self.mean[:, None]) / self.std[:, None]

class LazyTimeSeriesDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        
        # Open the file in read-only mode
        with h5py.File(h5_file, 'r') as f:
            self.terrains = list(f.keys())
            self.data_lengths = [f[terrain].shape[0] for terrain in self.terrains]
            self.total_samples = sum(self.data_lengths)
            self.cumulative_lengths = np.cumsum([0] + self.data_lengths)
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Determine which terrain the index corresponds to
        terrain_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        terrain_start_idx = self.cumulative_lengths[terrain_idx]
        relative_idx = idx - terrain_start_idx
        
        with h5py.File(self.h5_file, 'r') as f:
            terrain = self.terrains[terrain_idx]
            sample = f[terrain][relative_idx]
            label = terrain_idx  # Using the terrain index as the label
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.tensor(sample, dtype=torch.float32), label

    def compute_class_weights(self):
        """
        Computes the weights for each class based on the inverse of the class frequency.
        """
        # The number of samples per class is already stored in self.data_lengths
        class_counts = torch.tensor(self.data_lengths, dtype=torch.float)
        
        # Compute the class weights as the inverse of class frequency
        class_weights = 1.0 / class_counts
        
        # Create a weight for each sample
        sample_weights = []
        for terrain_idx, count in enumerate(self.data_lengths):
            sample_weights.extend([class_weights[terrain_idx]] * count)
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        
        return sample_weights
    
class TimeSeriesDataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        
        with h5py.File(h5_file, 'r') as f:
            self.terrains = list(f.keys())
            self.data = []
            self.labels = []
            
            for idx, terrain in enumerate(self.terrains):
                terrain_data = f[terrain][:]
                self.data.append(terrain_data)
                self.labels.extend([idx] * len(terrain_data))
            
            self.data = np.concatenate(self.data, axis=0)  # Combine all terrain data
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return torch.tensor(sample, dtype=torch.float32), label
    
    def compute_class_weights(self):
        class_counts = torch.bincount(self.labels)
        class_weights = 1.0 / class_counts.float()
        
        sample_weights = class_weights[self.labels]
        
        return sample_weights

    
class CausalCNNEncoderTrainer:
    def __init__(self, compared_length=None, nb_random_samples=20, negative_penalty=1,
                 batch_size=100, nb_steps=2000, lr=1e-4, early_stopping=None, 
                 in_channels=18, out_channels=320, channels=40, depth=10, reduced_size=160, 
                 cuda=True, gpu=0, kernel_size=3):
        
        self.architecture = 'CausalCNN'
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.early_stopping = early_stopping
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.encoder = self.__create_encoder(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)
        
        self.loss = losses.triplet_loss.TripletLoss(compared_length, nb_random_samples, negative_penalty)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.cuda = cuda
        self.gpu = gpu

        if self.cuda:
            self.encoder.cuda(gpu)

        # Initialize wandb
        wandb.init(project="causal-cnn-encoder", config={
            "architecture": "CausalCNN",
            "epochs": nb_steps,
            "batch_size": batch_size,
            "learning_rate": lr,
            "channels": channels,
            "depth": depth,
            "reduced_size": reduced_size,
            "out_channels": out_channels,
            "kernel_size": kernel_size
        })

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = CausalCNNEncoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size)
        encoder.float()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def load_encoder(self, prefix_file):
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def save_encoder(self, prefix_file):
        torch.save(self.encoder.state_dict(), prefix_file + '_' + self.architecture + '_encoder.pth')

    def fit_encoder(self, data_loader, full_train_data, save_memory=False, verbose=False):
        full_train_tensor = torch.tensor(full_train_data, dtype=torch.float32)
        if self.cuda:
            full_train_tensor = full_train_tensor.cuda(self.gpu)

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            epoch_loss = 0.0  # To track loss for the entire epoch
            if verbose:
                print(f'Epoch: {epochs + 1}/{self.nb_steps // len(data_loader)}')
            
            for batch, _ in data_loader:  # The dataloader returns batches with (sample, label) tuples
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()

                loss = self.loss(batch, self.encoder, full_train_tensor, save_memory=save_memory)
                loss.backward()
                self.optimizer.step()
                print(f'Epoch {epochs + 1}, Step {i + 1}: Loss = {loss:.4f}')
                epoch_loss += loss.item()  # Accumulate loss for this epoch
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1

            # Log the loss for this epoch
            avg_epoch_loss = epoch_loss / len(data_loader)
            print(f'Epoch {epochs}: Loss = {avg_epoch_loss:.4f}')
            wandb.log({"epoch": epochs, "loss": avg_epoch_loss})

        return self.encoder
    
    def fit_encoder_1(self, data_loader, save_memory=False, verbose=False):
        # full_train_tensor = torch.tensor(full_train_data, dtype=torch.float32)
        # if self.cuda:
        #     full_train_tensor = full_train_tensor.cuda(self.gpu)

        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs

        # Encoder training
        while i < self.nb_steps:
            epoch_loss = 0.0  # To track loss for the entire epoch
            if verbose:
                print(f'Epoch: {epochs + 1}/{self.nb_steps // len(data_loader)}')
            
            for batch, _ in data_loader:  # The dataloader returns batches with (sample, label) tuples
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()

                loss = self.loss(batch, self.encoder, batch, save_memory=save_memory)
                loss.backward()
                self.optimizer.step()
                print(f'Epoch {epochs + 1}, Step {i + 1}: Loss = {loss:.4f}')
                epoch_loss += loss.item()  # Accumulate loss for this epoch
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1

            # Log the loss for this epoch
            avg_epoch_loss = epoch_loss / len(data_loader)
            print(f'Epoch {epochs}: Loss = {avg_epoch_loss:.4f}')
            wandb.log({"epoch": epochs, "loss": avg_epoch_loss})

        return self.encoder

def main():

    h5_path = '/home/gershom/Documents/GAMMA/UnsupervisedScalableRepresentationLearningTimeSeries/Dataset/all_timeseries_by_terrain_train.h5'
    
    mean = np.array([-9.7528e-03,  7.1942e-02, -9.8050e+00, -1.3312e-03, -4.1957e-04, -2.5726e-02, 
                     -1.9685e+00,  1.6953e+01, -4.1611e+00,  1.8878e+01, -1.0623e+00,  1.6011e+01, 
                     -3.7153e+00,  1.9238e+01, -1.3624e+01, -1.3926e+01,  1.3605e+01,  1.2424e+01])
    
    std = np.array([ 1.5537,  0.9568,  3.3148,  0.1399,  0.1181,  0.3017, 
                    13.1651, 17.3949, 13.9258, 19.9675, 13.2737, 17.1195, 
                    13.4313, 19.9113, 20.9347, 20.9335, 20.7401, 19.3905])
    
    transform = NormalizeTransform(mean=mean, std=std)


    # dataset = TimeSeriesDataset(h5_path, transform=transform)
    # sample_weights = dataset.compute_class_weights()

    dataset = LazyTimeSeriesDataset(h5_path)
    sample_weights = dataset.compute_class_weights()

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    dataloader = DataLoader(
        dataset, 
        batch_size=200, 
        sampler=sampler, 
        num_workers=4,   
        pin_memory=True, 
        drop_last=True   
    )
    
    trainer = CausalCNNEncoderTrainer()
    
    # trainer.fit_encoder(dataloader, dataset.data)
    trainer.fit_encoder_1(dataloader)

if __name__ == "__main__":
    main()
