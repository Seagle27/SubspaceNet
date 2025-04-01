"""Subspace-Net 
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
    * create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset, Sampler
from pathlib import Path

from src.signal_creation import Samples
from src.system_model import SystemModelParams
from src.utils import *


def create_dataset(
        samples_model: Samples,
        samples_size: int,
        save_datasets: bool = False,
        datasets_path: Path = None,
        true_doa: list = None,
        true_range: list = None,
        phase: str = None,
):
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset.
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False.
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        true_range (list, optional): Predefined ranges. Defaults to None.
        phase (str, optional): The phase of the dataset (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired dataset comprised of (X-samples, Y-labels).

    """
    time_series, labels, sources_num = [], [], []

    for _ in tqdm(range(samples_size), desc="Creating Dataset"):
        M = samples_model.params.M
        if M is None:
            M = np.random.randint(2, samples_model.params.N-1)
        # Samples model creation
        samples_model.set_doa(true_doa, M)
        if samples_model.params.field_type.lower().endswith("near"):
            samples_model.set_range(true_range, M)
        # Observations matrix creation
        X = torch.tensor(
            samples_model.samples_creation(
                noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1, source_number=M
            )[0],
            dtype=torch.complex128,
        )
        # Ground-truth creation
        Y = torch.tensor(samples_model.doa, dtype=torch.float32)
        if samples_model.params.field_type.lower().endswith("near"):
            Y1 = torch.tensor(samples_model.distances, dtype=torch.float32)
            Y = torch.cat((Y, Y1), dim=0)
        time_series.append(X)
        labels.append(Y)
        sources_num.append(M)

    generic_dataset = TimeSeriesDataset(time_series, labels, sources_num)
    if save_datasets:
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(samples_model.params, samples_size)
        torch.save(obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename)

    return generic_dataset


# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert isinstance(path, (str, Path))
    with torch.serialization.safe_globals([TimeSeriesDataset, Samples]):
        data = torch.load(path)
    return data


def load_datasets(
        system_model_params: SystemModelParams,
        samples_size: int,
        datasets_path: Path,
        is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """

    dataset_filename = f"Generic_DataSet" + set_dataset_filename(system_model_params, samples_size)
    file_name = datasets_path / f"{'train' if is_training  else 'test'}" / dataset_filename
    try:
       dataset = read_data(file_name)
       return dataset
    except Exception as e:
        raise Exception(
            f"load_datasets: Error when loading {'Training' if is_training else 'Test'} dataset doesn't exist")


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: float):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    if system_model_params.M is None:
        M = "rand"
    else:
        M = system_model_params.M
    suffix_filename = (
            f"_{system_model_params.field_type}_field_"
            f"{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}_"
            + f"bias={system_model_params.bias}"
            + ".h5"
    )
    return suffix_filename


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y, M):
        self.X = X
        self.Y = Y
        self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.Y[idx]


def collate_fn(batch):
    time_series, source_num, labels = zip(*batch)

    # Find the maximum length in this batch
    max_length = max([lb.size(0) for lb in labels])

    # Pad labels and create masks
    padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_length, dtype=torch.float32)

    for i, lb in enumerate(labels):
        length = lb.size(0)
        if source_num[i] != length:
            # this is a near field dataset
            angles, distances = torch.split(lb, source_num[i], dim=0)
            lb = torch.cat((angles, torch.zeros(max_length // 2 - source_num[i], dtype=torch.float32)))
            lb = torch.cat((lb, distances, torch.zeros(max_length // 2 - source_num[i], dtype=torch.float32)))
            mask = torch.zeros(max_length, dtype=torch.float32)
            mask[: length // 2] = 1
            mask[max_length // 2: max_length // 2 + length // 2] = 1
        else:
            lb = torch.cat((lb, torch.zeros(max_length - length, dtype=torch.long)))
            mask = torch.zeros(max_length, dtype=torch.float32)
            mask[:length] = 1
        padded_labels[i] = lb
        masks[i] = mask

    # Stack labels
    time_series = torch.stack(time_series).squeeze()
    sources_num = torch.tensor(source_num)


    return time_series, sources_num, padded_labels, masks


class SameLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batches = self._create_batches()

    def _create_batches(self):
        length_to_indices = {}
        for idx, (_, source_num, _) in enumerate(self.data_source):
            if source_num not in length_to_indices:
                length_to_indices[source_num] = []
            length_to_indices[source_num].append(idx)
        # check that there is not bais in the labels
        max_length = 0
        min_length = np.inf
        for indices in length_to_indices.values():
            if len(indices) > max_length:
                max_length = len(indices)
            if len(indices) < min_length:
                min_length = len(indices)
        if max_length * 0.4 > min_length:
            # raise ValueError("SameLengthBatchSampler: There is a bias in the labels")
            warnings.warn("SameLengthBatchSampler: There is a bias in the labels")
            print(f"max_length: {max_length}, min_length: {min_length}")

        batches = []
        for indices in length_to_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        if self.shuffle:
            # Shuffle the batches
            np.random.shuffle(batches)
            # shuffle the indices in each batch
            for batch in batches:
                np.random.shuffle(batch)
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def get_data_source_length(self):
        return len(self.data_source)

    def get_max_batch_length(self):
        return max([len(batch) for batch in self.batches])
