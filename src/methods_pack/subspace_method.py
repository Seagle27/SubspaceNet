import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import *
from src.system_model import SystemModel


class SubspaceMethod(nn.Module):
    """

    """

    def __init__(self, system_model: SystemModel):
        super(SubspaceMethod, self).__init__()
        self.system_model = system_model
        self.eigen_threshold = nn.Parameter(torch.tensor(.5, requires_grad=False))
        self.normalized_eigenvals = None

    def subspace_separation(self,
                            covariance: torch.Tensor,
                            number_of_sources: torch.tensor = None) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.tensor):
        """

        Args:
            covariance:
            number_of_sources:

        Returns:
            the signal ana noise subspaces, both as torch.Tensor().
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        sorted_idx = torch.argsort(torch.real(eigenvalues), descending=True)
        sorted_eigvectors = torch.gather(eigenvectors, 2,
                                         sorted_idx.unsqueeze(-1).expand(-1, -1, covariance.shape[-1]).transpose(1, 2))
        # number of sources estimation
        real_sorted_eigenvals = torch.gather(torch.real(eigenvalues), 1, sorted_idx)
        self.normalized_eigen = real_sorted_eigenvals / real_sorted_eigenvals[:, 0][:, None]
        source_estimation = torch.linalg.norm(
            nn.functional.relu(
                self.normalized_eigen - self.__get_eigen_threshold() * torch.ones_like(self.normalized_eigen)),
            dim=1, ord=0).to(torch.int)
        if number_of_sources is None:
            warnings.warn("Number of sources is not defined, using the number of sources estimation.")
        # if source_estimation == sorted_eigvectors.shape[2]:
        #     source_estimation -= 1
            signal_subspace = sorted_eigvectors[:, :, :source_estimation]
            noise_subspace = sorted_eigvectors[:, :, source_estimation:]
        else:
            signal_subspace = sorted_eigvectors[:, :, :number_of_sources]
            noise_subspace = sorted_eigvectors[:, :, number_of_sources:]

        if self.training:
            l_eig = self.eigen_regularization(number_of_sources)
        else:
            l_eig = None

        return signal_subspace.to(device), noise_subspace.to(device), source_estimation, l_eig

    def eigen_regularization(self, number_of_sources: int):
        """

        Args:
            normalized_eigenvalues:
            number_of_sources:

        Returns:

        """
        l_eig = (self.normalized_eigen[:, number_of_sources - 1] - self.__get_eigen_threshold(level="high")) * \
                (self.normalized_eigen[:, number_of_sources] - self.__get_eigen_threshold(level="low"))
        # l_eig = -(self.normalized_eigen[:, number_of_sources - 1] - self.__get_eigen_threshold(level="high")) + \
                # (self.normalized_eigen[:, number_of_sources] - self.__get_eigen_threshold(level="low"))
        l_eig = torch.sum(l_eig)
        # eigen_regularization = nn.functional.elu(eigen_regularization, alpha=1.0)
        return l_eig

    def __get_eigen_threshold(self, level: str = None):
        if self.training:
            if level is None:
                return self.eigen_threshold
            elif level == "high":
                return self.eigen_threshold + 0.0
            elif level == "low":
                return self.eigen_threshold - 0.0
        else:
            return self.eigen_threshold + 0.1

    def pre_processing(self, x: torch.Tensor, mode: str = "sample"):
        if mode == "sample":
            Rx = self.__sample_covariance(x)
        elif mode == "sps":
            Rx = self.__spatial_smoothing_covariance(x)
        elif mode == "sparse":
            Rx = self.__virtual_array_covariance(x)
        elif mode == "sparse_sps":
            Rx = self.__virtual_array_covariance(x)
            Rx = self.__spatial_smoothing_coarray_cov(Rx)

        else:
            raise ValueError(
                f"SubspaceMethod.pre_processing: method {mode} is not recognized for covariance calculation.")

        return Rx

    def __sample_covariance(self, x: torch.Tensor):
        """
        Calculates the sample covariance matrix.

        Args:
        -----
            X (torch.Tensor): Input samples matrix.

        Returns:
        --------
            Rx (torch.Tensor): Covariance matrix.
        """
        if x.dim() == 2:
            x = x[None, :, :]
        batch_size, sensor_number, samples_number = x.shape
        Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
        return Rx

    def __virtual_array_covariance(self, x: torch.Tensor):
        """
        Calculates the virtual array covariance matrix, based on the paper: "Remarks on the Spatial Smoothing Step in
        Coarray MUSIC"

         Parameters
         ----------
          X (torch.Tensor): Input samples matrix.
          system_model (SystemModel): settings of the system model

        Returns
        -------
        Rx (torch.Tensor): virtual array's covariance matrix

        """
        R_real_array = self.__sample_covariance(x)

        L = len(self.system_model.virtual_array)
        Rx = torch.zeros(R_real_array.shape[0], L, L, dtype=torch.complex128)
        differences_array = self.system_model.array[:, None] - self.system_model.array[None, :]
        x_s_diff = torch.zeros(x.shape[0], 2*L - 1, dtype=torch.complex128)  # x.shape[0] = batch size
        max_sensor = np.max(self.system_model.virtual_array)

        for i, lag in enumerate(range(-max_sensor, max_sensor + 1)):
            pairs = torch.from_numpy(differences_array) == lag
            if pairs.any():
                x_s_diff[:, i] = torch.mean(R_real_array[:, pairs], dim=1)

        for j in range(L):
            start_idx = L - 1 - j
            Rx[:, :, j] = x_s_diff[:, start_idx:start_idx + L]

        return Rx

    def __spatial_smoothing_covariance(self, x: torch.Tensor, sub_array_size=None):
        """
        Calculates the covariance matrix using forward–backward spatial smoothing technique.

        Args:
        -----
            x (torch.Tensor): Input samples matrix with shape
                              (batch_size, sensor_number, samples_number).

        Returns:
        --------
            Rx_smoothed (torch.Tensor): Smoothed covariance matrix.
        """

        # Ensure x has three dimensions (batch, sensors, samples)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, sensor_number, samples_number = x.shape

        # Define subarray size and the number of overlapping subarrays
        if sub_array_size is None:
            sub_array_size = sensor_number // 2 + 1

        number_of_sub_arrays = sensor_number - sub_array_size + 1

        # Initialize the smoothed covariance matrix
        Rx_smoothed = torch.zeros(batch_size, sub_array_size, sub_array_size,
                                  dtype=torch.complex128, device=x.device)

        for j in range(number_of_sub_arrays):
            # Extract the j-th subarray
            x_sub = x[:, j:j + sub_array_size, :]

            # Forward covariance calculation
            cov_forward = torch.einsum("bmt, btl -> bml", x_sub,
                                       torch.conj(x_sub).transpose(1, 2)) / (samples_number - 1)

            # backward processing: take the complex conjugate before flipping
            x_sub_back = torch.flip(torch.conj(x_sub), dims=[1])
            cov_backward = torch.einsum("bmt, btl -> bml", x_sub_back,
                                        torch.conj(x_sub_back).transpose(1, 2)) / (samples_number - 1)

            # Average the forward and backward covariances for this subarray
            cov_fb = 0.5 * (cov_forward + cov_backward)

            # Aggregate over all subarrays
            Rx_smoothed += cov_fb / number_of_sub_arrays

        return Rx_smoothed

    @staticmethod
    def __spatial_smoothing_coarray_cov(R_coarray: torch.Tensor, sub_array_size: int = None) -> torch.Tensor:
        """
        Perform forward–backward spatial smoothing on the coarray covariance matrix R_coarray.

        Parameters
        ----------
        R_coarray : torch.Tensor
            The coarray covariance, shape = [batch_size, L, L].
            - L is the size of the virtual ULA in the coarray domain.
        sub_array_size : int, optional
            The length of each sub-subarray in the coarray domain.
            If None, it will default to L//2 + 1 (typical choice).

        Returns
        -------
        R_smoothed : torch.Tensor
            The smoothed covariance, shape = [batch_size, sub_array_size, sub_array_size].
            This can then be used in MUSIC/ESPRIT for coherent sources.
        """
        # R_coarray has shape [batch_size, L, L]
        batch_size, L, _ = R_coarray.shape

        # Default subarray size: L//2 + 1 if not provided
        if sub_array_size is None:
            sub_array_size = L // 2 + 1

        # Number of forward subarrays
        number_of_sub_arrays = L - sub_array_size + 1
        if number_of_sub_arrays <= 0:
            raise ValueError("sub_array_size is too large for the given L.")

        # Initialize the smoothed covariance accumulator
        R_smoothed = torch.zeros(
            (batch_size, sub_array_size, sub_array_size),
            dtype=R_coarray.dtype, device=R_coarray.device
        )

        for start_idx in range(number_of_sub_arrays):
            # Extract the forward subarray covariance block
            sub_cov = R_coarray[:, start_idx:start_idx + sub_array_size,
                      start_idx:start_idx + sub_array_size]
            # Compute the backward covariance block by taking the conjugate and flipping along both dimensions
            sub_cov_back = torch.flip(torch.conj(sub_cov), dims=[1, 2])
            # Average the forward and backward covariance blocks
            R_smoothed += 0.5 * (sub_cov + sub_cov_back)

        # Final averaging over the number of subarrays
        R_smoothed /= number_of_sub_arrays

        return R_smoothed

    def plot_eigen_spectrum(self, batch_idx: int=0):
        """
        Plot the eigenvalues spectrum.

        Args:
        -----
            batch_idx (int): Index of the batch to plot.
        """
        plt.figure()
        plt.stem(self.normalized_eigen[batch_idx].cpu().detach().numpy(), label="Normalized Eigenvalues")
        # ADD threshold line
        plt.axhline(y=self.__get_eigen_threshold(), color='r', linestyle='--', label="Threshold")
        plt.title("Eigenvalues Spectrum")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.grid()
        plt.show()
