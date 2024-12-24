import torch.nn as nn

from src.models_pack.subspacenet import SubspaceNet
from src.models_pack.parent_model import ParentModel
from src.system_model import SystemModel
from src.utils import *

from src.methods_pack.music import MUSIC
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.root_music import RootMusic, root_music


class SparseNet(SubspaceNet):

    def __init__(self, tau: int, diff_method: str = "esprit",
                 system_model: SystemModel = None):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.


        """
        super().__init__(tau, diff_method, system_model, field_type='Far')
        # self.N = len(self.system_model.virtual_array)

    # def pre_processing(self, x):
    #     """
    #     The input data is a complex signal of size [batch, N, T] and the input to the model supposed to be complex
    #      tensors of size [batch, tau, 2N_virtual, N_virtual].
    #     """
    #     batch_size = x.shape[0]
    #     x_virtual = torch.zeros(batch_size, self.N, x.shape[-1], device=device, dtype=x.dtype)
    #     indices = torch.tensor(self.system_model.array, device=device, dtype=torch.long)
    #     x_virtual[:, indices, :] = x
    #
    #     Rx_tau = torch.zeros(batch_size, self.tau, 2 * self.N, self.N, device=device)
    #     meu = torch.mean(x_virtual, dim=-1, keepdim=True).to(device)
    #     center_x = x_virtual - meu
    #     if center_x.dim() == 2:
    #         center_x = center_x[None, :, :]
    #
    #     for i in range(self.tau):
    #         x1 = center_x[:, :, :center_x.shape[-1] - i].to(torch.complex128)
    #         x2 = torch.conj(center_x[:, :, i:]).transpose(1, 2).to(torch.complex128)
    #         Rx_lag = torch.einsum("BNT, BTM -> BNM", x1, x2) / (center_x.shape[-1] - i - 1)
    #         Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), dim=1)
    #         Rx_tau[:, i, :, :] = Rx_lag
    #
    #     return Rx_tau
