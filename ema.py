import torch
import torch.nn as nn
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


class EMA(nn.Module):
    def __init__(self, model, decay=0.995):
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        self.params = [p.data for p in self.model.parameters() if p.requires_grad]
        self.shadow_params = [
            p.data for p in self.shadow.parameters() if p.requires_grad
        ]
        self.backup = []

    def update(self):
        decay = self.decay
        for param, shadow_param in zip(self.params, self.shadow_params):
            shadow_param.copy_(shadow_param * decay + (1 - decay) * param)

    def apply_shadow(self):
        self.backup = [p.clone() for p in self.params]
        for param, shadow_param in zip(self.params, self.shadow_params):
            param.data.copy_(shadow_param)

    def restore(self):
        for param, backup in zip(self.params, self.backup):
            param.data.copy_(backup)
        self.backup = []
