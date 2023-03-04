import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class STNkd(nn.Module):
    def __init__(self, k: int = 2):
        super(STNkd, self).__init__()
        self.k = k
        self.stn_layer_1 = nn.Sequential(nn.Conv1d(k, 16, kernel_size=1),
                                         nn.BatchNorm1d(16, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))
        self.stn_layer_2 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, stride=1),
                                         nn.BatchNorm1d(64, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))

        self.stn_layer_3 = nn.Sequential(nn.Linear(in_features=64, out_features=32),
                                         nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))
        self.stn_layer_4 = nn.Sequential(nn.Linear(in_features=32, out_features=16),
                                         nn.BatchNorm1d(16, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))

        self.stn_proj = nn.Linear(16, k * k)
        nn.init.zeros_(self.stn_proj.weight)
        nn.init.zeros_(self.stn_proj.bias)

        self.eye = torch.eye(k).unsqueeze(0)

    def forward(self, x):
        x = self.stn_layer_1(x)
        x = self.stn_layer_2(x)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        x = self.stn_layer_3(x)
        x = self.stn_layer_4(x)
        x = self.stn_proj(x)
        return x.view(-1, self.k, self.k) + self.eye.to(x.device)


class PointNet(nn.Module):
    def __init__(self, num_features: int = 6, num_global_features: int = 7, out_features: int = 4):
        super(PointNet, self).__init__()
        torch.manual_seed(0)
        self.stn = STNkd(k=2)

        self.ptn_layer_1 = nn.Sequential(nn.Conv1d(num_features, 32, kernel_size=1),
                                         nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))
        self.ptn_layer_2 = nn.Sequential(nn.Conv1d(32, 128, kernel_size=1),
                                         nn.BatchNorm1d(128, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))

        self.ptn_layer_3 = nn.Sequential(nn.Linear(128 + num_global_features, 34),
                                         nn.BatchNorm1d(34, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))
        self.ptn_layer_4 = nn.Sequential(nn.Linear(34, 32),
                                         nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True))
        self.ptn_layer_5 = nn.Sequential(nn.Linear(32, 32),
                                         nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(32, out_features))

    def forward(self, x: torch.Tensor, x_global: torch.Tensor):
        xy_transformed = torch.bmm(self.stn(x[:, :2, :]), x[:, :2, :])
        x = torch.cat([xy_transformed, x[:, 2:, :]], dim=1)
        x = self.ptn_layer_1(x)
        x = self.ptn_layer_2(x)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        x = torch.cat([x, x_global], dim=1)
        x = self.ptn_layer_3(x)
        x = self.ptn_layer_4(x)
        x = self.ptn_layer_5(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PointNet(num_features=6, num_global_features=6, out_features=4)
    model.to(device)
    print(model)

    # Create input
    model_input = torch.rand(10000, 6, 20).to(device)
    model_input_global = torch.rand(10000, 6).to(device)

    start = time.time()
    output = model(model_input, model_input_global)
    end = time.time()

    print(f"Time: {end - start}")
