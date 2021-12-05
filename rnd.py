import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from rlkit.torch.networks import ConcatMlp, PretrainedCNN, CNN


class RND_Net(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.layers(x)


class RND(nn.Module):
    def __init__(self, input_width, input_height, input_channels, action_dim):
        super().__init__()
        self.action_dim = action_dim

        self.target = PretrainedCNN(
                                    input_width,
                                    input_height,
                                    input_channels,
                                    output_size=1,
                                    hidden_sizes=[128, 64],  # this is the hidden sizes of FC layers after the CNN
                                    added_fc_input_size=action_dim,  # layer used to merge image output and action input
                                    batch_norm_fc=False,
                                    init_w=1e-4,
                                    # hidden_init=nn.init.xavier_uniform_,
                                    # hidden_activation=nn.ReLU(),
                                    # output_activation=identity,
                                    output_conv_channels=False,
                                    model_architecture=models.efficientnet_b0,
                                    model_pretrained=True,
                                    model_freeze=False,
                                )
        self.moving = PretrainedCNN(
                                    input_width,
                                    input_height,
                                    input_channels,
                                    output_size=1,
                                    hidden_sizes=[128, 64],  # this is the hidden sizes of FC layers after the CNN
                                    added_fc_input_size=action_dim,  # layer used to merge image output and action input
                                    batch_norm_fc=False,
                                    init_w=1e-4,
                                    # hidden_init=nn.init.xavier_uniform_,
                                    # hidden_activation=nn.ReLU(),
                                    # output_activation=identity,
                                    output_conv_channels=False,
                                    model_architecture=models.efficientnet_b0,
                                    model_pretrained=True,
                                    model_freeze=False,
                                    )
        self.optimizer = torch.optim.Adam(self.moving.parameters(), lr=3e-4)
        self.loss = nn.MSELoss()

        self.target.eval()

    def train(self, state):
        target = self.target(state)
        moving = self.moving(state)

        loss = self.loss(moving, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def get_reward(self, state):
        target = self.target(torch.tensor(state, device="cuda", dtype=torch.float32).unsqueeze(0))
        moving = self.moving(torch.tensor(state, device="cuda", dtype=torch.float32).unsqueeze(0))

        return float(nn.MSELoss(reduction="sum")(moving, target).cpu().data.numpy())


