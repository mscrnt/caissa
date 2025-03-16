import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ChessCNN(BaseFeaturesExtractor):
    """
    Custom CNN for chess board state (12x8x8 input).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(ChessCNN, self).__init__(observation_space, features_dim)

        # The CNN should work with (12, 8, 8) input, not standard images
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),  # (32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (64, 8, 8)
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute output shape after CNN
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()["board_state"]).unsqueeze(0)
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations["board_state"])
        return self.fc(x)
