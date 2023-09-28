import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import init

FixedNormal = Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)
entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean


class SiLU(nn.Module):

    def __init__(self):
        super().__init__()

    def silu(input):
        return input * torch.sigmoid(input)

    def forward(self, input):
        return self.silu(input)


class GuaussianAction(nn.Module):

    def __init__(self, size_in, size_out):
        super().__init__()
        self.fc_mean = nn.Linear(size_in, size_out)

        # ====== INITIALIZATION ======
        self.fc_mean.weight.data.mul_(0.1)
        self.fc_mean.bias.data.mul_(0.0)

        self.logstd = torch.zeros(1, size_out)

    def forward(self, x):
        action_mean = self.fc_mean(x)

        # print(action_mean.shape, self.logstd.shape)
        return FixedNormal(action_mean, self.logstd.exp())


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BaseActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False, use_continuous=False):
        super(BaseActorCriticNetwork, self).__init__()
        if use_noisy_net:
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.use_continuous = use_continuous

        # self.feature = nn.Sequential(
        #     linear(input_size, 128),
        #     nn.ReLU(),
        #     linear(128, 128),
        #     nn.ReLU()
        # )
        self.actor = nn.Sequential(
            linear(input_size, 128),
            nn.ReLU(),
            linear(128, 64),
            nn.ReLU(),
            GuaussianAction(64, output_size) if use_continuous else linear(64, output_size)
        )
        self.critic = nn.Sequential(
            linear(input_size, 128),
            nn.ReLU(),
            linear(128, 64),
            nn.ReLU(),
            linear(64, 1)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.xavier_normal_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.xavier_normal_(p.weight)
                p.bias.data.zero_()

    def forward(self, state):
        # x = self.feature(state)
        policy = self.actor(state)
        value = self.critic(state)
        return policy, value


class DeepCnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(DeepCnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4),
            nn.ReLU(),
            Flatten(),
            linear(50176, 512),
            nn.ReLU()
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                512),
            nn.LeakyReLU(),
        )
        self.actor = linear(512, output_size)
        self.critic = linear(512, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value


class CuriosityModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CuriosityModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(feature_output * 2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size)
        )

        self.forward_net = nn.Sequential(
            nn.Linear(output_size + feature_output, 512),
            nn.LeakyReLU(),
            nn.Linear(512, feature_output)
        )
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        # get pred action
        pred_action = torch.cat((encode_state, self.feature(next_state)), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature = torch.cat((encode_state, action), 1)
        pred_next_state_feature = self.forward_net(pred_next_state_feature)

        real_next_state_feature = self.feature(next_state)
        return real_next_state_feature, pred_next_state_feature, pred_action