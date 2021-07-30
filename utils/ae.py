#!/usr/bin/env python
import torch.nn as nn


def init_weights(m):
    """ initialize weights of fully connected layer
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_uniform(m):
    """
    initialize weights of discriminator with uniform distributions
    Args:
        m: weights of fully connected layers of the discriminator

    Returns:
        null
    """
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, 0, 1)
        nn.init.uniform_(m.bias, 0, 0.01)


def init_normal(m):
    """
    initialize weights of discriminator with normal distributions
    Args:
        m: weights of fully connected layers of the diecriminator

    Returns:
        null
    """
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0, 1)
        m.bias.data.fill_(0.01)


# autoencoder with hidden units 20, 2, 20
# Encoder
class Encoder_2(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            nn.Linear(20, 2))
        self.encoder.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        return x


# Decoder
class Decoder_2(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_2, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, num_inputs),
            nn.ReLU())
        self.decoder.apply(init_weights)

    def forward(self, x):
        x = self.decoder(x)
        return x


# Autoencoder
class autoencoder_2(nn.Module):
    def __init__(self, num_inputs):
        super(autoencoder_2, self).__init__()
        self.encoder = Encoder_2(num_inputs)
        self.decoder = Decoder_2(num_inputs)

    def forward(self, x):
        code = self.encoder(x)
        x = self.decoder(code)
        return code, x


# autoencoder with hidden units 200, 20, 200
# Encoder
class Encoder_20(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder_20, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 200),
            nn.ReLU(),
            nn.Linear(200, 20),
        )
        self.encoder.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        return x


# Decoder
class Decoder_20(nn.Module):
    def __init__(self, num_inputs):
        super(Decoder_20, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(),
            nn.Linear(200, num_inputs),
            nn.ReLU())
        self.decoder.apply(init_weights)

    def forward(self, x):
        x = self.decoder(x)
        return x


# Autoencoder
class autoencoder_20(nn.Module):
    def __init__(self, num_inputs):
        super(autoencoder_20, self).__init__()
        self.encoder = Encoder_20(num_inputs)
        self.trans = nn.Linear(20, 20)
        self.decoder = Decoder_20(num_inputs)

    def forward(self, x):
        additional = self.encoder(x)
        code = self.trans(additional)
        code_trans = self.trans(code)
        x = self.decoder(code_trans)
        return additional, code, x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_inputs):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_inputs, 20),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        self.main.apply(init_uniform)

    def forward(self, x):
        x = self.main(x)
        return x
