""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""
import torch
import torch.nn as nn


class DeepConvLSTMBe(nn.Module):
    def __init__(self, n_channels, conv_kernels=64, kernel_size=3, LSTM_units=128):
        super(DeepConvLSTMBe, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1), padding='same')
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1), padding='same')
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1), padding='same')
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1), padding='same')

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2, batch_first=True)

        self.out_dim = LSTM_units

        self.classifier = nn.Linear(LSTM_units, 1)

        self.activation = nn.ReLU()

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)

        x, h = self.lstm(x)
        x = x[:, -1, :]

        out = self.classifier(x)
        return out


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, conv_kernels=64, kernel_size=5, LSTM_units=128):
        super(DeepConvLSTM, self).__init__()

        """self.conv1 = nn.Conv1d(n_channels, conv_kernels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size, padding=kernel_size // 2)
        self.conv4 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size, padding=kernel_size // 2)"""

        self.conv1 = nn.Conv1d(n_channels, conv_kernels, kernel_size)
        self.conv2 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size)
        self.conv3 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size)
        self.conv4 = nn.Conv1d(conv_kernels, conv_kernels, kernel_size)

        self.dropout = nn.Dropout(0.5)
        # self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2)
        self.lstm = nn.LSTM(conv_kernels, LSTM_units, num_layers=2, batch_first=True)

        # Define output layer dimensions based on application
        self.classifier = nn.Linear(LSTM_units, 1)  # Assuming 1 output for heart rate prediction

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # Apply convolutional layers with activation
        self.lstm.flatten_parameters()

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # Apply dropout
        x = self.dropout(x)

        # Prepare data for LSTM by permuting to (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)  # Shape becomes (batch_size, temporal_length, conv_kernels)

        # Pass through LSTM
        x, _ = self.lstm(x)  # Output x shape: (batch_size, temporal_length, LSTM_units)

        # No repeat  # ToDo
        """x = self.classifier(x)  # Shape becomes (batch_size, 1)
        x = x.squeeze(-1)  # Shape becomes (batch_size, temporal_length)"""

        # Repeat
        x = x[:, -1, :]  # Shape becomes (batch_size, LSTM_units)
        x = self.classifier(x)  # Shape becomes (batch_size, 1)
        # x = x.repeat(1, 128)  # Shape becomes (batch_size, 128)

        return x  # Returning the output and final LSTM hidden state if needed


class PhysNetIMU(nn.Module):
    def __init__(self, frames=128):
        super(PhysNetIMU, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(1, 16, (1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), stride=1, padding=1),  # ToDo
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # ToDo
        """self.ConvBlockIMU = nn.Sequential(
            nn.Conv1d(1, 24, 3, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
        )"""

        # ToDo
        """self.ConvBlockIMU = nn.Sequential(  
            nn.Conv1d(1, 6144, 3, stride=1, padding=1),
            nn.BatchNorm1d(6144),
            nn.ReLU(inplace=True),
        )"""

        """self.LinearIMU1 = nn.Linear(256, 256)   # ToDo
        self.LinearIMU2 = nn.Linear(256, 128)   # ToDo"""

        self.ConvBlock10 = nn.Conv3d(64, 1, (1, 1, 1), stride=1, padding=0)  # ToDo

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames, 1, 1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))   # better for EDA!

    def forward(self, x, imu):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, height, width] = x.shape

        x = self.ConvBlock1(x)  # x [16, T, 128,128]

        """imu = imu.view(batch, 1, length)
        imu = self.ConvBlockIMU(imu)
        imu = imu.view(batch, 1, length, 48, 128)
        x = torch.concat((x, imu), 1)"""

        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # End IMU
        """imu = imu.view(batch, 1, length)
        imu = self.ConvBlockIMU(imu)
        imu = imu.view(batch, 1, length, 3, 8)
        x = torch.concat((x, imu), 1)"""

        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        out = x.view(-1, length)

        # End HR IMU
        """out = torch.concat((out, imu), 1)
        out = self.LinearIMU1(out)
        out = self.LinearIMU2(out)"""

        return out