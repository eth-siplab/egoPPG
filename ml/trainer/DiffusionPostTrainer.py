import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from source.ml.loss.FreqLoss import FreqLoss
from source.ml.models.DiffusionDenoiserPost import DiffusionDenoiserPost, PostProcessingDiffusionUNet, PostProcessingDiffusionTransformer
from source.ml import data_loader, trainer, ml_helper
from source.ml.config import get_config


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create a general generator for use with the validation dataloader,
    # the test dataloader, and the unsupervised dataloader
    general_generator = torch.Generator()
    general_generator.manual_seed(random_seed)
    # Create a training generator to isolate the train dataloader from
    # other dataloaders and better control non-deterministic behavior
    train_generator = torch.Generator()
    train_generator.manual_seed(random_seed)

    return general_generator, train_generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_sinusoidal_interference(signal, fs=30, interference_prob=0.3, freq_range=(0.7, 3.0), amp_range=(0.1, 0.5)):
    """
    Adds sinusoidal interference to the input signal.

    Args:
        signal (torch.Tensor): Input tensor of shape [batch, 1, signal_length].
        fs (float): Sampling frequency in Hz.
        interference_prob (float): Probability of applying sinusoidal interference to each sample.
        freq_range (tuple): Frequency range for the interference in Hz (e.g., (0.7, 3.0)).
        amp_range (tuple): Amplitude range for the interference.

    Returns:
        torch.Tensor: Signal with sinusoidal interference added.
    """
    noisy_signal = signal.clone()
    batch_size, _, signal_length = signal.shape
    # Create a time vector based on sampling frequency.
    t = torch.arange(signal_length, device=signal.device) / fs  # shape: [signal_length]

    for i in range(batch_size):
        if torch.rand(1).item() < interference_prob:
            # Random frequency within the specified range
            freq = torch.empty(1).uniform_(freq_range[0], freq_range[1]).item()
            # Random amplitude within the specified range
            amp = torch.empty(1).uniform_(amp_range[0], amp_range[1]).item()
            # Random phase between 0 and 2*pi
            phase = torch.empty(1).uniform_(0, 2 * math.pi).item()
            # Generate the interference signal
            interference = amp * torch.sin(2 * math.pi * freq * t + phase)
            # Add the interference to the entire signal of the current sample.
            noisy_signal[i, 0, :] += interference
    return noisy_signal


def add_burst_noise(signal, burst_prob=0.3, burst_length=10, burst_amplitude=0.5):
    """
    Adds burst noise (spikes) to the input signal.

    Args:
        signal (torch.Tensor): Input tensor of shape [batch, 1, signal_length].
        burst_prob (float): Probability of adding a burst event to each sample.
        burst_length (int): Number of consecutive samples affected by the burst.
        burst_amplitude (float): Maximum amplitude of the burst (can be positive or negative).

    Returns:
        torch.Tensor: Signal with burst noise added.
    """
    noisy_signal = signal.clone()
    batch_size, _, signal_length = signal.shape
    for i in range(batch_size):
        if torch.rand(1).item() < burst_prob:
            # Choose a random start index (ensure burst fits within signal)
            if signal_length > burst_length:
                start = torch.randint(0, signal_length - burst_length, (1,)).item()
            else:
                start = 0
                burst_length = signal_length
            # Generate a random amplitude in the range [-burst_amplitude, burst_amplitude]
            amplitude = burst_amplitude * (2 * torch.rand(1).item() - 1)
            # Optionally, one could shape the burst (e.g., with a window) to mimic a spike pattern.
            noisy_signal[i, 0, start:start + burst_length] += amplitude
    return noisy_signal


def add_illumination_change(signal, segment_fraction=0.2, illum_prob=0.3, factor_range=(0.5, 1.5)):
    """
    Applies an illumination change to a random segment of the input signal.

    Args:
        signal (torch.Tensor): Input tensor of shape [batch, 1, signal_length].
        segment_fraction (float): Fraction of the signal length to modify.
        illum_prob (float): Probability of applying an illumination change to each sample.
        factor_range (tuple): (min_factor, max_factor) defining the scaling factor range.

    Returns:
        torch.Tensor: Signal with illumination change applied.
    """
    noisy_signal = signal.clone()
    batch_size, _, signal_length = signal.shape
    segment_length = int(segment_fraction * signal_length)

    for i in range(batch_size):
        if torch.rand(1).item() < illum_prob:
            # Ensure the segment fits within the signal length.
            if signal_length > segment_length:
                start = torch.randint(0, signal_length - segment_length, (1,)).item()
            else:
                start = 0
                segment_length = signal_length
            # Choose a random scaling factor within the specified range.
            factor = torch.empty(1).uniform_(factor_range[0], factor_range[1]).item()
            noisy_signal[i, 0, start:start + segment_length] *= factor
    return noisy_signal


def add_combined_noise(signal, gaussian_std=0.1, burst_prob=0.3, burst_length=10, burst_amplitude=0.5):
    """
    Adds combined Gaussian noise and burst noise to the input signal.

    Args:
        signal (torch.Tensor): Input tensor of shape [batch, 1, signal_length].
        gaussian_std (float): Standard deviation for the Gaussian noise.
        burst_prob (float): Probability of adding a burst event to each sample.
        burst_length (int): Length of the burst event (number of samples).
        burst_amplitude (float): Maximum amplitude of the burst.

    Returns:
        torch.Tensor: Signal with both Gaussian and burst noise added.
    """
    # Add Gaussian noise first
    noisy_signal = signal + gaussian_std * torch.randn_like(signal)
    # Then add burst noise on top
    # noisy_signal = add_illumination_change(noisy_signal, segment_fraction=0.2, illum_prob=0.7, factor_range=(0.5, 1.5))
    noisy_signal = add_burst_noise(noisy_signal, burst_prob, burst_length, burst_amplitude)
    # noisy_signal = add_sinusoidal_interference(noisy_signal, fs=30, interference_prob=0.5,
    #                                                 freq_range=(0.7, 3.0), amp_range=(0.1, 0.5))

    # Normalize the IMU magnitude for each sample along the time dimension.
    """imu_min = imu.amin(dim=2, keepdim=True)
    imu_max = imu.amax(dim=2, keepdim=True)
    imu_normalized = (imu - imu_min) / (imu_max - imu_min + 1e-8)
    noisy_signal = signal + imu_scaling * imu_normalized"""

    return noisy_signal


# ----------------------- Training Routine -----------------------
def train_diffusion_model(config, num_steps, train_generator, i_split, model_dir):
    dataset = data_loader.DatasetLoader.DatasetLoader(
        dataset_name="Train",
        cv_split_files=cv_split_files['train'][cv_split],
        input_signals=config.INPUT_SIGNALS,
        label_signals=config.LABEL_SIGNALS,
        config_data=config.TRAIN.DATA)
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=16,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=train_generator)

    # Instantiate the diffusion model and optimizer
    model = DiffusionDenoiserPost(num_steps=num_steps).to(config.DEVICE)
    # model = PostProcessingDiffusionUNet(diffusion_steps=50, device=config.DEVICE).to(config.DEVICE)
    # model = PostProcessingDiffusionTransformer(frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH, diffusion_steps=50, device=config.DEVICE).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    smooth_loss = nn.SmoothL1Loss()

    # Define a linear noise schedule
    beta_start = 1e-4
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)  # cumulative product: shape [num_steps]
    # Move alpha_bars to torch tensor on the device
    alpha_bars = torch.tensor(alpha_bars, dtype=torch.float32, device=config.DEVICE)
    criterion = FreqLoss('rr', 30, config.DEVICE)
    max_burst_amplitude = 40

    print(f"Training diffusion model on {len(dataset)} samples for {config.TRAIN.EPOCHS} epochs...")
    for epoch in range(config.TRAIN.EPOCHS):
        model.train()
        running_loss = 0.0

        tbar = tqdm(dataloader, ncols=80)
        for idx, batch in enumerate(tbar):
            # x0: [batch, 1, signal_length]
            x0 = batch[1][0].to(config.DEVICE)
            x0 = x0[:, None, :]

            """imu = batch[0][1].to(config.DEVICE)
            imu = imu[:, None, :]
            imu = torch.cumsum(imu, dim=2)"""

            # x0 = x0.to(config.DEVICE)
            current_batch = x0.size(0)

            # Sample a random timestep t for each sample in the batch, in [0, num_steps-1]
            t = torch.randint(0, num_steps, (current_batch,), device=config.DEVICE)
            # Get the corresponding cumulative product value for each t, reshape for broadcasting.
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1)

            # Sample noise (epsilon) from a standard normal distribution.
            # noise = torch.randn_like(x0)
            # burst_prob = min(0.1 + epoch * 0.005, max_burst_prob)
            burst_amplitude = min(epoch, max_burst_amplitude)
            noise = add_combined_noise(torch.zeros_like(x0), gaussian_std=1.0, burst_prob=0.2, burst_length=100, burst_amplitude=burst_amplitude)

            # Simulate the forward diffusion process:
            # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

            # Normalize timestep t to [0, 1] for conditioning the network.
            t_normalized = t.float() / num_steps

            # Predict the noise using the diffusion model.
            pred_noise = model(x_t, t_normalized)

            """if idx == 0:
                fig, ax = plt.subplots()
                ax.plot(x0[0, 0, :].detach().cpu().numpy())
                fig.show()
                fig, ax = plt.subplots()
                ax.plot(x_t[0, 0, :].detach().cpu().numpy())
                fig.show()
                fig, ax = plt.subplots()
                ax.plot(pred_noise[0, 0, :].detach().cpu().numpy())
                fig.show()"""

            # Compute MSE loss between the predicted noise and the actual noise.
            # loss = mse_loss(pred_noise, noise)
            loss = smooth_loss(pred_noise, noise)  # Huber loss
            # loss = criterion(pred_noise[:, 0, :], noise[:, 0, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * current_batch

        avg_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{config.TRAIN.EPOCHS}] - Loss: {avg_loss:.6f}")

    # Save the trained diffusion model.
    torch.save(model.state_dict(), f"{model_dir}/diffusion_denoiser_{i_split}.pth")
    print(f"Training complete. Model saved as 'diffusion_denoiser_{i_split}.pth'.")


if __name__ == "__main__":
    # %% Data input & output
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='Name of the configuration file')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # %% Load configs
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    # Variable parameters
    n_steps = 50
    lr = 1e-3

    # Set seed
    random_seed = 0
    print(f'\nRandom seed: {random_seed}')
    general_generator, train_generator = set_seed(random_seed)

    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                    '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']

    # Get cv splits depending on defined cv split method and dataset
    cv_split_files = ml_helper.get_cv_split_files(config, participants, None, random_seed)

    # Create dirs for logging models and training/testing outputs
    model_dir = config.MODEL.PATH_MODEL + f'/seed_{random_seed}'
    log_training_dir = config.LOG.PATH_TRAINING + f'/seed_{random_seed}'
    log_testing_dir = config.LOG.PATH_TESTING + f'/seed_{random_seed}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_training_dir):
        os.makedirs(log_training_dir)
    if not os.path.exists(log_testing_dir):
        os.makedirs(log_testing_dir)

    for i_split, cv_split in enumerate(cv_split_files['train'].keys()):
        train_diffusion_model(config, n_steps, train_generator, i_split, model_dir)
