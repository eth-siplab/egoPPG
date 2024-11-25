import numpy as np

from source.utils import normalize


# %% Video preprocessing
def convert_to_uint8(data):
    """Convert data to uint8"""
    data = data + abs(np.min(data))
    data = np.around((data / np.max(data)) * 255).astype(np.uint8)
    return data


def diff_standardize_extended_video(data):
    """Difference frames and normalization data (see DeepPhys paper) """
    n, h, w, c = data.shape
    diff_standardized_extended_len = n - 1
    diff_standardized_extended_data = np.zeros((diff_standardized_extended_len, h, w, c), dtype=np.float32)
    diff_standardized_extended_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diff_standardized_extended_len - 1):
        diff_standardized_extended_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diff_standardized_extended_data = diff_standardized_extended_data / np.std(diff_standardized_extended_data)
    diff_standardized_extended_data = np.append(diff_standardized_extended_data,
                                                diff_standardized_extended_data_padding, axis=0)
    diff_standardized_extended_data[np.isnan(diff_standardized_extended_data)] = 0

    # Convert to uint8
    # diff_standardized_extended_data = convert_to_uint8(diff_standardized_extended_data)

    return diff_standardized_extended_data


def diff_standardize_video(data):
    """Difference frames and normalization data"""
    n, h, w, c = data.shape
    diff_standardized_len = n - 1
    diff_standardized_data = np.zeros((diff_standardized_len, h, w, c), dtype=np.float32)
    diff_standardized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diff_standardized_len - 1):
        diff_standardized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :])

    diff_standardized_data = diff_standardized_data / np.std(diff_standardized_data)
    diff_standardized_data = np.append(diff_standardized_data, diff_standardized_data_padding, axis=0)
    diff_standardized_data[np.isnan(diff_standardized_data)] = 0

    # Convert to uint8
    # diff_standardized_data = convert_to_uint8(diff_standardized_data)
    # diff_standardized_data = normalize(diff_standardized_data, 'zero_one')

    return diff_standardized_data


def diff_video(data):
    """Difference frames and normalization data"""
    n, h, w, c = data.shape
    diff_len = n - 1
    diff_data = np.zeros((diff_len, h, w, c), dtype=np.float32)
    diff_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diff_len - 1):
        diff_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :])
    diff_data = diff_data / np.std(diff_data)
    diff_data = np.append(diff_data, diff_data_padding, axis=0)
    diff_data[np.isnan(diff_data)] = 0

    # Convert to uint8
    # diff_data = convert_to_uint8(diff_data)

    return diff_data


def standardize_video(data):
    """Standardize data"""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0

    # Convert to uint8
    # data = convert_to_uint8(data)
    # data = normalize(data, 'zero_one')

    return data


# %% Label preprocessing
def diff_standardize_label(label):
    """Difference frames and normalization labels"""
    diff_labels = np.diff(label, axis=0)
    diff_standardize_labels = diff_labels / np.std(diff_labels)
    diff_standardize_labels = np.append(diff_standardize_labels, np.zeros(1), axis=0)  # (1, 3)
    diff_standardize_labels[np.isnan(diff_standardize_labels)] = 0
    return diff_standardize_labels


def diff_label(label):
    diff_labels = np.diff(label, axis=0)
    diff_labels = diff_labels / np.std(diff_labels)
    diff_labels = np.append(diff_labels, np.zeros(1), axis=0)
    diff_labels[np.isnan(diff_labels)] = 0
    return diff_labels


def standardize_label(label):
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label