import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import scipy.io as sio
import yaml

from math import ceil


def save_ppg_chunks(ppg_clips, save_name, save_path):
    for i in range(len(ppg_clips)):
        input_path_name = save_path + f'/{save_name}_label_ppg{i}'
        np.save(input_path_name, ppg_clips[i])


def read_mat(mat_file):
    try:
        mat = sio.loadmat(mat_file)
    except:
        for _ in range(20):
            print(mat_file)
    frames = np.array(mat['video'])
    return frames


def chunk_frames(frames, clip_length):
    """Chunks the data into clips."""
    clip_num = frames.shape[0] // clip_length
    frames_clips = [frames[i * clip_length:(i + 1) * clip_length] for i in range(clip_num)]

    return np.array(frames_clips)


def chunk_biosignal(biosignal, clip_length):
    """Chunks the data into clips."""
    clip_num = biosignal.shape[0] // clip_length
    biosignal_clips = [biosignal[i * clip_length:(i + 1) * clip_length] for i in range(clip_num)]
    return np.array(biosignal_clips)


# Load configs parameters from yaml file
def load_configs(cfg_path):
    with open(cfg_path, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return configs


def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(np.linspace(1, a.shape[0], len),
                     np.linspace(1, a.shape[0], a.shape[0]),
                     a)


#%% Mediapipe face detection
def get_landmark_coords(landmarks, width, height):
    """Extract normalized landmark coordinates to array of pixel coordinates."""
    xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return np.multiply(xyz, [width, height, width]).astype(int)


def fill_roimask(point_list, img):
    """Create binary mask, filled inside contour given by list of points.
    """
    mask = np.zeros(img.shape[:2], dtype="uint8")
    if len(point_list) > 2:
        contours = np.reshape(point_list, (1, -1, 1, 2))  # expected by OpenCV
        cv2.drawContours(mask, contours, 0, color=255, thickness=cv2.FILLED)
    return mask


def resize_face_frames_mediapipe(frames, configs, detector, old_coords, participant):
    channel_values_temp = []
    resized_frames = np.zeros((len(frames), configs['h'], configs['w'], 1), dtype='uint8')
    x1, x2, y1, y2 = None, None, None, None
    for i_frame, frame_orig in enumerate(frames):
        if i_frame % configs['detection_length'] == 0:
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_orig)
            detection_result = detector.detect(frame)

            # Get landmark coordinates
            landmarks_extracted = detection_result.face_landmarks

            # If no landmark coordinates can be extracted, use coordinates from previous frame
            if len(landmarks_extracted) == 0:
                print(f"No landmarks detected for face for participant {participant}")
                coords = old_coords
            else:
                coords = get_landmark_coords(landmarks_extracted[0], frame.width, frame.height)
                old_coords = coords

            # Get mesh indices of bounding box of the whole face for ML
            # x_min, x_max, y_min, y_max = coords[[234, 454, 10, 152], :2]
            # x_min, x_max, y_min, y_max = coords[[234, 454, 10, 152], :2]
            # diff = abs((x_max[0] - x_min[0]) - (y_max[1] - y_min[1]))
            # x_min = x_min - diff // 2
            # x_max = x_max + diff // 2

            # Get mean RGB values of specific region
            # Forehead: [107, 108, 109, 10, 338, 337, 336]
            # Periorbital: [130, 225, 228, 445, 448, 359]
            # Periorbital wide: [117, 46, 276, 346]
            landmark_coord = [117, 46, 276, 346]
            point_list = coords[landmark_coord, :2]
            # roimask = fill_roimask(point_list, frame_orig)

            x_roi, y_roi, w_roi, h_roi = cv2.boundingRect(point_list)  # point_list is (N, 2)

            # 2) clip the rectangle so we never fall outside the image
            x1 = max(0, x_roi)
            y1 = max(0, y_roi)
            x2 = min(frame_orig.shape[1], x_roi + w_roi)
            y2 = min(frame_orig.shape[0], y_roi + h_roi)

            """# 3) crop both the RGB frame and the mask to that rectangle
            # crop_rgb = frame_orig[y1:y2, x1:x2]  # (h_roi, w_roi, 3)

            # 5) resize the masked ROI to 48×128  (cv2: width first, height second)
            # roi_48x128 = cv2.resize(crop_rgb, (128, 48), interpolation=cv2.INTER_AREA)

            fig, ax = plt.subplots()
            ax.imshow(crop_rgb)
            fig.show()

            fig, ax = plt.subplots()
            ax.imshow(roi_48x128)
            fig.show()"""

        # Crop to bounding box (whole face) and resize frames for ML model
        """resized_frames[i_frame] = cv2.resize(frame_orig[max(y_min[1], 0):min(y_max[1], frame_orig.shape[0]),
                                             max(x_min[0], 0):min(x_max[0], frame_orig.shape[1])],
                                             (configs['w'], configs['h']), interpolation=cv2.INTER_AREA)"""

        resized_frames[i_frame] = np.expand_dims(cv2.cvtColor(cv2.resize(frame_orig[y1:y2, x1:x2], (configs['w'], configs['h']),
                                             interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY), axis=2)

        """fig, ax = plt.subplots()
        ax.imshow(resized_frames[i_frame], cmap='gray')
        fig.show()"""

    return resized_frames, old_coords


# %% Video preprocessing
def facial_detection(frame, larger_box=False, larger_box_size=1.0):
    """Conducts face detection on a single frame.
    Sets larger_box=True for larger bounding box, e.g. moving trials."""
    detector = cv2.CascadeClassifier('./configs/haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if (len(face_zone) < 1):
        print("ERROR:No Face Detected")
        result = [0, 0, frame.shape[0], frame.shape[1]]
    elif (len(face_zone) >= 2):
        result = np.argmax(face_zone, axis=0)
        result = face_zone[result[2]]
        # print("Warning: More than one faces are detected (only cropping the biggest one).")
    else:
        result = face_zone[0]
    if larger_box:
        print("Larger Bounding Box")
        result[0] = max(0, result[0] - (larger_box_size-1.0) / 2 * result[2])
        result[1] = max(0, result[1] - (larger_box_size-1.0) / 2 * result[3])
        result[2] = larger_box_size * result[2]
        result[3] = larger_box_size * result[3]
    return result


def resize_frames(frames, configs, participant):
    if configs['face_detect']:
        if configs['dynamic_detection']:
            det_num = ceil(frames.shape[0] / configs['detection_length'])
        else:
            det_num = 1
        face_region = list()
        for idx in range(det_num):
            if configs['crop_face']:
                if type(configs['larger_box_size']) == float or type(configs['larger_box_size']) == int:
                    face_region.append(facial_detection(frames[configs['detection_length'] * idx],
                                                        larger_box=configs['large_face_box'],
                                                        larger_box_size=configs['larger_box_size']))
                elif type(configs['larger_box_size']) == str:
                    crop_coords = configs['crop_coords_all'][participant][configs['larger_box_size']]
                    face_region.append([crop_coords[2], crop_coords[0],
                                       crop_coords[3] - crop_coords[2], crop_coords[1] - crop_coords[0]])
            else:
                face_region.append(frames[0])
        face_region_all = np.asarray(face_region, dtype='int')
    else:
        assert (configs['dynamic_detection'] is False)  # dynamic_det can be True only when face_detection is True
        face_region_all = [0, 0, frames[0].shape[0], frames[0].shape[1]]

    resize_frames = np.zeros((frames.shape[0], configs['h'], configs['w'], 3))

    for i in range(0, frames.shape[0]):
        frame = frames[i]
        if configs['dynamic_detection']:
            reference_index = i // configs['detection_length']
        else:
            reference_index = 0
        if configs['crop_face']:
            face_region = face_region_all[reference_index]

            if type(configs['larger_box_size']) == str:
                frame = cv2.resize(frame, (256, 205), interpolation=cv2.INTER_AREA)

            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]

        resize_frames[i] = cv2.resize(frame, (configs['w'], configs['h']), interpolation=cv2.INTER_AREA)
        # resize_frames[i] = resize_frames[i].astype(np.uint8)

    # resize_frames = np.float32(resize_frames) / 255
    # resize_frames[resize_frames > 1] = 1
    # resize_frames[resize_frames < (1 / 255)] = 1 / 255

    return resize_frames


"""def diff_normalize_data(data):
    # Difference frames and normalization data
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len - 1):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


def diff_data(data):
    # Difference frames and normalization data
    n, h, w, c = data.shape
    normalized_len = n - 1
    normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
    normalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(normalized_len - 1):
        normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :])
    normalized_data = normalized_data / np.std(normalized_data)
    normalized_data = np.append(normalized_data, normalized_data_padding, axis=0)
    normalized_data[np.isnan(normalized_data)] = 0

    return normalized_data


def diff_nostd_data(data):
    # Difference frames and normalization data
    n, h, w, c = data.shape
    normalized_len = n - 1
    normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
    normalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(normalized_len - 1):
        normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :])

    # normalized_data[np.isnan(normalized_data)] = 0
    # normalized_data = normalized_data / np.std(normalized_data)
    normalized_data = np.append(normalized_data, normalized_data_padding, axis=0)
    normalized_data[np.isnan(normalized_data)] = 0
    return normalized_data


def standardized_data(data):
    # Standardize data
    # data[data < 1] = 1
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data"""


# %% Label preprocessing
"""def diff_normalize_label(label):
    # Difference frames and normalization labels
    diff_label = np.diff(label, axis=0)
    diffnormalized_label = diff_label / np.std(diff_label)
    diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
    diffnormalized_label[np.isnan(diffnormalized_label)] = 0
    return diffnormalized_label


def diff_label(label):
    diff_label = np.diff(label, axis=0)
    diff_label = diff_label / np.std(diff_label)
    diff_label = np.append(diff_label, np.zeros(1), axis=0)
    diff_label[np.isnan(diff_label)] = 0
    return diff_label


def standardized_label(label):
    label = label - np.mean(label)
    label = label / np.std(label)
    label[np.isnan(label)] = 0
    return label"""