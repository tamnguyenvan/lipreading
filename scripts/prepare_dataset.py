import os
import string
import glob
import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = list(string.ascii_lowercase) + list(string.digits)
MAX_FRAMES = 40
MOUTH_WIDTH = 96
MOUTH_HEIGHT = 96
HORIZONTAL_PAD = 0.11


def pad_frame(frame):
    """
    """
    height, width = frame.shape[:2]
    pad_y = max(0, MOUTH_HEIGHT - height)
    pad_x = max(0, MOUTH_WIDTH - width)
    pad_x_beg = pad_x // 2
    pad_x_end = pad_x - pad_x_beg
    pad_y_beg = pad_y // 2
    pad_y_end = pad_y - pad_y_beg
    padded_frame = np.pad(frame, [[pad_y_beg, pad_y_end], [pad_x_beg, pad_x_end]], mode='edge')
    padded_frame = padded_frame[:MOUTH_HEIGHT, :MOUTH_WIDTH]
    return padded_frame


def pad_video(video):
    """
    """
    num_pad = max(0, MAX_FRAMES - len(video))
    padded_video = video + [video[-1] for _ in range(num_pad)]
    padded_video = padded_video[:MAX_FRAMES]
    return padded_video


def get_mouth_roi(detector, predictor, frame):
    """
    """
    normalize_ratio = None
    dets = detector(frame, 1)
    shape = None
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        i = -1
    if shape is None: # Detector doesn't detect face, just return as is
        return frame

    mouth_points = []
    for part in shape.parts():
        i += 1
        if i < 48: # Only take mouth region
            continue
        mouth_points.append((part.x,part.y))
    np_mouth_points = np.array(mouth_points)

    mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

    if normalize_ratio is None:
        mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
        mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

        normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

    new_img_shape = (int(frame.shape[1] * normalize_ratio), int(frame.shape[0] * normalize_ratio))
    resized_img = cv2.resize(frame, new_img_shape)

    mouth_centroid_norm = mouth_centroid * normalize_ratio

    mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH // 2)
    mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH // 2)
    mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT // 2)
    mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT // 2)

    mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
    mouth_crop_image = cv2.cvtColor(mouth_crop_image, cv2.COLOR_BGR2GRAY)
    return mouth_crop_image


def extract_opencv(filename):
    """
    """
    curr_dir = os.path.dirname(__file__)
    shape_predictor_path = os.path.join(curr_dir, 'shape_predictor_68_face_landmarks.dat')
    assert os.path.exists(shape_predictor_path), f'Not found {shape_predictor_path}'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    video = []
    cnt = 0
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read() # BGR
        if not ret:
            break
        
        mouth_roi = get_mouth_roi(detector, predictor, frame)
        mouth_roi = pad_frame(mouth_roi)
        video.append(mouth_roi)
        if len(video) >= MAX_FRAMES:
            break
    cap.release()
    video = pad_video(video)
    return video


def extract_mat(filename):
    """
    """
    data = loadmat(filename)
    vid_np_raw = data['vid']
    size = data['siz'][0].astype('int32')  # (height, width, num_frames)
    vid_np = np.reshape(vid_np_raw, (size[1], size[0], size[2]))
    vid_np = np.transpose(vid_np, (2, 1, 0))
    num_frames = size[2]
    video = []
    for i in range(num_frames):
        frame = pad_frame(vid_np[i, :, :])
        video.append(frame)
        if len(video) >= MAX_FRAMES:
            break
    video = pad_video(video)
    return video


class VisualDataset(Dataset):
    def __init__(self, paths, target_dir):
        """
        """
        # Labels should range in [0, 35] that includes 26 letters and 10 digits
        # Letters a-z would be indexed from 0 to 25
        self.paths = paths[:25*32]
        self.target_dir = target_dir
        self.labels = []
        for path in self.paths:
            filename = Path(path).stem
            if path.endswith('.mat'):
                # AVLetters data file format like this.
                # CLASS_PERSON-lips.mat
                class_name = filename[0].lower()
                label = CLASS_NAMES.index(class_name)
            else:
                # AVDigits data file format like this.
                class_name = filename[1]
                label = CLASS_NAMES.index(class_name)
            self.labels.append(label)

    def __getitem__(self, idx):
        filepath = self.paths[idx]
        if filepath.endswith('.mat'):
            video_input = extract_mat(filepath)
        else:
            print(f'Extracting video {filepath}')
            video_input = extract_opencv(filepath)

        result = {}
        result['video'] = video_input
        result['label'] = self.labels[idx]
        result['duration'] = np.ones(len(video_input)).astype(bool)
        filename = Path(filepath).stem
        savename = os.path.join(self.target_dir, filename + '.pkl')
        torch.save(result, savename)

        return result

    def __len__(self):
        return len(self.paths)


def load_paths(root, split=0.1):
    """
    """
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(os.path.abspath(root)):
        root_dir = os.path.join(project_dir, root)
    else:
        root_dir = root

    # Load AVLetters
    avletters_mat_paths = sorted(glob.glob(os.path.join(root_dir, 'avletters', '*.mat')))
    avdigits_vid_paths = sorted(glob.glob(os.path.join(root_dir, 'avdigits', '*.mp4')))

    assert len(avletters_mat_paths) > 0, 'AVLetters dataset is empty'
    assert len(avdigits_vid_paths) > 0, 'AVDigits dataset is empty'

    # Split
    paths = avletters_mat_paths + avdigits_vid_paths
    num_train = int(len(paths) * (1 - split))
    indices = np.arange(len(paths))
    np.random.seed(12)
    np.random.shuffle(indices)
    paths = np.array(paths)[indices].tolist()

    train_paths = paths[:num_train]
    test_paths = paths[num_train:]
    print('Number of training samples:', len(train_paths))
    print('Number of testing samples', len(test_paths))

    splits = {
        'train': train_paths,
        'test': test_paths,
    }
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='datasets/',
                        help='Path to data dir')
    parser.add_argument('--target-dir', type=str,
                        default='datasets/avletters_digits_npy_gray_pkl_jpeg',
                        help='Target dir')
    parser.add_argument('--split', type=float, default=0.1, help='Train split')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)

    splits = load_paths(args.data_dir, args.split)
    for name, paths in splits.items():
        name = 'test'
        ds_target_dir = os.path.join(args.target_dir, name)
        os.makedirs(ds_target_dir, exist_ok=True)
        dataset = VisualDataset(paths, ds_target_dir)
        loader = DataLoader(dataset,
                batch_size=8,
                num_workers=8,
                shuffle=False,
                drop_last=False)

        print(f'Creating {name} dataset...')
        import time
        tic = time.time()
        total_batches = len(loader)
        for i, batch in enumerate(loader):
            toc = time.time()
            eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
            print(f'[Batch {i+1}/{total_batches}] eta:{eta:.5f}')
