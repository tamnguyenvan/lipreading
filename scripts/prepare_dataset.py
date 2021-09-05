import os
import string
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import imutils
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = list(string.ascii_lowercase) + list(string.digits)
MAX_FRAMES = 40
MOUTH_WIDTH = 96
MOUTH_HEIGHT = 96


def pad_frame(frame):
    """
    """
    height, width = frame.shape[:2]
    if width < height:
        frame = imutils.resize(frame, width=MOUTH_WIDTH)
    else:
        frame = imutils.resize(frame, height=MOUTH_HEIGHT)

    height, width = frame.shape[:2]
    if height > MOUTH_HEIGHT:
        margin = height - MOUTH_HEIGHT
        offset = margin // 2
        frame = frame[offset:offset+MOUTH_HEIGHT, :]
    else:
        pad_y = MOUTH_HEIGHT - height
        pad_y_beg = pad_y // 2
        pad_y_end = pad_y - pad_y_beg
        frame = np.pad(frame, [[pad_y_beg, pad_y_end], [0, 0]])
    if width > MOUTH_WIDTH:
        margin = width - MOUTH_WIDTH
        offset = margin // 2
        frame = frame[:, offset:offset+MOUTH_WIDTH]
    else:
        pad_x = MOUTH_WIDTH - width
        pad_x_beg = pad_x // 2
        pad_x_end = pad_x - pad_x_beg
        frame = np.pad(frame, [[0, 0], [pad_x_beg, pad_x_end]])
    
    msg = [frame.shape]
    assert frame.shape == (MOUTH_HEIGHT, MOUTH_WIDTH), ' '.join(map(str, msg))
    return frame


def pad_video(video):
    """
    """
    num_pad = max(0, MAX_FRAMES - len(video))
    padded_video = video + [video[-1] for _ in range(num_pad)]
    padded_video = padded_video[:MAX_FRAMES]
    return padded_video


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


def load_video(frame_dir):
    video = []
    paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
    for path in paths:
        frame = cv2.imread(path, 0)
        frame = pad_frame(frame)
        video.append(frame)
        if len(video) >= MAX_FRAMES:
            break
    video = pad_video(video)
    video = np.array(video)
    return video


class VisualDataset(Dataset):
    def __init__(self, paths, target_dir):
        """
        """
        # Labels should range in [0, 35] that includes 26 letters and 10 digits
        # Letters a-z would be indexed from 0 to 25
        self.paths = paths
        self.target_dir = target_dir
        self.labels = []
        for path in self.paths:
            dirname = os.path.basename(os.path.dirname(path))
            filename = Path(path).stem
            # AVDigits data file format like this.
            if dirname == 'avletters_cropped':
                class_name = filename[0].lower()
            elif dirname == 'avdigits_cropped':
                class_name = filename.split('_')[1]
            else:
                raise Exception(f'Wrong path {path}')
            print(dirname, path, class_name)
            label = CLASS_NAMES.index(class_name)
            self.labels.append(label)

    def __getitem__(self, idx):
        frame_dir = self.paths[idx]
        video_input = load_video(frame_dir)

        result = {}
        result['video'] = video_input
        result['label'] = self.labels[idx]
        result['duration'] = np.ones(len(video_input)).astype(bool)
        filename = Path(frame_dir).stem
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
    avletters_mat_paths = sorted(glob.glob(os.path.join(root_dir, 'avletters_cropped', '*')))
    avdigits_vid_paths = sorted(glob.glob(os.path.join(root_dir, 'avdigits_cropped', '*')))

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
        ds_target_dir = os.path.join(args.target_dir, name)
        os.makedirs(ds_target_dir, exist_ok=True)
        dataset = VisualDataset(paths, ds_target_dir)
        loader = DataLoader(dataset,
                batch_size=64,
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