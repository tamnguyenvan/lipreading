import os
import string
import glob
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import VideoModel
import face_alignment
from utils.cvtransforms import CenterCrop


CLASS_NAMES = list(string.ascii_lowercase) + list(string.digits)
MAX_FRAMES = 40
MOUTH_WIDTH = 96
MOUTH_HEIGHT = 96


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


def get_mouth_roi(detector, frame):
    """
    """
    height, width = frame.shape[:2]
    normalize_ratio = None
    dets = detector.get_landmarks_from_image(frame)
    if dets is None:
        print('Can not detect any faces.')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(frame, (MOUTH_WIDTH, MOUTH_HEIGHT))

    det = dets[0]
    lips = det[48:60, :]
    xmin, ymin = int(np.min(lips[:, 0])), int(np.min(lips[:, 1]))
    xmax, ymax = int(np.max(lips[:, 0])), int(np.max(lips[:, 1]))
    mouth_w, mouth_h = xmax - xmin, ymax - ymin
    cy = (ymax + ymin) // 2

    # pad = 0.1
    cv2.imwrite('mouth_before.jpg', frame[ymin:ymax, xmin:xmax, :])
    pad_x = int(mouth_w * 0.4)
    pad_x_beg = pad_x // 2
    pad_x_end = pad_x - pad_x_beg

    xmin = max(0, xmin - pad_x_beg)
    xmax = min(width, xmax + pad_x_end)
    new_mouth_w = xmax - xmin
    ymin = max(0, cy - new_mouth_w // 2)
    ymax = min(height, cy + new_mouth_w // 2)
    mouth_roi = frame[ymin:ymax, xmin:xmax, :]
    cv2.imwrite('mouth_after.jpg', mouth_roi)
    mouth_roi = cv2.resize(mouth_roi, (MOUTH_WIDTH, MOUTH_HEIGHT))

    mouth_crop_image = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    return mouth_crop_image


def extract_opencv(filename):
    """
    """
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', face_detector='sfd')
    video = []
    cnt = 0
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read() # BGR
        if not ret:
            break
        
        mouth_roi = get_mouth_roi(detector, frame)
        mouth_roi = pad_frame(mouth_roi)
        video.append(mouth_roi)
        if len(video) >= MAX_FRAMES:
            break
    cap.release()
    video = pad_video(video)
    return video


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPUs would be used. e.g 0,1,2,3')
    parser.add_argument('--n_class', type=int, required=True, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    
    parser.add_argument('--source', type=str, help='Input dir or file')

    # load opts
    parser.add_argument('--weights', type=str, required=False, default=None,
                        help='Path to pretrained weights')

    # dataset
    parser.add_argument('--border', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--label_smooth', action='store_true')
    parser.add_argument('--se', action='store_true')
    args = parser.parse_args()

    # Load model
    print('Loading model...')

    video_model = VideoModel(args)
    ckpt_path = args.weights
    state_dict = torch.load(ckpt_path, map_location='cpu')['video_model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k.replace('module.', '')] = v

    video_model.load_state_dict(new_state_dict)
    video_model.cuda()
    video_model.eval()

    CLASS_NAMES = list(string.ascii_lowercase) + list(string.digits)
    if os.path.isdir(args.source):
        video_paths = sorted(glob.glob(os.path.join(args.source, '*')))
    else:
        video_paths = [args.source]

    for path in video_paths:
        video = extract_opencv(path)
        for i, frame in enumerate(video):
            cv2.imwrite(f'mouth_{i+1}.jpg', frame)

        inputs = np.stack(video, 0) / 255.
        inputs = CenterCrop(inputs, (88, 88))
        inputs = torch.FloatTensor(inputs[:, np.newaxis, ...])
        inputs = torch.unsqueeze(inputs, 0)
        inputs = inputs.cuda()
        print(inputs.shape)

        if args.border:
            border = torch.from_numpy(np.ones(inputs.shape[1])).cuda(non_blocking=True).float()
            preds = video_model(inputs, border)
        else:
            preds = video_model(inputs)

        pred_cls = preds.argmax(-1).item()
        print(F.softmax(preds, -1))
        score = torch.max(F.softmax(preds, -1), -1)[0].item()
        class_name = CLASS_NAMES[pred_cls]

        print(f'Video: {path} Prediction: {class_name} Score: {score:.4f}')