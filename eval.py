import argparse
import string

from model import VideoModel
from utils import AVDataset as Dataset
import torch
from torch.utils.data import DataLoader


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True)
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPUs would be used. e.g 0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=400,
                        help='Batch size')
    parser.add_argument('--n_class', type=int, required=True, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    
    # load opts
    parser.add_argument('--weights', type=str, required=False, default=None,
                        help='Path to pretrained weights')

    # dataset
    parser.add_argument('--dataset', type=str, default='av', help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='datasets/avletters_digits_npy_gray_pkl_jpeg',
                        help='Path to .pkl dataset files')
    parser.add_argument('--border', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--label_smooth', action='store_true')
    parser.add_argument('--se', action='store_true')
    args = parser.parse_args()


    test_dataset = Dataset('datasets/avletters_digits_npy_gray_pkl_jpeg', 'test')
    loader = dataset2dataloader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

    # Load model
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
    for i, input in enumerate(loader):
        print(f'Batch {i+1}')
        video = input.get('video').cuda(non_blocking=True)
        label = input.get('label').cuda(non_blocking=True)
        border = input.get('duration').cuda(non_blocking=True).float()

        if args.border:
            y_v = video_model(video, border)
        else:
            y_v = video_model(video)
        
        preds = y_v.argmax(-1).cpu().numpy()
        label = label.cpu().numpy()

        for i in range(len(preds)):
            pred_cls = int(preds[i])
            class_id = int(label[i])
            print(f'Prediction: {CLASS_NAMES[pred_cls]} Label: {CLASS_NAMES[class_id]}')