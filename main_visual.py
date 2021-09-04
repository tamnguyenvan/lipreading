import os
import time
import argparse

import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from model import VideoModel
from lsr import LSR


def parallel_model(model):
    model = nn.DataParallel(model)
    return model


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True)
    return loader


def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg


def test():
    """
    """
    with torch.no_grad():
        dataset = Dataset(args.data_dir, 'test')
        print('Start Testing, Data Length:',len(dataset))
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)

        print('start testing')
        v_acc = []
        total = 0

        for (i_iter, input) in enumerate(loader):

            video_model.eval()

            tic = time.time()
            if is_cuda:
                video = input.get('video').cuda(non_blocking=True)
                label = input.get('label').cuda(non_blocking=True)
                border = input.get('duration').cuda(non_blocking=True).float()
            else:
                video = input.get('video').to(device)
                label = input.get('label').to(device)
                border = input.get('duration').to(device).float()
            total = total + video.size(0)

            with autocast():
                if(args.border):
                    y_v = video_model(video, border)
                else:
                    y_v = video_model(video)


            v_acc.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            if(i_iter % 10 == 0):
                msg = ''
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(loader)-i_iter)/3600.0)

                print(msg)

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)

        return acc, msg

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)


def train():
    dataset = Dataset(args.data_dir, 'train')
    print('Start Training, Data Length:',len(dataset))

    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
    max_epoch = args.max_epoch

    tot_iter = 0
    best_acc = 0.0
    alpha = 0.2
    scaler = GradScaler()
    for epoch in range(max_epoch):
        lsr = LSR()

        for (i_iter, input) in enumerate(loader):
            tic = time.time()

            video_model.train()
            if is_cuda:
                video = input.get('video').cuda(non_blocking=True)
                label = input.get('label').cuda(non_blocking=True).long()
                border = input.get('duration').cuda(non_blocking=True).float()
            else:
                video = input.get('video').to(device)
                label = input.get('label').to(device).long()
                border = input.get('duration').to(device).float()

            loss = {}

            if args.label_smooth:
                loss_fn = lsr
            else:
                loss_fn = nn.CrossEntropyLoss()

            with autocast():
                if args.mixup:
                    lambda_ = np.random.beta(alpha, alpha)
                    if is_cuda:
                        index = torch.randperm(video.size(0)).cuda(non_blocking=True)
                    else:
                        index = torch.randperm(video.size(0)).to(device)

                    mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                    mix_border = lambda_ * border + (1 - lambda_) * border[index, :]

                    label_a, label_b = label, label[index]

                    if args.border:
                        y_v = video_model(mix_video, mix_border)
                    else:
                        y_v = video_model(mix_video)

                    loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)

                else:
                    if args.border:
                        y_v = video_model(video, border)
                    else:
                        y_v = video_model(video)

                    loss_bp = loss_fn(y_v, label)

            loss['CE V'] = loss_bp

            optim_video.zero_grad()
            scaler.scale(loss_bp).backward()
            scaler.step(optim_video)
            scaler.update()

            toc = time.time()

            msg = 'epoch={},train_iter={}/{},eta={:.5f}'.format(epoch, tot_iter, len(loader), (toc-tic)*(len(loader)-i_iter)/3600.0)
            for k, v in loss.items():
                msg += ',{}={:.5f}'.format(k, v)
            msg = msg + str(',lr=' + str(show_lr(optim_video)))
            msg = msg + str(',best_acc={:2f}'.format(best_acc))
            print(msg)

            # if i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0):

            tot_iter += 1
            break
        
        # Evaluation
        acc, msg = test()
        if acc > best_acc:
            savename = '{}_iter_{}_epoch_{}_{}.pt'.format(args.save_prefix, tot_iter, epoch, msg)

            temp = os.path.split(savename)[0]
            if(not os.path.exists(temp)):
                os.makedirs(temp)
            torch.save({'video_model': video_model.state_dict()}, savename)

        if tot_iter != 0:
            best_acc = max(acc, best_acc)

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--gpus', type=str, default='0',
                        help='GPUs would be used. e.g 0,1,2,3')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=400,
                        help='Batch size')
    parser.add_argument('--n_class', type=int, required=True, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--max_epoch', type=int, default=120,
                        help='Number of epochs')
    parser.add_argument('--test', action='store_true', help='Testing mode')

    # load opts
    parser.add_argument('--weights', type=str, required=False, default=None,
                        help='Path to pretrained weights')

    # save prefix
    parser.add_argument('--save_prefix', type=str, default='checkpoints',
                        help='Path to checkpoint directory')

    # dataset
    parser.add_argument('--dataset', type=str, default='av', help='Dataset name')
    parser.add_argument('--data-dir', type=str, default='datasets/avletters_digits_npy_gray_pkl_jpeg',
                        help='Path to .pkl dataset files')
    parser.add_argument('--border', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--label_smooth', action='store_true')
    parser.add_argument('--se', action='store_true')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if(args.dataset == 'lrw'):
        from utils import LRWDataset as Dataset
    elif(args.dataset == 'lrw1000'):
        from utils import LRW1000_Dataset as Dataset
    elif args.dataset == 'av':
        from utils import AVDataset as Dataset
    else:
        raise Exception('lrw or lrw1000')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_cuda = torch.cuda.is_available()
    video_model = VideoModel(args).to(device)

    if is_cuda:
        lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
    else:
        lr = args.lr
    optim_video = optim.Adam(video_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max=args.max_epoch, eta_min=5e-6)

    if args.weights is not None:
        print('load weights')
        weight = torch.load(args.weights, map_location=torch.device('cpu'))
        load_missing(video_model, weight.get('video_model'))


    if is_cuda:
        video_model = parallel_model(video_model)

    torch.backends.cudnn.benchmark = True

    if args.test:
        acc, msg = test()
        print(f'acc={acc}')
        exit()
    train()
