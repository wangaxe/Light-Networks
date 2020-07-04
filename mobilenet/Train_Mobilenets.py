import argparse
import time
import os
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR,StepLR
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from mobilenets import mobilenetV1


def load_CIFAR10(_dir, batch_size, valid_size, seed):
    np.random.seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=_dir, train=True, download=True,
                                    transform=train_transform)
    valid_dataset = datasets.CIFAR10(root=_dir, train=True, download=True,
                                    transform=valid_transform)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, sampler=valid_sampler)
    
    test_dataset = datasets.CIFAR10(root=_dir, train=False, download=True,
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                    batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

def evaluation(data_loader, model, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_acc = 0
    test_n = 0
    for data, label in data_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)

        test_loss += loss.item() * label.size(0)
        test_acc += (output.max(1)[1] == label).sum().item()
        test_n += label.size(0)
    return test_loss/test_n, test_acc/test_n

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../datasets/cifar-data', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'flat'])
    parser.add_argument('--lr-min', default=0.01, type=float)
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--valid-size', default=0.1, type=float)
    parser.add_argument('--device-id', default=0, type=int)
    parser.add_argument('--fname', default='test', type=str)
    parser.add_argument('--out-dir', default='output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

def main():
    args = get_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, args.fname+'.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfile,
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO)
    logger.info(args)
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, valid_loader, test_loader = load_CIFAR10(
                args.data_dir, args.batch_size, args.valid_size, args.seed)
    model = mobilenetV1(num_classes=10).to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max,
    #                         momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_steps = args.epochs * len(train_loader)
    # scheduler = CyclicLR(optimizer, base_lr=args.lr_min, max_lr=args.lr_max,
    #         step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    # Training
    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Valid Loss \t Valid Acc')
    total_time = 0
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * label.size(0)
            train_acc += (output.max(1)[1] == label).sum().item()
            train_n += label.size(0)

        end_epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        total_time += end_epoch_time - start_epoch_time
        valid_acc, valid_loss = evaluation(valid_loader, model, device)
        logger.info('%d \t\t %.1f \t\t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, end_epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n,
            valid_acc, valid_loss)

    logger.info('Total train time: %.4f minutes', (total_time)/60)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f'{args.fname}.pth'))

    # Evaluation
    # model_test = mobilenetV1(num_classes=10).to(device)
    # model_test.load_state_dict(model.state_dict())
    # model_test.float()
    test_acc, test_loss = evaluation(test_loader, model_test, device)
    logger.info('Test Loss \t Test Acc')
    logger.info('{:.4f} \t\t {:.4f}'.format(test_loss, test_acc))

if __name__ == "__main__":
    main()
