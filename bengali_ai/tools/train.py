import _init_paths
import time
from typing import Dict, Union
import numpy as np
import pdb
import random
import argparse
import torch
import os
import math
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from utils import transform
from config import config
from config import update_config
from utils.avg_meter import AverageMeter, AccuracyMeter
from utils.utils import create_logger, save_checkpoint
from utils.training_utils import poly_learning_rate, poly_learning_rate2
from utils.normalization_utils import get_imagenet_mean_std
from utils.dataloader import BongLoader
from tqdm import tqdm
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Bengali OCR')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')

    args = parser.parse_args()
    assert args.config is not None
    update_config(config, args)
    return args


def main():
    print('Using PyTorch version: ', torch.__version__)
    args = get_parser()
    main_worker(args)


def main_worker(args):
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    BatchNorm = nn.BatchNorm2d
    print('Using batchnorm variant: ', BatchNorm)

    model = get_model()
    optimizer = get_optimizer(model)

    # iou = config.TEST.IOU
    # channel_count = config.MODEL.NUM_CLASSES
    # img_conv_th = config.TEST.IMAGE_CONVERSION_TH
    # device = config.DEVICE
    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH

    logger, final_output_dir, tb_log_dir = create_logger(config, args.config, 'train')

    logger.info(config)
    logger.info("=> creating model ...")
    if config.TRAIN.PRETRAINED_BACKBONE:
        logger.info("=> using pretrained backbone")
    logger.info("CONSONANT Classes: {}".format(config.MODEL.NUM_CLASSES_CONSONANT))
    logger.info("ROOT Classes: {}".format(config.MODEL.NUM_CLASSES_ROOT))
    logger.info("VOWEL Classes: {}".format(config.MODEL.NUM_CLASSES_VOWEL))
    logger.info(model)

    print(config)
    print("=> creating model ...")
    if config.TRAIN.PRETRAINED_BACKBONE:
        print("=> using pretrained backbone")
    print("CONSONANT Classes: {}".format(config.MODEL.NUM_CLASSES_CONSONANT))
    print("ROOT Classes: {}".format(config.MODEL.NUM_CLASSES_ROOT))
    print("VOWEL Classes: {}".format(config.MODEL.NUM_CLASSES_VOWEL))
    print(model)

    model = torch.nn.DataParallel(model.cuda(), device_ids=config.GPUS)

    if config.TRAIN.RESUME:
        if config.TRAIN.PRETRAINED_MODEL:
            model_state_file = config.TRAIN.PRETRAINED_MODEL
            if os.path.isfile(model_state_file):
                checkpoint = torch.load(model_state_file)
                # print(checkpoint.module.keys())
                # last_epoch = checkpoint['epoch']
                # best_perf = checkpoint['perf']
                # model.module.load_state_dict(checkpoint['state_dict'])
                model = checkpoint.module
                model = torch.nn.DataParallel(model.cuda())
                # optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded saved model")
                print("=> loaded saved model")
                # best_model = True
            else:
                logger.info("=> no checkpoint found. " 'Running from Scratch')
                print("=> no checkpoint found. " 'Running from Scratch')
        else:
            model_state_file = os.path.join(final_output_dir,
                                            'checkpoint.pth.tar')
            if os.path.isfile(model_state_file):
                checkpoint = torch.load(model_state_file)
                last_epoch = checkpoint['epoch']
                best_perf = checkpoint['perf']
                model.module.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint (epoch {})"
                            .format(checkpoint['epoch']))
                print("=> loaded checkpoint (epoch {})"
                      .format(checkpoint['epoch']))
                best_model = True
            else:
                logger.info("=> no checkpoint found. " 'Running from Scratch')
                print("=> no checkpoint found. " 'Running from Scratch')

    train_indices_file = config.DATASET.TRAIN_SET
    val_indices_file = config.DATASET.VAL_SET

    mean, std = get_imagenet_mean_std()

    # transform.ResizeTest((config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
    # transform.RandScale([config.TRAIN.SCALE_MIN, config.TRAIN.SCALE_MAX]),
    # ###
    # transform.Crop(
    #     [config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W],
    #     crop_type="rand",
    #     padding=mean,
    #     ignore_label=config.TRAIN.IGNOE_LABEL,
    # ),

    train_transform_list = [
        transform.ResizeTest((config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
        transform.RandScale([config.TRAIN.SCALE_MIN, config.TRAIN.SCALE_MAX]),
        transform.RandRotate(
            [config.TRAIN.ROTATE_MIN, config.TRAIN.ROTATE_MAX],
            padding=1.0,
            ignore_label=config.TRAIN.IGNOE_LABEL,
        ),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop(
                [config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W],
                crop_type="rand",
                padding=1.0,
                ignore_label=config.TRAIN.IGNOE_LABEL,
            ),
        transform.ToTensor(),
    ]

    # transform.ResizeTest((config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
    # transform.Crop(
    #     [config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W],
    #     crop_type="center",
    #     padding=mean,
    #     ignore_label=config.TRAIN.IGNOE_LABEL,
    # ),

    val_transform_list = [
        transform.ResizeTest((config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W)),
        transform.Crop(
            [config.TRAIN.TRAIN_H, config.TRAIN.TRAIN_W],
            crop_type="center",
            padding=1.0,
            ignore_label=config.TRAIN.IGNOE_LABEL,
        ),
        transform.ToTensor(),
    ]

    if config.DATASET.DATASET == 'KAGGLE':
        train_dataset = BongLoader(
            output_size=(config.DATASET.IMAGE_H, config.DATASET.IMAGE_W),
            dataset_root=config.DATASET.ROOT,
            image_format=config.DATASET.DATA_FORMAT,
            json_file=config.DATASET.JSON_CLASS_MAP,
            indices_file=train_indices_file,
            normalizer=transform.Compose(train_transform_list)
        )
        val_dataset = BongLoader(
            output_size=(config.DATASET.IMAGE_H, config.DATASET.IMAGE_W),
            dataset_root=config.DATASET.ROOT,
            image_format=config.DATASET.DATA_FORMAT,
            json_file=config.DATASET.JSON_CLASS_MAP,
            indices_file=val_indices_file,
            normalizer=transform.Compose(val_transform_list)
        )
    else:
        train_dataset = None
        val_dataset = None
        logger.info("=> no dataset found. " 'Exiting...')
        print("=> no dataset found. " 'Exiting...')
        exit()

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    logger.info(f'Train loader has len {len(train_loader)}')
    print(f'Train loader has len {len(train_loader)}')

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=val_sampler
    )

    for ep_it in range(last_epoch, config.TRAIN.END_EPOCH):
        epoch = ep_it + 1
        logger.info(f'New epoch {epoch} starts on cuda')
        print(f'New epoch {epoch} starts on cuda')

        train_accuracy, train_loss = train(train_loader, model, optimizer,
                                           epoch, logger)

        val_accuracy, val_loss = validate(val_loader, model, logger)

        if val_accuracy > best_perf:
            best_perf = val_accuracy
            best_model = True
        else:
            best_model = False

        if epoch % config.TRAIN.SAVE_FREQ == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            print('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'perf': val_accuracy,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir, filename='checkpoint.pth.tar')

        if best_model:
            logger.info('=> saving best model to {}'.format(final_output_dir))
            print('=> saving best model to {}'.format(final_output_dir))
            torch.save(
                {
                    'epoch': epoch,
                    'perf': val_accuracy,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(final_output_dir, 'epoch_%02d_acc_%0.2f_model.pth.tar' % (epoch, val_accuracy)))

        if epoch == config.TRAIN.END_EPOCH:
            filename = final_output_dir + '/train_epoch_final.pth'
            logger.info('Saving checkpoint to: ' + filename)
            print('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict()}, filename)
            exit()


def train(train_loader, model, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()

    model.train()

    end = time.time()

    for i, (image, c_head, r_head, v_head) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # i_c_gt = [int(torch.argmax(c)) for c in c_head]
        # i_r_gt = [int(torch.argmax(r)) for r in r_head]
        # i_v_gt = [int(torch.argmax(v)) for v in v_head]

        image = image.cuda()
        c_head = c_head.cuda()
        r_head = r_head.cuda()
        v_head = v_head.cuda()

        pred_c, pred_r, pred_v = model(image)

        # print(pred_c, pred_r, pred_v)
        # ###
        # i_c_pred = [int(torch.argmax(output_c)) for output_c in pred_c]
        # i_r_pred = [int(torch.argmax(output_r)) for output_r in pred_r]
        # i_v_pred = [int(torch.argmax(output_v)) for output_v in pred_v]

        i_c_pred = pred_c.data.max(1, keepdim=True)[1]
        i_r_pred = pred_r.data.max(1, keepdim=True)[1]
        i_v_pred = pred_v.data.max(1, keepdim=True)[1]

        i_c_pred = i_c_pred.squeeze(1)
        i_r_pred = i_r_pred.squeeze(1)
        i_v_pred = i_v_pred.squeeze(1)

        # print(i_c_gt, i_r_gt, i_v_gt)
        # print(i_c_pred.detach().cpu().numpy(), i_r_pred.detach().cpu().numpy(), i_v_pred.detach().cpu().numpy())
        #
        # print(pred_c.shape, c_head.shape)
        # print(pred_r.shape, r_head.shape)
        # print(pred_v.shape, v_head.shape)

        loss_c = F.nll_loss(pred_c, c_head)
        loss_r = F.nll_loss(pred_r, r_head)
        loss_v = F.nll_loss(pred_v, v_head)
        loss = loss_c + loss_r + loss_v

        acc_meter.update(c_head.detach().cpu().numpy(),
                         r_head.detach().cpu().numpy(),
                         v_head.detach().cpu().numpy(),
                         i_c_pred.detach().cpu().numpy(),
                         i_r_pred.detach().cpu().numpy(),
                         i_v_pred.detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = image.size(0)

        loss_meter.update(loss.item(), n)

        if i > 0:
            batch_time.update(time.time() - end)
        end = time.time()

        current_iter = (epoch-1) * len(train_loader) + i + 1
        max_iter = config.TRAIN.END_EPOCH * len(train_loader)
        index_split = -1

        if config.TRAIN.BASE_LR > 1e-6:
            optimizer = poly_learning_rate(optimizer, config.TRAIN.BASE_LR,
                                           current_iter, max_iter, power=config.TRAIN.POWER)
            # optimizer = poly_learning_rate2(optimizer, config.TRAIN.BASE_LR, current_iter, max_iter,
            #                                 power=config.TRAIN.POWER, index_split=index_split,
            #                                 warmup=config.TRAIN.WARMUP,
            #                                 warmup_step=len(train_loader) // 2)

        # optimizer.param_groups[0]['lr'] = current_lr

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if current_iter % config.TRAIN.PRINT_FREQ == 0 and True:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'LR {current_lr:.8f} '
                        'Loss {loss_meter.val:.4f} '
                        'Consonant Accuracy {accuracy_c:.4f}. '
                        'Root Accuracy {accuracy_r:.4f}. '
                        'Vowel Accuracy {accuracy_v:.4f}. '
                        'All Accuracy {accuracy:.4f}. '.format(epoch, config.TRAIN.END_EPOCH,
                                                               i + 1, len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time,
                                                               remain_time=remain_time,
                                                               current_lr=get_lr(optimizer),
                                                               loss_meter=loss_meter,
                                                               accuracy_c=acc_meter.consonant_accuracy.avg,
                                                               accuracy_r=acc_meter.root_accuracy.avg,
                                                               accuracy_v=acc_meter.vowel_accuracy.avg,
                                                               accuracy=acc_meter.accuracy_avg.avg) +
                        f'current_iter: {current_iter}')
            print('Epoch: [{}/{}][{}/{}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Remain {remain_time} '
                  'LR {current_lr:.8f} '
                  'Loss {loss_meter.val:.4f} '
                  'Consonant Accuracy {accuracy_c:.4f}. '
                  'Root Accuracy {accuracy_r:.4f}. '
                  'Vowel Accuracy {accuracy_v:.4f}. '
                  'All Accuracy {accuracy:.4f}. '.format(epoch, config.TRAIN.END_EPOCH,
                                                         i + 1, len(train_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         remain_time=remain_time,
                                                         current_lr=get_lr(optimizer),
                                                         loss_meter=loss_meter,
                                                         accuracy_c=acc_meter.consonant_accuracy.avg,
                                                         accuracy_r=acc_meter.root_accuracy.avg,
                                                         accuracy_v=acc_meter.vowel_accuracy.avg,
                                                         accuracy=acc_meter.accuracy_avg.avg) +
                  f'current_iter: {current_iter}')

    logger.info(
        'Train result at epoch [{}/{}]: allAcc {:.4f}.'.format(epoch,
                                                               config.TRAIN.END_EPOCH,
                                                               acc_meter.accuracy_avg.avg))
    print(
        'Train result at epoch [{}/{}]: allAcc {:.4f}.'.format(epoch,
                                                               config.TRAIN.END_EPOCH,
                                                               acc_meter.accuracy_avg.avg))
    return acc_meter.accuracy_avg.avg, loss_meter.avg


def get_model():
    if config.TRAIN.ARCH == "bongnet":
        from models.bongnet import BongNet
        model = BongNet(layers=config.TRAIN.LAYERS, dropout=0.1, c_heads=config.MODEL.NUM_CLASSES_CONSONANT,
                        r_heads=config.MODEL.NUM_CLASSES_ROOT, v_heads=config.MODEL.NUM_CLASSES_VOWEL,
                        pretrained=config.TRAIN.PRETRAINED_BACKBONE, resnet_path=config.TRAIN.RESNET_PRETRAINED_MODEL)
        return model
    else:
        print("No {a} Model Found".format(a=config.TRAIN.ARCH))
        exit()


def get_optimizer(model):
    if config.TRAIN.ARCH == "bongnet":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": filter(lambda p: p.requires_grad, model.parameters()),
                    "lr": config.TRAIN.BASE_LR,
                }
            ],
            lr=config.TRAIN.BASE_LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        return optimizer


def validate(val_loader, model, logger):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AccuracyMeter()

    model.eval()

    end = time.time()
    for i, (image, c_head, r_head, v_head) in enumerate(val_loader):
        data_time.update(time.time() - end)

        image = image.cuda(non_blocking=True)
        c_head = c_head.cuda(non_blocking=True)
        r_head = r_head.cuda(non_blocking=True)
        v_head = v_head.cuda(non_blocking=True)

        pred_c, pred_r, pred_v = model(image)

        # print(pred_c)

        i_c_pred = pred_c.data.max(1, keepdim=True)[1]
        i_r_pred = pred_r.data.max(1, keepdim=True)[1]
        i_v_pred = pred_v.data.max(1, keepdim=True)[1]

        i_c_pred = i_c_pred.squeeze(1)
        i_r_pred = i_r_pred.squeeze(1)
        i_v_pred = i_v_pred.squeeze(1)

        # print(i_c_gt, i_r_gt, i_v_gt)
        # print(i_c_pred, i_r_pred, i_v_pred)

        # print(pred_c.shape, c_head.shape)
        # print(pred_r.shape, r_head.shape)
        # print(pred_v.shape, v_head.shape)

        loss_c = F.nll_loss(pred_c, c_head)
        loss_r = F.nll_loss(pred_r, r_head)
        loss_v = F.nll_loss(pred_v, v_head)
        loss = loss_c + loss_r + loss_v

        acc_meter.update(c_head.detach().cpu().numpy(),
                         r_head.detach().cpu().numpy(),
                         v_head.detach().cpu().numpy(),
                         i_c_pred.detach().cpu().numpy(),
                         i_r_pred.detach().cpu().numpy(),
                         i_v_pred.detach().cpu().numpy())

        n = image.size(0)
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % config.TRAIN.PRINT_FREQ == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Consonant Accuracy {accuracy_c:.4f}. '
                        'Root Accuracy {accuracy_r:.4f}. '
                        'Vowel Accuracy {accuracy_v:.4f}. '
                        'All Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy_c=acc_meter.consonant_accuracy.avg,
                                                              accuracy_r=acc_meter.root_accuracy.avg,
                                                              accuracy_v=acc_meter.vowel_accuracy.avg,
                                                              accuracy=acc_meter.accuracy_avg.avg))
            print('Test: [{}/{}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                  'Consonant Accuracy {accuracy_c:.4f}. '
                  'Root Accuracy {accuracy_r:.4f}. '
                  'Vowel Accuracy {accuracy_v:.4f}. '
                  'All Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                        data_time=data_time,
                                                        batch_time=batch_time,
                                                        loss_meter=loss_meter,
                                                        accuracy_c=acc_meter.consonant_accuracy.avg,
                                                        accuracy_r=acc_meter.root_accuracy.avg,
                                                        accuracy_v=acc_meter.vowel_accuracy.avg,
                                                        accuracy=acc_meter.accuracy_avg.avg))

    logger.info('Val result: allAcc {:.4f}.'.format(acc_meter.accuracy_avg.avg))
    print('Val result: allAcc {:.4f}.'.format(acc_meter.accuracy_avg.avg))

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return acc_meter.accuracy_avg.avg, loss_meter.avg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    main()
