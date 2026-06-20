import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import torch.nn.functional as F
from datasets.dataloader import test_dataset

def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        # eval Dice
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

def generate_multiscale_labels(original_labels):
    labels_28 = F.interpolate(original_labels, size=(28, 28), mode='bilinear', align_corners=True).squeeze(1)
    labels_56 = F.interpolate(original_labels, size=(56, 56), mode='bilinear', align_corners=True).squeeze(1)
    labels_112 = F.interpolate(original_labels, size=(112, 112), mode='bilinear', align_corners=True).squeeze(1)
    return labels_28, labels_56, labels_112

def trainer_polyp(args, model, snapshot_path):
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    # from datasets.dataset_polyp import polyp_dataset, RandomGenerator
    from datasets.dataloader import PolypDataset
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = PolypDataset(image_root=args.img_path, gt_root=args.gt_path, trainsize=args.img_size)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda() #[B,3,224,224] [B,1,224,224]
            outputs, muti_logits_list, out = model(image_batch)

            # ============ swav loss ... ============
            noise1 = torch.clamp(torch.randn_like(out['pred_masks']) * 0.1, -0.2, 0.2)
            noise2 = torch.clamp(torch.randn_like(out['pred_masks']) * 0.1, -0.2, 0.2)
            enhanced_feature1 = out['pred_masks'] + noise1  # [B, 32, 224, 224]
            enhanced_feature2 = out['pred_masks'] + noise2
            # cluster assignment prediction
            Q1 = torch.einsum("bnk,bnhw->bnhw", out['pred_logits'], enhanced_feature1)  # [B,32,224,224]
            Q2 = torch.einsum("bnk,bnhw->bnhw", out['pred_logits'], enhanced_feature2)  # B N H W
            similarity_scores1 = F.cosine_similarity(enhanced_feature1, Q2, dim=1)
            similarity_scores2 = F.cosine_similarity(enhanced_feature2, Q1, dim=1)
            noise_similarity_scores = F.cosine_similarity(enhanced_feature1, out['pred_masks'], dim=1)
            noise_similarity_scores2 = F.cosine_similarity(enhanced_feature2, out['pred_masks'], dim=1)
            loss_contrast1 = -torch.log(torch.exp(similarity_scores1 / args.temperature) / (
                    torch.exp(similarity_scores1 / args.temperature) + torch.exp(
                noise_similarity_scores / args.temperature))).mean()
            loss_contrast2 = -torch.log(torch.exp(similarity_scores2 / args.temperature) / (
                    torch.exp(similarity_scores2 / args.temperature) + torch.exp(
                noise_similarity_scores2 / args.temperature))).mean()
            loss_contrast = 0.5 * loss_contrast1 + 0.5 * loss_contrast2

            multiscale_labels = generate_multiscale_labels(label_batch.float())  # torch.Size([B, 1, 224, 224])
            loss_ce = 0.8 * ce_loss(outputs, label_batch[:].squeeze(1).long())# outputs[B,2,224,224] label_batch[2,224,224]
            loss_dice = 0.8 * dice_loss(outputs, label_batch.squeeze(1), softmax=True)
            loss_weight = [0.05, 0.05, 0.1]
            for i in range(3):
                loss_ce += loss_weight[i] * ce_loss(muti_logits_list[i], multiscale_labels[i][:].long())
                loss_dice += loss_weight[i] * dice_loss(muti_logits_list[i], multiscale_labels[i], softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.01 * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_contrast: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_contrast.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch.squeeze(1)
                labs = labs[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        if (epoch_num + 1) % 1 == 0:
            for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                dataset_dice = test(model, snapshot_path, dataset)
                logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch_num, dataset, dataset_dice))
                print(dataset, ': ', dataset_dice)
                dict_plot[dataset].append(dataset_dice)
            meandice = test(model, snapshot_path, 'test')
            dict_plot['test'].append(meandice)
            if meandice > best:
                best = meandice
                torch.save(model.state_dict(), snapshot_path + 'Polyp.pth')
                torch.save(model.state_dict(), snapshot_path + str(epoch_num) + 'Polyp-best.pth')
                print('##############################################################################best', best)
                logging.info(
                    '##############################################################################best:{}'.format(
                        best))

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #
        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return "Training Finished!"