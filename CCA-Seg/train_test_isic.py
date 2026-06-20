import sys
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_isic import NPY_datasets, RandomGenerator
from utils import DiceLoss, test_single_volume, Contrast_loss
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datetime import datetime

# from utils import contrast_loss_p2c_label

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../data/ACDC', help='root dir for data')
parser.add_argument('--dataset', type=str, default='isic18', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_acdc', help='list dir')
parser.add_argument('--num_classes', type=int, default=5, help='output channel of network')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument("--save_path", default="./checkpoint/ACDC/best_unet")

parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

#########################
## swav specific params #
#########################
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--threshold", default=0.5, type=float, help=" ")

args = parser.parse_args()

def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    msk_pred = np.argmax(msk_pred, axis=1)

    msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
    # if datasets == 'retinal':
    #     msk = np.squeeze(msk, axis=0)
    #     msk_pred = np.squeeze(msk_pred, axis=0)
    # else:
    #     msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
    #     msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()

def generate_multiscale_labels(original_labels):
    
    labels_28 = F.interpolate(original_labels, size=(32, 32), mode='bilinear',align_corners=True)
    labels_56 = F.interpolate(original_labels, size=(64, 64), mode='bilinear',align_corners=True)
    labels_112 = F.interpolate(original_labels, size=(128, 128), mode='bilinear',align_corners=True)
    return labels_28, labels_56, labels_112

# from MTNet
# def inference(args, model, val_loader, test_save_path=None):
# #     logging.info("{} test iterations per epoch".format(len(val_loader)))
# #     model.eval()
# #     metric_list = 0.0
# #     with torch.no_grad():
# #         for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
# #             h, w = sampled_batch["image"].size()[2:]
# #             image, label = sampled_batch["image"], sampled_batch["label"]
# #             # image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
# #             metric_i = test_single_volume(image, label, model, classes=args.num_classes,
# #                                           patch_size=[args.img_size, args.img_size],
# #                                           test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
# #             metric_list += np.array(metric_i)
# #             logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
# #                 i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
# #         metric_list = metric_list / len(val_loader)
# #         for i in range(1, args.num_classes):
# #             logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
# #         performance = np.mean(metric_list, axis=0)[0]
# #         mean_hd95 = np.mean(metric_list, axis=0)[1]
# #         logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
# #         logging.info("Testing Finished!")
# #         return performance, mean_hd95

def inference(args, model, val_loader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(test_loader)))
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(val_loader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label = sampled_batch["image"], sampled_batch["label"]
            # image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(val_loader)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance, mean_hd95

def test_one_epoch(args, test_loader, model, test_data_name=None):
    # switch to evaluate mode

    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for i, sampled_batch in enumerate(tqdm(test_loader)):
            img, msk = sampled_batch['image'], sampled_batch['label']
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            if i <= 500 and i % 100 == 0:
                save_imgs(img, msk, out, i,
                          'results/' + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/' + 'outputs/',
                          args.dataset, args.threshold, test_data_name=test_data_name)
            if i > 500 and i % 50 == 0:
                    save_imgs(img, msk, out, i, 'results/' + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/' + 'outputs/', args.dataset, args.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=args.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0



        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logging.info(log_info)
        log_info = f'test of best model, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logging.info(log_info)

    return miou, f1_or_dsc, accuracy


def trainer_tester_ISIC(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    Test_Accuracy = []
    Best_Accuracy = []
    # max_iterations = args.max_iterations
    # db_train = acdc_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    #                         transform=transforms.Compose(
    #                             [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # print("The length of train set is: {}".format(len(db_train)))

    # def worker_init_fn(worker_id):
    #     random.seed(args.seed + worker_id)

    train_dataset = NPY_datasets(args.root_path, args, train=True)
    print("The length of train set is: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0)
    val_dataset = NPY_datasets(args.root_path, args, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=1,
                            drop_last=True)

    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    #                          worker_init_fn=worker_init_fn, drop_last=True)
    # db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.cuda()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    Best_dcs = 0.0
    Best_acc = 0.0

    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.float().cuda(), label_batch.cuda()
            outputs, muti_logits_list, out = model(image_batch)

            # ============ swav loss ... ============
            noise1 = torch.clamp(torch.randn_like(out['pred_masks']) * 0.1, -0.2, 0.2)
            noise2 = torch.clamp(torch.randn_like(out['pred_masks']) * 0.1, -0.2, 0.2)
            enhanced_feature1 = out['pred_masks'] + noise1#[B, 32, 224, 224]
            enhanced_feature2 = out['pred_masks'] + noise2
            # cluster assignment prediction
            Q1 = torch.einsum("bnk,bnhw->bnhw", out['pred_logits'], enhanced_feature1)#[B,32,224,224]
            Q2 = torch.einsum("bnk,bnhw->bnhw", out['pred_logits'], enhanced_feature2)# B N H W
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
            loss_contrast = 0.5*loss_contrast1 + 0.5*loss_contrast2
            multiscale_labels = generate_multiscale_labels(label_batch.float())   #label_batch(4,1,256,256) # torch.Size([1, 2, 224, 224])
            loss_ce = 0.8*ce_loss(outputs, label_batch[:].squeeze(1).long())#outputs [B,4,224,224]; label_batch [4,224,224]
            loss_dice = 0.8*dice_loss(outputs, label_batch.squeeze(1), softmax=True)
            loss_weight = [0.05, 0.05, 0.1]
            for i in range(3):
                loss_ce += loss_weight[i] * ce_loss(muti_logits_list[i], multiscale_labels[i][:].squeeze(1).squeeze(1).long())
                loss_dice += loss_weight[i] * dice_loss(muti_logits_list[i], multiscale_labels[i].squeeze(1), softmax=True)
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

            # if loss.detach() < Best_loss:
            #     Best_loss = loss.detach()
            #     avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
            #     Test_Accuracy.append((epoch_num, avg_dcs, avg_hd))
            #     logging.info("Test_Accuracy: {}".format(Test_Accuracy))
            #     save_bestLossMode_path = os.path.join(snapshot_path, 'Best_iter_num_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_bestLossMode_path)

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs.squeeze(1), iter_num) #labs [1,224,224]

        # save_interval = 50  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

        if epoch_num == 1:
            avg_miou, avg_dsc, avg_acc = test_one_epoch(args, val_loader, model)
            # avg_dcs, avg_hd = inference(args, model, val_loader, test_save_path=None)
            print(avg_miou, avg_dsc, avg_acc)

        if epoch_num == 149:
            # avg_dcs, avg_hd = inference(args, model, val_loader, test_save_path=None)
            avg_miou, avg_dsc, avg_acc = test_one_epoch(args, val_loader, model)
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            print('epoch149', end="")
            print(avg_miou, avg_dsc, avg_acc)
        # if epoch_num == 299:
        #     avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     print('epoch299', end="")
        #     print(avg_dcs, avg_hd)
        if 400 > epoch_num > 200 and (epoch_num + 1) % 20 == 0:
            avg_miou, avg_dsc, avg_acc = test_one_epoch(args, val_loader, model)
            if avg_acc > Best_acc or avg_dsc > Best_dcs :
                save_mode_path = os.path.join(args.save_path,
                                              'epoch={}_lr={}_avg_miou={}_avg_dsc={}_avg_acc={}.pth'.format(epoch_num, lr_, avg_miou, avg_dsc, avg_acc))
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                # temp = 1
                Best_dcs = avg_dsc
                Best_Accuracy.append((epoch_num, Best_dcs, Best_acc))
            # else:
            #     if os.path.exists(save_mode_path):
            #         model.load_state_dict(torch.load(save_mode_path))
            Test_Accuracy.append((epoch_num, avg_dsc, avg_acc))

        if epoch_num > 700 and (epoch_num + 1) % 10 == 0:
            avg_miou, avg_dsc, avg_acc = test_one_epoch(args, val_loader, model)
            if avg_dsc > Best_dcs:
                save_mode_path = os.path.join(args.save_path,
                                              'epoch={}_lr={}_avg_miou={}_avg_dsc={}_avg_acc={}.pth'.format(epoch_num, lr_, avg_miou, avg_dsc, avg_acc))
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                # temp = 1
                Best_dcs = avg_dsc
                Best_Accuracy.append((epoch_num, Best_dcs, avg_acc))
            else:
                if os.path.exists(save_mode_path) and epoch_num > 800:
                    model.load_state_dict(torch.load(save_mode_path))
            Test_Accuracy.append((epoch_num, avg_dsc, avg_acc))
    logging.info("Test_Accuracy: {}".format(Test_Accuracy))
    logging.info("Best_Accuracy: {}".format(Best_Accuracy))
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'isic17': {
            'Dataset': NPY_datasets,
            'root_path': '../data/isic2017/',
            'num_classes': 2,
            'z_spacing': 1,
        },
        'isic18': {
            'Dataset': NPY_datasets,
            'root_path': '../data/isic2018/',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.is_pretrain = True

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))
    # net.cuda()
    trainer_tester = {'isic18': trainer_tester_ISIC, 'isic17': trainer_tester_ISIC,}
    trainer_tester[dataset_name](args, net, snapshot_path)

