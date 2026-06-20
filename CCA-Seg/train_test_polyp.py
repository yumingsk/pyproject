import sys
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_polyp import polyp_dataset, RandomGenerator
from utils import DiceLoss, test_single_volume, Contrast_loss
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

# from utils import contrast_loss_p2c_label

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../data/ACDC', help='root dir for data')
parser.add_argument('--dataset', type=str, default='polyp', help='experiment_name')
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
                    default=224, help='input patch size of network input')
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

args = parser.parse_args()


def generate_multiscale_labels(original_labels):
    labels_28 = F.interpolate(original_labels, size=(28, 28), mode='bilinear', align_corners=True).squeeze(1)
    labels_56 = F.interpolate(original_labels, size=(56, 56), mode='bilinear', align_corners=True).squeeze(1)
    labels_112 = F.interpolate(original_labels, size=(112, 112), mode='bilinear', align_corners=True).squeeze(1)
    return labels_28, labels_56, labels_112


# from MTNet
def inference(args, model, testloader, test_save_path=None):
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
                i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i - 1][0], metric_list[i - 1][1]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
        logging.info("Testing Finished!")
        return performance, mean_hd95


def trainer_tester_polyp(args, model, snapshot_path):
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
    db_train = polyp_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.cuda()
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
    Best_dcs = 0.0

    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
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

            multiscale_labels = generate_multiscale_labels(
                label_batch.unsqueeze(1).float())  # torch.Size([1, 2, 224, 224])
            loss_ce = 0.8 * ce_loss(outputs, label_batch[:].long())
            loss_dice = 0.8 * dice_loss(outputs, label_batch, softmax=True)
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

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_contrast: %f' % (
            iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_contrast.item()))

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
                writer.add_image('train/GroundTruth', labs, iter_num)

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

        # if epoch_num == 1:
        #     avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
        #     print(avg_dcs, avg_hd)

        if epoch_num == 149:
            avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            print('epoch149', end="")
            print(avg_dcs, avg_hd)
        # if epoch_num == 299:
        #     avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     print('epoch299', end="")
        #     print(avg_dcs, avg_hd)
        if 400 > epoch_num > 200 and (epoch_num + 1) % 20 == 0:
            avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
            if avg_dcs > Best_dcs:
                save_mode_path = os.path.join(args.save_path,
                                              'epoch={}_lr={}_avg_dcs={}_avg_hd={}.pth'.format(epoch_num, lr_, avg_dcs,
                                                                                               avg_hd))
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                # temp = 1
                Best_dcs = avg_dcs
                Best_Accuracy.append((epoch_num, Best_dcs, avg_hd))
            # else:
            #     if os.path.exists(save_mode_path):
            #         model.load_state_dict(torch.load(save_mode_path))
            Test_Accuracy.append((epoch_num, avg_dcs, avg_hd))

        if epoch_num > 700 and (epoch_num + 1) % 10 == 0:
            avg_dcs, avg_hd = inference(args, model, testloader, test_save_path=None)
            if avg_dcs > Best_dcs:
                save_mode_path = os.path.join(args.save_path,
                                              'epoch={}_lr={}_avg_dcs={}_avg_hd={}.pth'.format(epoch_num, lr_, avg_dcs,
                                                                                               avg_hd))
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                # temp = 1
                Best_dcs = avg_dcs
                Best_Accuracy.append((epoch_num, Best_dcs, avg_hd))
            else:
                if os.path.exists(save_mode_path) and epoch_num > 800:
                    model.load_state_dict(torch.load(save_mode_path))
            Test_Accuracy.append((epoch_num, avg_dcs, avg_hd))
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
        'polyp': {
            'Dataset': polyp_dataset,
            'root_path': '../data/polyp',
            'volume_path': '../data/polyp/test',
            'list_dir': './lists/lists_acdc',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
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
    trainer_tester = {'polyp': trainer_tester_polyp, }
    trainer_tester[dataset_name](args, net, snapshot_path)

