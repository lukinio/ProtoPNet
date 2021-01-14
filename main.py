import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Compose, Lambda
import torchvision.datasets as datasets
from torchvision.datasets import MNIST, CIFAR10
from mnist_data_loader import MnistBags
from plot_logs import plot_logs, plot_conf_matrix

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from bag_generator import bag_generator
from colon_dataset import ColonCancerBagsCross

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.is_available())

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
if "small" in base_architecture:
    base_architecture_type = "small_" + base_architecture_type

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
makedir(model_dir+"logs/")
makedir(model_dir+"logs/conf_matrix/")
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean, std=std)


# transformation = transforms.Compose([
#     transforms.ToTensor(),
#     Lambda(lambda x: x.repeat(3, 1, 1) )
# ])

split_val = 70
train_range, test_range = range(split_val), range(split_val, 100)

ds = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range, test_idxs=test_range)            
ds_push = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range, test_idxs=test_range, push=True)            
ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range, test_idxs=test_range)

# ds = MnistBags(train=True)            
# ds_test = MnistBags(train=False)

# all datasets
# train set
train_loader = torch.utils.data.DataLoader(
    ds, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# push set
train_push_loader = torch.utils.data.DataLoader(
    ds_push, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
test_loader = torch.utils.data.DataLoader(
    ds_test, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=False, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
ACCURACY = 0.9 # over this accuracy save model
train_total_loss, train_acc = [], []
train_cross_ent, train_cluster_cost, train_sep_cost = [], [], []

test_total_loss, test_acc = [], []
test_cross_ent, test_cluster_cost, test_sep_cost = [], [], []

push_total_loss, push_acc = [], []
push_cross_ent, push_cluster_cost, push_sep_cost = [], [], []
import copy
from sklearn.metrics import plot_confusion_matrix
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        # joint_lr_scheduler.step() # Move after optimizer.step()
        tmp = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
        joint_lr_scheduler.step()
        # update logs
        train_acc.append(tmp[0])
        train_cross_ent.append(tmp[1])
        train_cluster_cost.append(tmp[2])
        train_sep_cost.append(tmp[3])
        train_total_loss.append(tmp[1]+tmp[2]+tmp[3])
        

    tmp = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    # update logs
    accu = tmp[0]
    test_acc.append(accu)
    test_cross_ent.append(tmp[1])
    test_cluster_cost.append(tmp[2])
    test_sep_cost.append(tmp[3])
    test_total_loss.append(tmp[1]+tmp[2]+tmp[3])

    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=ACCURACY, log=log)
    
    with torch.no_grad():
        TP, FP = 0, 0
        FN, TN = 0, 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = ppnet(inputs)
            predicted = torch.argmax(outputs, dim=1)

            TP += (predicted == targets == 1).sum().item()
            FP += (predicted == 1 and predicted != targets).sum().item()
            TN += (predicted == targets == 0).sum().item()
            FN += (predicted == 0 and predicted != targets).sum().item()

        plot_conf_matrix([[TP, FP], [FN, TN]],
                         epoch, model_dir+"logs/conf_matrix/")
        

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        tmp = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        # update logs
        accu = tmp[0]
        push_acc.append(tmp[0])
        push_cross_ent.append(tmp[1])
        push_cluster_cost.append(tmp[2])
        push_sep_cost.append(tmp[3])
        push_total_loss.append(tmp[1]+tmp[2]+tmp[3])
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=ACCURACY, log=log)

        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(10):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                tmp = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                accu = tmp[0]
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=ACCURACY, log=log)


plot_logs(model_dir+"logs/train_", train_acc, train_total_loss, train_cross_ent, train_cluster_cost, train_sep_cost)
plot_logs(model_dir+"logs/push_", push_acc, push_total_loss, push_cross_ent, push_cluster_cost, push_sep_cost)
plot_logs(model_dir+"logs/test_", test_acc, test_total_loss, test_cross_ent, test_cluster_cost, test_sep_cost)
logclose()

