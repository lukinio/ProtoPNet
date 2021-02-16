import argparse
import datetime
import os
import platform
from enum import Enum

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import model
import push
import save
from datasets.colon_dataset import ColonCancerBagsCross
from datasets.mnist_dataset import MnistBags
from helpers import makedir
from preprocess import preprocess_input_function
from settings import base_architecture, prototype_shape, num_classes, \
    prototype_activation_function, add_on_layers_type, joint_optimizer_lrs, joint_lr_step_size, \
    last_layer_optimizer_lr, warm_optimizer_lrs, coefs, \
    num_train_epochs, num_warm_epochs, push_start, push_epochs, num_last_layer_iterations, class_specific
from train_and_test import warm_only, train, joint, test, last_only

# noinspection PyTypeChecker
parser = argparse.ArgumentParser(prog='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpuid', type=int, default=0, help='CUDA device id to use')
parser.add_argument('-d', '--dataset', type=str, default='colon_cancer', choices=['mnist', 'colon_cancer'],
                    help='Select dataset')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
print('CUDA available:', torch.cuda.is_available())

if args.dataset == 'colon_cancer':
    split_val = 70
    train_range, test_range = range(split_val), range(split_val, 100)

    ds = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range, test_idxs=test_range,
                              shuffle_bag=True)
    ds_push = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range,
                                   test_idxs=test_range,
                                   push=True, shuffle_bag=True)
    ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range,
                                   test_idxs=test_range)
    img_size = 27

    # to create bags with only one nucleus type set nucleus_type to 'epithelial', 'inflammatory', 'fibroblast' or 'others' as shown below
    # ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range, test_idxs=test_range, nucleus_type='epithelial')
    # ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range, test_idxs=test_range, nucleus_type='inflammatory')
    # ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range, test_idxs=test_range, nucleus_type='fibroblast')
    # ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range, test_idxs=test_range, nucleus_type='others')

elif args.dataset == 'mnist':
    ds = MnistBags(train=True)
    ds_push = MnistBags(train=True)
    ds_test = MnistBags(train=False)
    img_size = 28
else:
    raise NotImplementedError()

ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=False, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()

joint_optimizer_specs = [
    {
        'params': ppnet.features.parameters(),
        'lr': joint_optimizer_lrs['features'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.add_on_layers.parameters(),
        'lr': joint_optimizer_lrs['add_on_layers'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.prototype_vectors,
        'lr': joint_optimizer_lrs['prototype_vectors']
    }
]

warm_optimizer_specs = [
    {
        'params': ppnet.add_on_layers.parameters(),
        'lr': warm_optimizer_lrs['add_on_layers'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.prototype_vectors,
        'lr': warm_optimizer_lrs['prototype_vectors']
    },
]

last_layer_optimizer_specs = [
    {
        'params': ppnet.last_layer.parameters(),
        'lr': last_layer_optimizer_lr['last_layer']
    },
    {
        'params': ppnet.attention_V.parameters(),
        'lr': last_layer_optimizer_lr['attention']
    },
    {
        'params': ppnet.attention_U.parameters(),
        'lr': last_layer_optimizer_lr['attention']
    },
    {
        'params': ppnet.attention_weights.parameters(),
        'lr': last_layer_optimizer_lr['attention']
    }
]

joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# TODO: checkpoint logic

experiment_run = '{}.{}.{}'.format(args.dataset, platform.node(), datetime.datetime.now().isoformat())

model_dir = os.path.join('saved_models', experiment_run)
makedir(model_dir)
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)

log_writer = SummaryWriter(os.path.join('runs', experiment_run))

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

seed = torch.seed()
torch.manual_seed(seed)

log_writer.add_text('seed', str(seed))

# all datasets
# train set
train_loader = torch.utils.data.DataLoader(
    ds, batch_size=None, shuffle=True,
    num_workers=8,
    pin_memory=True)
# push set
train_push_loader = torch.utils.data.DataLoader(
    ds_push, batch_size=None, shuffle=False,
    num_workers=8,
    pin_memory=True)
# test set
test_loader = torch.utils.data.DataLoader(
    ds_test, batch_size=None, shuffle=False,
    num_workers=8,
    pin_memory=True)

# noinspection PyTypeChecker
log_writer.add_text('dataset_stats',
                    'training set size: {}, push set size: {}, test set size: {}'.format(
                        len(train_loader.dataset), len(train_push_loader.dataset), len(test_loader.dataset)))

# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)


# train the model
print('Training started')
ACCURACY = 0.9  # over this accuracy save model

test_total_loss, test_acc = [], []

step = -1


class TrainMode(Enum):
    WARM = 'warm'
    JOINT = 'joint'
    PUSH = 'push'
    LAST_ONLY = 'last_only'


def write_mode(mode: TrainMode, log_writer: SummaryWriter, step: int):
    log_writer.add_scalar('mode/warm', int(mode == TrainMode.WARM), global_step=step)
    log_writer.add_scalar('mode/joint', int(mode == TrainMode.JOINT), global_step=step)
    log_writer.add_scalar('mode/push', int(mode == TrainMode.PUSH), global_step=step)
    log_writer.add_scalar('mode/last_only', int(mode == TrainMode.LAST_ONLY), global_step=step)


for epoch in range(num_train_epochs):
    step += 1
    print('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        write_mode(TrainMode.WARM, log_writer, step)
        warm_only(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer,
              class_specific=class_specific, coefs=coefs, log_writer=log_writer, step=step)
    else:
        write_mode(TrainMode.JOINT, log_writer, step)
        joint(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer,
              class_specific=class_specific, coefs=coefs, log_writer=log_writer, step=step)
        joint_lr_scheduler.step()

    accu = test(model=ppnet, dataloader=test_loader,
                class_specific=class_specific, log_writer=log_writer, step=step)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=ACCURACY)

    if epoch >= push_start and epoch in push_epochs:
        step += 1
        write_mode(TrainMode.PUSH, log_writer, step)
        push.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True)
        accu = test(model=ppnet, dataloader=test_loader,
                    class_specific=class_specific, log_writer=log_writer, step=step)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=ACCURACY)

        if prototype_activation_function != 'linear':
            last_only(model=ppnet)
            for i in range(num_last_layer_iterations):
                step += 1
                write_mode(TrainMode.LAST_ONLY, log_writer, step)
                print('iteration: \t{0}'.format(i))
                _ = train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
                          class_specific=class_specific, coefs=coefs, log_writer=log_writer, step=step)
                accu = test(model=ppnet, dataloader=test_loader,
                            class_specific=class_specific, log_writer=log_writer, step=step)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=ACCURACY)

log_writer.close()
