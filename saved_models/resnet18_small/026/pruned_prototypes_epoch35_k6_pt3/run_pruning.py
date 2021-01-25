import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse

from colon_dataset import ColonCancerBagsCross
from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]


optimize_last_layer = True

# pruning parameters
k = 6
prune_threshold = 3

original_model_dir = args.modeldir[0] #'./saved_models/densenet161/003/'
original_model_name = args.model[0] #'10_16push0.8007.pth'

need_push = ('nopush' in original_model_name)
if need_push:
    assert(False) # pruning must happen after push
else:
    epoch = original_model_name.split('push')[0]

if '_' in epoch:
    epoch = int(epoch.split('_')[0])
else:
    epoch = int(epoch)

model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch,
                                          k,
                                          prune_threshold))
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

ppnet = torch.load(original_model_dir + original_model_name)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# load the data
from settings import train_dir, test_dir, train_push_dir

train_batch_size = 1
test_batch_size = 1
img_size = 27
train_push_batch_size = 1

normalize = transforms.Normalize(mean=mean,
                                 std=std)

split_val = 70
train_range, test_range = range(split_val), range(split_val, 100)

ds = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range, test_idxs=test_range,
                          shuffle_bag=True)
ds_push = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range, test_idxs=test_range,
                               push=True, shuffle_bag=True)
ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range, test_idxs=test_range)

# train set
# train_dataset = datasets.ImageFolder(
#     train_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#         normalize,
#     ]))
train_loader = torch.utils.data.DataLoader(
    ds, batch_size=train_batch_size, shuffle=True,
    num_workers=0, pin_memory=False)



# test set
# test_dataset = datasets.ImageFolder(
#     test_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#         normalize,
#     ]))
test_loader = torch.utils.data.DataLoader(
    ds_test, batch_size=test_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# push set: needed for pruning because it is unnormalized
# train_push_dataset = datasets.ImageFolder(
#     train_push_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#     ]))
train_push_loader = torch.utils.data.DataLoader(
    ds_push, batch_size=train_push_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)
    
log('push set size: {0}'.format(len(train_push_loader.dataset)))

tnt.test(model=ppnet_multi, dataloader=test_loader,
         class_specific=class_specific, log=log)

# prune prototypes
log('prune')
prune.prune_prototypes(dataloader=train_push_loader,
                       prototype_network_parallel=ppnet_multi,
                       k=k,
                       prune_threshold=prune_threshold,
                       preprocess_input_function=None, # normalize
                       original_model_dir=original_model_dir,
                       epoch_number=epoch,
                       #model_name=None,
                       log=log,
                       copy_prototype_imgs=True)
accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                class_specific=class_specific, log=log)
save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                            model_name=original_model_name.split('push')[0] + 'prune',
                            accu=accu,
                            target_accu=0.70, log=log)

# last layer optimization
if optimize_last_layer:
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
    }

    log('optimize last layer')
    tnt.last_only(model=ppnet_multi, log=log)
    for i in range(100):
        log('iteration: \t{0}'.format(i))
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                    model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                    accu=accu,
                                    target_accu=0.70, log=log)

logclose()
