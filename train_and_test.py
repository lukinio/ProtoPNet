import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter

from focalloss import FocalLoss
from helpers import list_of_distances


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log_writer: SummaryWriter = None, step: int = 0):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_loss = 0
    conf_matrix = np.zeros((2, 2), dtype='int32')

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)

            cross_entropy = FocalLoss(alpha=0.5, gamma=2)(output, target)

            if class_specific:
                max_dist = (model.prototype_shape[1]
                            * model.prototype_shape[2]
                            * model.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                            dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            conf_matrix += confusion_matrix(target.cpu().numpy(), predicted.cpu().numpy(), labels=[0, 1])

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if class_specific:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['sep'] * separation_cost
                        + coefs['l1'] * l1)
            else:
                loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
        else:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['l1'] * l1)
            else:
                loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
        total_loss += loss.item()
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # gradient clipping
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    total_cross_entropy /= n_batches
    total_cluster_cost /= n_batches
    total_separation_cost /= n_batches
    total_loss /= n_batches

    print('accuracy:', n_correct / n_examples)
    print('total_loss:', total_loss)

    suffix = '/train' if is_train else '/test'
    if log_writer:

        log_writer.add_scalar('total_loss' + suffix, total_loss, global_step=step)
        log_writer.add_scalar('cross_entropy' + suffix, total_cross_entropy, global_step=step)
        log_writer.add_scalar('cluster_cost' + suffix, total_cluster_cost, global_step=step)

        if class_specific:
            log_writer.add_scalar('separation_cost' + suffix, total_separation_cost, global_step=step)
            log_writer.add_scalar('avg_separation_cost' + suffix, total_avg_separation_cost / n_batches,
                                  global_step=step)

        log_writer.add_scalar('accuracy' + suffix, n_correct / n_examples, global_step=step)
        log_writer.add_scalar('l1' + suffix, model.last_layer.weight.norm(p=1).item(), global_step=step)
        conf_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(cmap='Blues', values_format='d')
        log_writer.add_figure('confusion_matrix' + suffix, conf_plot.figure_, global_step=step, close=True)

    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))

    if log_writer:
        log_writer.add_scalar('p_avg_pair_dist' + suffix, p_avg_pair_dist, global_step=step)

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log_writer: SummaryWriter = None,
          step: int = 0):
    assert (optimizer is not None)

    print('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log_writer=log_writer, step=step)


def test(model, dataloader, class_specific=False, log_writer: SummaryWriter = None, step: int = 0):
    print('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log_writer=log_writer, step=step)


def _freeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = False


def _unfreeze_layer(layer):
    for p in layer.parameters():
        p.requires_grad = True


def last_only(model):
    _freeze_layer(model.features)
    _freeze_layer(model.add_on_layers)

    model.prototype_vectors.requires_grad = False

    _unfreeze_layer(model.attention_V)
    _unfreeze_layer(model.attention_U)
    _unfreeze_layer(model.attention_weights)
    _unfreeze_layer(model.last_layer)

    print('\tlast layer')


def warm_only(model):
    _freeze_layer(model.features)
    _unfreeze_layer(model.add_on_layers)

    _freeze_layer(model.attention_V)
    _freeze_layer(model.attention_U)
    _freeze_layer(model.attention_weights)

    model.prototype_vectors.requires_grad = True

    _unfreeze_layer(model.last_layer)

    print('\twarm')


def joint(model):
    _unfreeze_layer(model.features)
    _unfreeze_layer(model.add_on_layers)

    model.prototype_vectors.requires_grad = True

    _freeze_layer(model.attention_V)
    _freeze_layer(model.attention_U)
    _freeze_layer(model.attention_weights)
    _freeze_layer(model.last_layer)

    print('\tjoint')
