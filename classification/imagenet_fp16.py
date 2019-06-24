# borrow from https://github.com/iVMCL/AOGNets

# From: https://github.com/NVIDIA/apex
### some tweaks
# USE pillow-simd to speed up pytorch image loader
# pip uninstall pillow
# conda uninstall --force jpeg libtiff -y
# conda install -c conda-forge libjpeg-turbo
# CC="cc -mavx2" pip install --no-cache-dir -U --force-reinstall --no-binary :all: --compile pillow-simd

# Install NCCL https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html

import argparse
import os
import shutil
import time
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import models.imagenet as customized_models
from flops_counter import get_model_complexity_info
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from utils import mkdir_p


# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# for servers to immediately record the logs
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()
    return new_print
print = flush_print(print)


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import math
import sys
import re

parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='aognet_s',
                    help='arch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=125, type=int,
                    metavar='N', help='mini-batch size per process (default: 125)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate.  \
                        Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. \
                             A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('--deterministic', action='store_true')

parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')

parser.add_argument('--save-dir', '--sd', type=str, default='/checkpoints/models')
parser.add_argument('--nesterov', type=str, default=None)
parser.add_argument('--remove-norm-weight-decay', type=bool, default=True)

#TODO: warmup_epochs, labelsmoothing_rate, mixup_rate, dataset, crop_size, crop_interpolation
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--labelsmoothing_rate', type=float, default=0.1)
parser.add_argument('--mixup_rate', default=0., type=float)
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--crop_size', default=224, type=int)
parser.add_argument('--crop_interpolation', default=2, type=int) 
#2: BILINEAR, 3: CUBIC

cudnn.benchmark = True

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

best_prec1 = 0
best_prec1_val = 0
prec5_val = 0
best_prec5_val = 0

args = parser.parse_args()

if args.local_rank == 0:
    print("PyTorch VERSION: {}".format(torch.__version__))  # PyTorch version
    print("CUDA VERSION: {}".format(torch.version.cuda))              # Corresponding CUDA version
    print("CUDNN VERSION: {}".format(torch.backends.cudnn.version()))  # Corresponding cuDNN version
    print("GPU TYPE: {}".format(torch.cuda.get_device_name(0)))   # GPU type

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.local_rank)
    torch.set_printoptions(precision=10)

def main():
    global best_prec1, args

    if args.local_rank == 0 and not os.path.isdir(args.save_dir):
        mkdir_p(args.save_dir)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 requires cudnn backend to be enabled."
    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # create model
    if args.pretrained:
        if args.local_rank == 0:
            print("=> using pre-trained model '{}'".format(args.arch))
        elif args.arch.startswith('resnet'):
            model = resnets.__dict__[args.arch](pretrained=True)
        elif args.arch.startswith('mobilenet'):
            model = mobilenets.__dict__[args.arch](pretrained=True)
        else:
            raise NotImplementedError("Unkown network arch.")
    else:
        if args.local_rank == 0:
            print("=> creating {}".format(args.arch))
        # update args

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif 'resnext' in args.arch:
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.local_rank == 0:
        if args.dataset.startswith('cifar'):
            H, W = 32, 32
        elif args.dataset.startswith('imagenet'):
            H, W = 224, 224
        else:
            raise NotImplementedError("Unknown dataset")
        flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
        print('=> FLOPs: {:.6f}G, Params: {:.6f}M'.format(flops/1e9, params/1e6))
        print('=> Params (double-check): %.6fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    if args.sync_bn:
        import apex
        if args.local_rank == 0:
            print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    model = model.cuda()
    if args.fp16:
        model = FP16Model(model)
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    if args.pretrained:
        model.load_state_dict(checkpoint['state_dict'])

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256 

    if args.remove_norm_weight_decay:
        if args.local_rank == 0:
            print("=> ! Weight decay NOT applied to FeatNorm parameters ")
        norm_params=set() #TODO: need to check this via experiments
        rest_params=set()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                for param in m.parameters(False):
                    norm_params.add(param)
            else:
                for param in m.parameters(False):
                    rest_params.add(param)

        optimizer = torch.optim.SGD([{'params': list(norm_params), 'weight_decay' : 0.0},
                            {'params': list(rest_params)}],
                           args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)
    else:
        if args.local_rank == 0:
            print("=> ! Weight decay applied to FeatNorm parameters ")
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)

    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.static_loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale)

    # define loss function (criterion) and optimizer
    criterion_train = nn.CrossEntropyLoss().cuda() if args.labelsmoothing_rate == 0.0 \
                        else LabelSmoothing(args.labelsmoothing_rate).cuda()
    criterion_val   = nn.CrossEntropyLoss().cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                if args.local_rank == 0:
                    print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.local_rank == 0:
                    print("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
            else:
                if args.local_rank == 0:
                    print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Data loading code
    if args.dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
            ])
        train_dataset = datasets.CIFAR10('./datasets', train=True, download=False, transform=train_transform)
        val_dataset = datasets.CIFAR10('./datasets', train=False, download=False)
    elif args.dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
            ])
        train_dataset = datasets.CIFAR100('./datasets', train=True, download=False, transform=train_transform)
        val_dataset = datasets.CIFAR100('./datasets', train=False, download=False)
    elif args.dataset == "imagenet":
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'valf')

        crop_size = args.crop_size # 224
        val_size = args.crop_size + 32 # 256

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(crop_size, interpolation=args.crop_interpolation),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(), Too slow
                # normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(val_size, interpolation=args.crop_interpolation),
                transforms.CenterCrop(crop_size),
            ]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=fast_collate)

    if args.evaluate:
        validate(val_loader, model, criterion_val)
        return

    scheduler = CosineAnnealingLR(optimizer.optimizer if args.fp16 else optimizer,
        args.epochs, len(train_loader), eta_min=0., warmup=args.warmup_epochs) 

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion_train, optimizer, epoch, scheduler, args.warmup_epochs,
                args.mixup_rate, args.labelsmoothing_rate) 
        #TODO: warmup_epochs, labelsmoothing_rate, mixup_rate, args.dataset, args.cropsize, args.crop_interpolation
        if args.prof:
            break
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion_val)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_dir)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        if args.dataset == 'cifar10':
            self.mean = torch.tensor([0.49139968 * 255, 0.48215827 * 255, 0.44653124 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.24703233 * 255, 0.24348505 * 255, 0.26158768 * 255]).cuda().view(1,3,1,1)
        elif args.dataset == 'cifar100':
            self.mean = torch.tensor([0.5071 * 255, 0.4867 * 255, 0.4408 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.2675 * 255, 0.2565 * 255, 0.2761 * 255]).cuda().view(1,3,1,1)
        elif args.dataset == 'imagenet':
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        else:
            raise NotImplementedError
        if args.fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            if args.fp16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

# from NVIDIA DL Examples
def prefetched_loader(loader):
    if args.dataset == 'cifar10':
        self.mean = torch.tensor([0.49139968 * 255, 0.48215827 * 255, 0.44653124 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.24703233 * 255, 0.24348505 * 255, 0.26158768 * 255]).cuda().view(1,3,1,1)
    elif args.dataset == 'cifar100':
        self.mean = torch.tensor([0.5071 * 255, 0.4867 * 255, 0.4408 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.2675 * 255, 0.2565 * 255, 0.2761 * 255]).cuda().view(1,3,1,1)
    elif args.dataset == 'imagenet':
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
    else:
        raise NotImplementedError

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def train(train_loader, model, criterion, optimizer, epoch, scheduler, warmup_epoch=0,
            mixup_rate=0.0, labelsmoothing_rate=0.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = -1
    beta_distribution = torch.distributions.beta.Beta(mixup_rate, mixup_rate)
    while input is not None:
        i += 1

        lr = scheduler.update(epoch, i)

        if args.prof:
            if i > 10:
                break
        # measure data loading time
        data_time.update(time.time() - end)

        # Mixup input
        if mixup_rate > 0.0:
            lambda_ = beta_distribution.sample([]).item()
            index = torch.randperm(input.size(0)).cuda()
            input = lambda_ * input + (1 - lambda_) * input[index, :]

        # compute output
        if args.prof: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof: torch.cuda.nvtx.range_pop()

        # Mixup loss
        if mixup_rate > 0.0:
            # Mixup loss
            loss = (lambda_ * criterion(output, target)
                    + (1 - lambda_) * criterion(output, target[index]))

            # Mixup target
            if labelsmoothing_rate > 0.0:
                N = output.size(0)
                C = output.size(1)
                off_prob = labelsmoothing_rate / C
                target_1 = torch.full(size=(N, C), fill_value=off_prob ).cuda()
                target_2 = torch.full(size=(N, C), fill_value=off_prob ).cuda()
                target_1.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1.0-labelsmoothing_rate+off_prob)
                target_2.scatter_(dim=1, index=torch.unsqueeze(target[index], dim=1), value=1.0-labelsmoothing_rate+off_prob)
                target = lambda_ * target_1 + (1 - lambda_) * target_2
            else:
                target = lambda_ * target + (1 - lambda_) * target[index]
        else:
            loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof: torch.cuda.nvtx.range_push("backward")
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()
        if args.prof: torch.cuda.nvtx.range_pop()

        # debug
        # if args.local_rank == 0:
        #     for name_, param in model.named_parameters():
        #         print(name_, param.data.double().sum().item(), param.grad.data.double().sum().item())

        if args.prof: torch.cuda.nvtx.range_push("step")
        optimizer.step()
        if args.prof: torch.cuda.nvtx.range_pop()

        # Measure accuracy
        if mixup_rate > 0.0:
            prec1 = rmse(output.data, target)
            prec5 = prec1
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # Average loss and accuracy across processes for logging
        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        # to_python_float incurs a host<->device sync
        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # torch.cuda.synchronize() # no this in torchvision ex. and cause nan loss problems in deep models with fp16

        batch_time.update(time.time() - end)
        end = time.time()
        input, target = prefetcher.next()

        if i%args.print_freq == 0 and args.local_rank == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'lr {lr:.6f}\t'.format(
                       epoch, i, len(train_loader),
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr[0]))

def validate(val_loader, model, criterion):
    global best_prec1_val, prec5_val, best_prec5_val
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        print_str = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        input, target = prefetcher.next()

    if args.local_rank == 0:
        print(print_str)
        if top1.avg >= best_prec1_val:
            best_prec1_val = top1.avg
            prec5_val = top5.avg
        best_prec5_val = max(best_prec5_val, top5.avg)
        print('Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t Best_Prec@1 {best:.3f}\t Prec@5 {prec5_val:.3f}\t Best_Prec@5 {bestprec5_val:.3f}'
              .format(top1=top1, top5=top5, best=best_prec1_val, prec5_val=prec5_val, bestprec5_val=best_prec5_val))

    return top1.avg


def save_checkpoint(state, is_best, save_dir='./'):
    filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    best_file = os.path.join(save_dir, 'model_best.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class CosineAnnealingLR(object):
    def __init__(self, optimizer, T_max, N_batch, eta_min=0, last_epoch=-1, warmup=0):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.T_max = T_max
        self.N_batch = N_batch
        self.eta_min = eta_min
        self.warmup = warmup

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.update(last_epoch+1)
        self.last_epoch = last_epoch
        self.iter = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            lrs = [base_lr * (self.last_epoch + self.iter / self.N_batch) / self.warmup for base_lr in self.base_lrs]
        else:
            lrs = [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup + self.iter / self.N_batch) / (self.T_max - self.warmup))) / 2
                    for base_lr in self.base_lrs]
        return lrs

    def update(self, epoch, batch=0):
        self.last_epoch = epoch
        self.iter = batch + 1
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

        return lrs


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rmse(yhat,y):
    if args.fp16:
        res = torch.sqrt(torch.mean((yhat.float()-y.float())**2))
    else:
        res = torch.sqrt(torch.mean((yhat-y)**2))
    return res

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


if __name__ == '__main__':
    # to suppress annoying warnings
    import warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    main()
