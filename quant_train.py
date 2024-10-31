import argparse
import os
import time
import math
import logging
import numpy as np

import torch
import torch.nn as nn
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy

from qmodels import *
from utils import *

import wandb

#왜안될까요
# os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
# # 피어 투 피어(P2P) 및 InfiniBand 비활성화
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"

# # 타임아웃 시간을 증가시켜 통신 지연 문제 완화 (초 단위)
# os.environ["NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
# os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker0"
# os.environ["NCCL_TIMEOUT"] = "150"

# # GPU에서 직접 메모리 접근을 비활성화해 오류를 방지
# os.environ["NCCL_SHM_DISABLE"] = "1"
# os.environ["NCCL_NET_GDR_LEVEL"] = "PHB"



DIST_PORT=6006
WANDB_LOG = False
WANDB_PROJ_NAME = 'forward&backward training'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['RANK'] = '4'
os.environ['WORLD_SIZE'] = '4'
# os.environ["MASTER_PORT"] = '6006'


def get_args_parser():
    parser = argparse.ArgumentParser(description="I-ViT")

    #custom
    parser.add_argument('--abits', default=4, type=int)
    parser.add_argument('--wbits', default=4, type=int)
    parser.add_argument('--gbits', default=None, type=int)
    parser.add_argument('--qdtype', default='int8')

    parser.add_argument("--model", default='deit_tiny',
                        choices=['deit_tiny', 'deit_small', 'deit_base', 
                                'swin_tiny', 'swin_small', 'swin_base'],
                        help="model")
    parser.add_argument('--data', metavar='DIR', default='/dataset/imagenet/',
                        help='path to dataset')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument("--nb-classes", default=1000, type=int, help="number of classes")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=1000,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument('--output-dir', type=str, default='results/',
                        help='path to save log and quantized model')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--best-acc1', type=float, default=0, help='best_acc1')

    # distributed training parameters
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def str2model(name):
    d = {'deit_tiny': deit_tiny_patch16_224,
         'deit_small': deit_small_patch16_224,
         'deit_base': deit_base_patch16_224,
        #  'swin_tiny': swin_tiny_patch4_window7_224,
        #  'swin_small': swin_small_patch4_window7_224,
        #  'swin_base': swin_base_patch4_window7_224,
         }
    print('Model: %s' % d[name].__name__)
    return d[name]


def main(args):
    if args.qdtype == 'int8':
        args.qdtype = torch.int8
    elif args.qdtype == 'int32':
        args.qdtype = torch.int32
    elif args.qdtype == 'int64':
        args.qdtype = torch.int64
    elif args.qdtype == 'float32':
        args.qdtype = torch.float32
    else:
        raise ValueError("지원되지 않는 자료형입니다.")

    node_rank = getattr(args, "ddp_rank", 0)
    if node_rank==0:
        print(args)
    
    

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    import warnings
    warnings.filterwarnings('ignore')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S', filename=args.output_dir + 'log.log')
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    device = torch.device(args.device)
    device_id = getattr(args, "dev_device_id", torch.device("cpu"))

    #Set Wandb
    wandb_log = WANDB_LOG 
    wandb_project = WANDB_PROJ_NAME
    wandb_run_name = args.model 
    if wandb_log:
        wandb.init(project=wandb_project, name=wandb_run_name, config=args)

    # Dataset
    train_loader, val_loader = dataloader(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Model
    model = str2model(args.model)(
                                  abits=args.abits,
                                  wbits=args.wbits, 
                                  gbits=args.gbits, 
                                  qdtype=args.qdtype,
                                  pretrained=False,
                                  num_classes=args.nb_classes,
                                  drop_rate=args.drop,
                                  drop_path_rate=args.drop_path)
    model.to(device)



    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if node_rank==0:
        print('number of params:', n_parameters)

    args.min_lr = args.lr / 15
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if wandb_log:
        wandb.watch(model)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion_v = nn.CrossEntropyLoss()

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if node_rank==0:
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(args,  wandb_log, node_rank, train_loader, model, criterion, optimizer, epoch,
              loss_scaler, args.clip_grad, model_ema, mixup_fn, device)
        lr_scheduler.step(epoch)


        # if args.output_dir:  # this is for resume training
        #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth.tar')
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'model_ema': get_state_dict(model_ema),
        #         'scaler': loss_scaler.state_dict(),
        #         'args': args,
        #     }, checkpoint_path)

        acc1 = validate(args, wandb_log, node_rank, val_loader, model, criterion_v, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > args.best_acc1
        args.best_acc1 = max(acc1, args.best_acc1)
        if is_best:
            # record the best epoch
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint.pth.tar'))
        logging.info(f'Acc at epoch {epoch}: {acc1}')
        logging.info(f'Best acc at epoch {best_epoch}: {args.best_acc1}')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if node_rank == 0: 
        print('Training time {}'.format(total_time_str))
        if wandb_log:
            wandb.finish()


def train(args, wandb_log, node_rank, train_loader, model, criterion, optimizer, epoch, loss_scaler, max_norm, model_ema, mixup_fn, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    # unfreeze_model(model)

    end = time.time()
    with torch.autograd.detect_anomaly():
        for i, (data, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if mixup_fn is not None:
                data, target = mixup_fn(data, target)

            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), data.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                if node_rank == 0:
                    progress.display(i)
                    if wandb_log:
                        wandb.log({
                                "train/loss": loss.item,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "iteration": i + epoch * len(train_loader)
                                })
    if wandb_log: 
        wandb.log({
                    "epoch": epoch,
                    "train/loss": losses.avg
        })


def validate(args, wandb_log, node_rank, val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    freeze_model(model)

    end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            if node_rank == 0:
                progress.display(i)
                if wandb_log:
                    wandb.log({
                            "val/loss": loss.item,
                            "iteration": i 
                            })

    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def distributed_init(args) -> int:
    ddp_url = getattr(args, "ddp_dist_url", None)

    node_rank = getattr(args, "ddp_rank", 0)
    is_master_node = (node_rank == 0)

    if ddp_url is None:
        ddp_port = DIST_PORT
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(args, "ddp_dist_url", ddp_url)

    node_rank = getattr(args, "ddp_rank", 0)
    world_size = getattr(args, "ddp_world_size", 0)

    # 하나의 포트를 여러개의 DDP 프로세스 그룹이 같이 쓸 수 없음. 무조건 하나의 포트, 하나의 DDP 프로세스 그룹임.
    # 이거 몰라서 에러 터트린적 많으니 실험 여러개 돌릴 때 필히 포트 번호를 신경쓸 것
    if torch.distributed.is_initialized():
        print("DDP is already initialized and cannot be initialize twice!")
    else:
        print("distributed init (rank {}): {}".format(node_rank, ddp_url))

        dist_backend = getattr(args, "ddp_backend", "nccl")  # "gloo"

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                print(
                    "Using NCCL as distributed backend with version={}".format(
                        torch.cuda.nccl.version()
                    )
                )
        elif dist_backend is None:
            dist_backend = "gloo"
        if not torch.distributed.is_initialized():
            dist.init_process_group(
                backend=dist_backend,
                init_method=ddp_url,
                world_size=world_size,
                rank=node_rank,
            )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(args, "ddp_rank", node_rank)
    return node_rank

def distributed_worker(i, main, args):
    setattr(args, "dev_device_id", i)
    torch.cuda.set_device(i)
    setattr(args, "dev_device", torch.device(f"cuda:{i}"))

    # local rank, started from 0 
    ddp_rank =  i
    setattr(args, "ddp_rank", ddp_rank)
    
    node_rank = distributed_init(args)
    setattr(args, "ddp_rank", node_rank)
    #cProfile.runctx('main(logger, args)', globals(), locals(), sort='time')

    main(args)

def ddp_or_single_process(args):
    setattr(args, "dev_num_gpus", torch.cuda.device_count()) 
    setattr(args, "ddp_world_size", torch.cuda.device_count())
    torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, args),
            nprocs=int(os.environ['WORLD_SIZE']),
        )

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ddp_or_single_process(args)