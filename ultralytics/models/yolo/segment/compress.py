# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys, os, torch, math, time, warnings
import torch_pruning as tp
import matplotlib
matplotlib.use('AGG')
import matplotlib.pylab as plt
import torch.nn as nn
from torch import optim
from thop import clever_format
from functools import partial
from torch import distributed as dist
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from ..detect.c2f_transfer import replace_c2f_with_c2f_v2
from datetime import datetime

from copy import copy, deepcopy
from pathlib import Path

import numpy as np

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, clean_url, colorstr, emojis, yaml_save, callbacks, __version__
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.checks import check_imgsz, print_args, check_amp
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.torch_utils import ModelEMA, EarlyStopping, one_cycle, init_seeds, select_device
from torch.utils.tensorboard import SummaryWriter
# from ultralytics.nn.extra_modules.kernel_warehouse import get_temperature

from ultralytics.nn.modules import Segment, RepConv
from ultralytics.nn.extra_modules.MyPruner import DiverseBranchBlockPruner, LayerNormPruner, RepConvPruner, DyHeadBlockPruner, RepConvNPruner
from timm.models.layers import SqueezeExcite


from ultralytics.nn.extra_modules import Detect_Efficient, Detect_DyHead_Prune
from ultralytics.nn.extra_modules.block import Faster_Block, Fusion, IFM, InjectionMultiSum_Auto_pool, TopBasicLayer, SimFusion_3in, SimFusion_4in, AdvPoolFusion, PyramidPoolAgg, RepVGGBlock, RepConvN
from ultralytics.nn.extra_modules.rep_block import DiverseBranchBlock
from ultralytics.nn.extra_modules.dyhead_prune import DyHeadBlock_Prune
from ultralytics.nn.backbone.convnextv2 import LayerNorm

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_pruner(opt, model, example_inputs):
    sparsity_learning = False
    if opt.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "l1":
        # https://arxiv.org/abs/1608.08710
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "lamp":
        # https://arxiv.org/abs/2010.07611
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "slim":
        # https://arxiv.org/abs/1708.06519
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_slim":
        # https://tibshirani.su.domains/ftp/sparse-grlasso.pdf
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=opt.reg, global_pruning=opt.global_pruning, group_lasso=True)
    elif opt.prune_method == "group_norm":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_sl":
        # https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=opt.reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "growing_reg":
        # https://arxiv.org/abs/2012.09243
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=opt.reg, delta_reg=opt.delta_reg, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_hessian":
        # https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
        imp = tp.importance.HessianImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    elif opt.prune_method == "group_taylor":
        # https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf
        imp = tp.importance.TaylorImportance(group_reduction='mean')
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=opt.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    customized_pruners = {}
    round_to = None
    
    # ignore output layers
    # for yolov8.yaml
    for k, m in model.named_modules():
        if isinstance(m, Segment):
            ignored_layers.append(m.cv2[0][2])
            ignored_layers.append(m.cv2[1][2])
            ignored_layers.append(m.cv2[2][2])
            ignored_layers.append(m.cv3[0][2])
            ignored_layers.append(m.cv3[1][2])
            ignored_layers.append(m.cv3[2][2])
            ignored_layers.append(m.cv4[0][2])
            ignored_layers.append(m.cv4[1][2])
            ignored_layers.append(m.cv4[2][2])
            ignored_layers.append(m.dfl)
            ignored_layers.append(m.proto)
    
    print(ignored_layers)
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=opt.iterative_steps,
        pruning_ratio=1.0,
        pruning_ratio_dict=pruning_ratio_dict,
        max_pruning_ratio=opt.max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        customized_pruners=customized_pruners,
        round_to=round_to,
        root_module_types=[nn.Conv2d, nn.Linear]
    )
    return sparsity_learning, imp, pruner

linear_trans = lambda epoch, epochs, reg, reg_ratio: (1 - epoch / (epochs - 1)) * (reg - reg_ratio) + reg_ratio

class SegmentationCompressor(BaseTrainer):
    """
    A class extending the DetectionTrainer class for training based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml', epochs=3)
        trainer = PoseTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        if self.args.sl_model:
            self.save_dir = Path(self.args.project) / self.args.name
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            (self.save_dir / 'visual').mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.sl_epochs
        self.start_epoch = 0
        self.args.task = 'segment'
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in ('cpu', 'mps'):
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = self.args.model
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.split('.')[-1] in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)
    
    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import pandas as pd  # scope for faster startup
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last and best
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if self.best_sl[f'{self.sparsity_ratio:.3f}'] == self.fitness:
            torch.save({'model': deepcopy(de_parallel(self.model)).half(),
                        'ema': deepcopy(self.ema.ema).half(),}, self.wdir / 'best_sl_{:.3f}.pt'.format(self.sparsity_ratio))
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
      
    def validate_prune(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        self.ema.ema = deepcopy(self.model)
        metrice = self.validator(self)
        self.ema.ema = None
        torch.cuda.empty_cache()
        return metrice
    
    def model_prune(self, imp, prune, example_inputs):
        N_batchs = 10

        base_model = deepcopy(self.model)
        with HiddenPrints():
            ori_flops, ori_params = tp.utils.count_ops_and_params(base_model, example_inputs)
        ori_flops = ori_flops * 2.0
        ori_flops_f, ori_params_f = clever_format([ori_flops, ori_params], "%.3f")
        ori_result = self.validate_prune()
        ori_map50, ori_map = ori_result['metrics/mAP50(B)'], ori_result['metrics/mAP50-95(B)']
        iter_idx, prune_flops = 0, ori_flops
        speed_up = 1.0
        LOGGER.info('begin pruning...')
        while speed_up < self.args.speed_up:
            self.model.train()
            if isinstance(imp, tp.importance.HessianImportance):
                for k, batch in enumerate(self.train_loader):
                    if k >= N_batchs: break
                    batch = self.preprocess_batch(batch)
                    # compute loss for each sample
                    loss = self.model(batch)[0]
                    imp.zero_grad() # clear accumulated gradients
                    self.model.zero_grad() # clear gradients
                    loss.backward(retain_graph=True) # simgle-sample gradient
                    imp.accumulate_grad(self.model) # accumulate g^2
            elif isinstance(imp, tp.importance.TaylorImportance):
                for k, batch in enumerate(self.train_loader):
                    if k >= N_batchs: break
                    batch = self.preprocess_batch(batch)
                    loss = self.model(batch)[0]
                    loss.backward()
            
            iter_idx += 1
            prune.step(interactive=False)
            prune_result = self.validate_prune()
            prune_map50, prune_map = prune_result['metrics/mAP50(B)'], prune_result['metrics/mAP50-95(B)']
            with HiddenPrints():
                prune_flops, prune_params = tp.utils.count_ops_and_params(self.model, example_inputs)
            prune_flops = prune_flops * 2.0
            prune_flops_f, prune_params_f = clever_format([prune_flops, prune_params], "%.3f")
            speed_up = ori_flops / prune_flops # ori_model_GFLOPs / prune_model_GFLOPs
            LOGGER.info(f'pruning... iter:{iter_idx} ori model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%) map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f}) map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f}) Speed Up:{ori_flops / prune_flops:.2f}')
            
            if prune.current_step == prune.iterative_steps:
                break
        
        if isinstance(imp, tp.importance.HessianImportance):
            imp.zero_grad()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        
        LOGGER.info('pruning done...')
        LOGGER.info(f'model flops:{ori_flops_f} => {prune_flops_f}({prune_flops / ori_flops * 100:.2f}%) Speed Up:{ori_flops / prune_flops:.2f}')
        LOGGER.info(f'model params:{ori_params_f} => {prune_params_f}({prune_params / ori_params * 100:.2f}%)')
        LOGGER.info(f'model map@50:{ori_map50:.3f} => {prune_map50:.3f}({prune_map50 - ori_map50:.3f})')
        LOGGER.info(f'model map@50:95:{ori_map:.3f} => {prune_map:.3f}({prune_map - ori_map:.3f})')
    
    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        # self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        # self.scaler.step(self.optimizer)
        self.optimizer.step()
        # self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
    
    def sparsity_learning(self, ckpt, world_size, prune):
        # Optimizer
        
        self.batch_size = self.batch_size // 2
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                            name=self.args.optimizer,
                                            lr=self.args.lr0,
                                            momentum=self.args.momentum,
                                            decay=weight_decay,
                                            iterations=iterations)
        
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.best_sl = {}
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')
        
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting Sparsity training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # if hasattr(self.model, 'net_update_temperature'):
                #     temp = get_temperature(i + 1, epoch, len(self.train_loader), temp_epoch=20, temp_init_value=1.0)
                #     self.model.net_update_temperature(temp)
                
                # Forward
                # with torch.cuda.amp.autocast(self.amp):
                batch = self.preprocess_batch(batch)
                self.loss, self.loss_items = self.model(batch)
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                    else self.loss_items

                # Backward
                # self.scaler.scale(self.loss).backward()
                self.loss.backward()
                
                if isinstance(prune, (tp.pruner.BNScalePruner,)):
                    if self.args.reg_decay_type == 'linear':
                        reg = linear_trans(epoch, self.epochs, self.args.reg, self.args.reg * self.args.reg_decay)
                    elif self.args.reg_decay_type == 'step':
                        reg = self.args.reg * (self.args.reg_decay ** (epoch // self.args.reg_decay_step))
                    elif self.args.opt.reg_decay_type == 'constant':
                        reg = self.args.reg
                    prune.regularize(self.model, reg=reg)
                elif isinstance(prune, (tp.pruner.GroupNormPruner, tp.pruner.GrowingRegPruner)):
                    reg = self.args.reg
                    prune.regularize(self.model)

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            if isinstance(prune, (tp.pruner.GrowingRegPruner,)):
                prune.update_reg()
            
            if self.ema:
                model_sl = self.ema.ema.state_dict()
            else:
                model_sl = self.model.state_dict()
            
            tb = SummaryWriter(self.save_dir)
            
            bn_weight = []
            for name in model_sl:
                if 'weight' in name and len(model_sl[name].size()) == 1:
                    weight = model_sl[name].data.cpu().abs().clone().numpy().reshape((-1))
                    bn_weight.append(weight)
            bn_weight = np.concatenate(bn_weight)
            bn_weight = np.sort(bn_weight)
            bn_weight_percent = np.percentile(bn_weight, [1, 5, 10, 25, 50, 75])
            self.sparsity_ratio = np.sum(bn_weight < 1e-6) / bn_weight.shape[0]
            if f'{self.sparsity_ratio:.3f}' not in self.best_sl:
                self.best_sl[f'{self.sparsity_ratio:.3f}'] = 0.0
            if tb:
                tb.add_histogram('hist', bn_weight, epoch, bins='doane')
            
            del model_sl
            
            plt.figure(figsize=(15, 5), clear=True)
            plt.plot(bn_weight)
            plt.title(f'sparsity_ratio:{self.sparsity_ratio:.3f}\n')
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/visual/{epoch}_sl_{self.sparsity_ratio:.3f}.png')
            plt.close('all')
            LOGGER.info(f'epoch:{epoch} reg:{reg:.5f} sparsity_ratio:{self.sparsity_ratio:.5f} bn_weight_1:{bn_weight_percent[0]:.10f} bn_weight_5:{bn_weight_percent[1]:.8f} bn_weight_10:{bn_weight_percent[2]:.8f}\nbn_weight_25:{bn_weight_percent[3]:.5f} bn_weight_50:{bn_weight_percent[4]:.5f} bn_weight_75:{bn_weight_percent[5]:.5f}')
            
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                if self.fitness > self.best_sl[f'{self.sparsity_ratio:.3f}']:
                    self.best_sl[f'{self.sparsity_ratio:.3f}'] = self.fitness
                
                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
        
        sl = sorted(self.best_sl.keys(), key=lambda x:float(x))[-1]
        best_sl_model = self.wdir / 'best_sl_{}.pt'.format(sl)
        
        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
        return best_sl_model
    
    def compress(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(','))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0
        
        # init model
        ckpt = self.setup_model()
        replace_c2f_with_c2f_v2(self.model)
        # print(self.model)
        self.model = self.model.to(self.device)
        torch.save({'model':deepcopy(self.model).half(), 'ema':None}, self.wdir / 'model_c2f_v2.pt')
        self.set_model_attributes()
        
        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True
        
        # for sparsity learning
        self.amp = False
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])
        
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()
        
        # get prune
        if type(self.args.imgsz) is int:
            example_inputs = torch.randn((1, 3, self.args.imgsz, self.args.imgsz)).to(self.device)
        elif type(self.args.imgsz) is list:
            example_inputs = torch.randn((1, 3, self.args.imgsz[0], self.args.imgsz[1])).to(self.device)
        else:
            assert f'self.args.imgsz type error! {self.args.imgsz}'
            
        # pre initt 
        batch_data = self.preprocess_batch(next(iter(self.train_loader), None))
        _, self.loss_items = self.model(batch_data)
        self.stopper = EarlyStopping(0)
        
        sparsity_learning, imp, prune = get_pruner(self.args, self.model, example_inputs)
        
        if sparsity_learning and not self.args.sl_model:
            self.args.sl_model = self.sparsity_learning(ckpt, world_size, prune)
        
        if sparsity_learning:
            model = torch.load(self.args.sl_model, map_location=self.device)
            model = model['ema' if model.get('ema') else 'model'].float()
            for p in model.parameters():
                p.requires_grad_(True)
            self.model = model
            self.model = self.model.to(self.device)
            self.set_model_attributes()
            sparsity_learning, imp, prune = get_pruner(self.args, self.model, example_inputs)
        
        # using for val
        if not hasattr(self, 'epoch'):
            self.epoch, self.epochs = 0, 1
        
        self.ema.ema = None
        self.model_prune(imp, prune, example_inputs)
        
        # test fuse
        fuse_model = deepcopy(self.model)
        for p in fuse_model.parameters():
            p.requires_grad_(False)
        fuse_model.fuse()
        del fuse_model
        
        prune_path = self.wdir / 'prune.pt'
        ckpt = {'model': deepcopy(de_parallel(self.model)).half(), 'ema': None}
        torch.save(ckpt, prune_path)
        LOGGER.info(colorstr(f'Pruning after Finetune before the model is saved in:{prune_path}'))
        return str(prune_path)
    
    def __del__(self):
        self.train_loader = None
        self.test_loader = None
        self.model = None

class SegmentationFinetune(yolo.segment.SegmentationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = torch.load(self.args.model, map_location=self.device)
        model = model['ema' if model.get('ema') else 'model'].float()
        for p in model.parameters():
            p.requires_grad_(True)
        LOGGER.info(colorstr("prune_model info:"))
        model.info()
        return model
    
    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        # if str(model).endswith('.pt'):
        #     weights, ckpt = attempt_load_one_weight(model)
        #     cfg = ckpt['model'].yaml
        # else:
        #     cfg = model
        self.model = self.get_model(cfg=model, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return None
    
    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')