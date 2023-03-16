import argparse
import torch
import numpy as np
import sys
from datetime import datetime
from exp import getExp
from dataloaders import make_data_loader
import importlib
from utils.lr_scheduler import LR_Scheduler
from exp_config import cfg
from tqdm import tqdm
import os
from utils.summaries import TensorboardSummary
from utils.saver import Saver   
from utils.logger import Logger


class Trainer(object):
    def __init__(self, cfg):
        # Define Saver
        self.saver = Saver(cfg)
        self.saver.save_experiment_config(cfg)
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(cfg, self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        #self.device = torch.device("cuda:{:d}".format(int(cfg.SYSTEM.GPU_IDS[0])) if cfg.SYSTEM.USE_GPU else "cpu")
        self.device = torch.device("cuda:0" if cfg.SYSTEM.USE_GPU else "cpu")

        # Define Dataloader
        kwargs = {'num_workers': cfg.SYSTEM.NUM_CPU, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(cfg, **kwargs)

        # Define network
        self.model = getExp(cfg).to(self.device)

        # Define Optimizer
        train_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(train_params,lr=cfg.OPTIMIZER.LR, momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, 
                                    nesterov=cfg.OPTIMIZER.NESTEROV)
        # Define Criterion + whether to use class balanced weights

        weight = None
        kwargs = {'cfg': cfg, 'loss_cfg': cfg.LOSS, 'weight': weight, 'use_cuda': cfg.SYSTEM.USE_GPU}
        criterion_module = importlib.import_module("net.loss")
        self.criterion = getattr(criterion_module, cfg.LOSS.TYPE)(**kwargs)
       
        # Define Evaluator
        evaluation_module = importlib.import_module("utils.metrics")
        kwargs = {"num_class": self.nclass}
        self.evaluator = getattr(evaluation_module, cfg.EXPERIMENT.EVAL_METRIC)(cfg, **kwargs)

        # Define lr scheduler
        self.scheduler = LR_Scheduler(cfg.OPTIMIZER.LR_SCHEDULER, cfg.OPTIMIZER.LR,
                                      cfg.EXPERIMENT.EPOCHS, len(self.train_loader))



        self.start_epoch = 0
        self.epochs = cfg.EXPERIMENT.EPOCHS 
        # Resuming checkpoint
        self.best_pred = 0.0


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if cfg.SYSTEM.USE_GPU:
                image, target = image.to(self.device), target.to(self.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0 and epoch % 10 == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, cfg.DATASET.TRAIN, image, target, output, global_step, epoch, i)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d, Loss: %.3f]' % (epoch, i * cfg.INPUT.BATCH_SIZE_TRAIN + image.data.shape[0], train_loss))

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        num_img_val = len(self.val_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if cfg.SYSTEM.USE_GPU:
                image, target = image.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (test_loss / (i + 1)))
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, output)

            # Show 10 * 3 inference results each epoch
            if i % 5 == 0 and epoch % 10 == 0:
                global_step = i + num_img_val * epoch
                self.summary.visualize_image(self.writer, cfg.DATASET.TRAIN, image, target, output, global_step, epoch, i, validation=True)

        # Fast test during the training
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d, Loss: %.3f]' % (epoch, i * cfg.INPUT.BATCH_SIZE_TRAIN + image.data.shape[0], test_loss))
        new_pred = self.evaluator.compute_stats(writer=self.writer, epoch=epoch)

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        else:
            is_best = False 

        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.fuse_decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--exp_cfg', type=str, default=None, help='Configuration file for experiment (it overrides the default settings).')
    parser.add_argument('--model_name', type=str, choices=['eug_tiny','eug_base','eug_heter'], default='eug_base')
    args = parser.parse_args()
    
    cfg.EXPERIMENT.MODEL_NAME=args.model_name
    if args.exp_cfg is not None and os.path.isfile(args.exp_cfg):
        cfg.merge_from_file(args.exp_cfg)

    if not torch.cuda.is_available(): 
        print ("GPU is disabled")
        cfg.SYSTEM.USE_GPU = False


    if cfg.INPUT.BATCH_SIZE_TRAIN is None:
        cfg.INPUT.BATCH_SIZE_TRAIN = 4

    if cfg.INPUT.BATCH_SIZE_TEST is None:
        cfg.INPUT.BATCH_SIZE_TEST = 4

    if cfg.EXPERIMENT.NAME is None:
        cfg.EXPERIMENT.NAME = datetime.now().strftime(r'%Y%m%d_%H%M%S.%f').replace('.','_')

    sys.stdout = Logger(os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.NAME))
    torch.manual_seed(cfg.SYSTEM.RNG_SEED)
    np.random.seed(cfg.SYSTEM.RNG_SEED)

    trainer = Trainer(cfg)

    # print (trainer.model)
    #summary(trainer.model, input_size=(3, cfg.INPUT.CROP_SIZE, cfg.INPUT.CROP_SIZE))

    # print("Saving experiment to:", trainer.saver.experiment_dir) 
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.epochs)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        if (epoch % cfg.EXPERIMENT.EVAL_INTERVAL) == (cfg.EXPERIMENT.EVAL_INTERVAL - 1):
            trainer.validation(epoch)
    trainer.writer.close()