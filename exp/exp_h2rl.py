"""
Experiment class for H²RL model
Place this file in: exp/exp_h2rl.py
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.augmentations import masked_data
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')


class Exp_H2RL(Exp_Basic):
    def __init__(self, args):
        super(Exp_H2RL, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        # Import H2RL model
        from models.H2RL import Model
        model = Model(self.args).float()

        if self.args.load_checkpoints:
            print(f"Loading checkpoint: {self.args.load_checkpoints}")
            try:
                checkpoint = torch.load(self.args.load_checkpoints, map_location=self.device)
                
                # Extract state dict
                if 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                else:
                    pretrained_dict = checkpoint
                
                # Filter out task-specific heads (keep only encoder parameters)
                encoder_dict = {}
                head_keys_skipped = []
                
                for k, v in pretrained_dict.items():
                    # Skip task-specific heads
                    if 'reconstruction_head' in k or 'forecast_head' in k:
                        head_keys_skipped.append(k)
                        continue
                    encoder_dict[k] = v
                
                # Load encoder parameters with strict=False
                missing_keys, unexpected_keys = model.load_state_dict(encoder_dict, strict=False)
                
                # Count actual parameters loaded
                loaded_params = sum(p.numel() for k, p in model.named_parameters() if k in encoder_dict)
                total_params = sum(p.numel() for p in model.parameters())
                
                # Report what was loaded
                print(f"✓ Successfully loaded {loaded_params:,} / {total_params:,} parameters from checkpoint")
                print(f"✓ Loaded {len(encoder_dict)} weight tensors (skipped {len(head_keys_skipped)} head tensors)")
                print(f"✓ Missing keys (randomly initialized): {len(missing_keys)}")
                print(f"✓ Forecast head initialized randomly for fine-tuning")
                
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {str(e)}")
                print("Starting from scratch...")

        if torch.cuda.device_count() > 1 and self.args.use_multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print(f'Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):
        """Pre-training with masked modeling and contrastive learning"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # Optimizer
        model_optim = self._select_optimizer()
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=model_optim,
            T_max=self.args.train_epochs
        )

        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss, train_cl_loss, train_rb_loss = self.pretrain_one_epoch(
                train_loader, model_optim, model_scheduler
            )
            vali_loss, vali_cl_loss, vali_rb_loss = self.valid_one_epoch(vali_loader)

            # Log and Loss
            end_time = time.time()
            print(
                f"Epoch: {epoch}, Lr: {model_scheduler.get_last_lr()[0]:.7f}, "
                f"Time: {end_time - start_time:.2f}s | "
                f"Train Loss: {train_loss:.4f}/{train_cl_loss:.4f}/{train_rb_loss:.4f} "
                f"Val Loss: {vali_loss:.4f}/{vali_cl_loss:.4f}/{vali_rb_loss:.4f}"
            )

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'vali_loss': vali_loss,
                'valid_cl_loss': vali_cl_loss,
                'valid_rb_loss': vali_rb_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # Checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    f"Validation loss decreased ({min_vali_loss:.4f} --> {vali_loss:.4f}). "
                    f"Saving model epoch {epoch}..."
                )

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'module.' in k:
                        k = k.replace('module.', '')  # multi-gpu
                    self.encoder_state_dict[k] = v
                
                encoder_ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.encoder_state_dict,
                    'optimizer_state_dict': model_optim.state_dict(),
                    'loss': vali_loss
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print(f"Saving model at epoch {epoch + 1}...")

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'module.' in k:
                        k = k.replace('module.', '')
                    self.encoder_state_dict[k] = v
                
                encoder_ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.encoder_state_dict,
                    'optimizer_state_dict': model_optim.state_dict()
                }
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        """One epoch of pre-training"""
        train_loss = []
        train_cl_loss = []
        train_rb_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            # Data augmentation (masking)
            batch_x_m, batch_x_mark_m, mask = masked_data(
                batch_x, batch_x_mark,
                self.args.mask_rate,
                self.args.lm,
                self.args.positive_nums
            )
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            batch_x = batch_x.float().to(self.device)

            # Masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # To device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # Encoder
            loss, loss_cl, loss_rb, _, _, _, _ = self.model(
                batch_x_om, batch_x_mark, batch_x, mask=mask_om
            )

            # Backward
            loss.backward()
            model_optim.step()

            # Record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_cl_loss = np.average(train_cl_loss)
        train_rb_loss = np.average(train_rb_loss)

        return train_loss, train_cl_loss, train_rb_loss

    def valid_one_epoch(self, vali_loader):
        """One epoch of validation"""
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Data augmentation
                batch_x_m, batch_x_mark_m, mask = masked_data(
                    batch_x, batch_x_mark,
                    self.args.mask_rate,
                    self.args.lm,
                    self.args.positive_nums
                )
                batch_x_om = torch.cat([batch_x, batch_x_m], 0)

                # Masking matrix
                mask = mask.to(self.device)
                mask_o = torch.ones(size=batch_x.shape).to(self.device)
                mask_om = torch.cat([mask_o, mask], 0).to(self.device)

                # To device
                batch_x = batch_x.float().to(self.device)
                batch_x_om = batch_x_om.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Encoder
                loss, loss_cl, loss_rb, _, _, _, _ = self.model(
                    batch_x_om, batch_x_mark, batch_x, mask=mask_om
                )

                # Record
                valid_loss.append(loss.item())
                valid_cl_loss.append(loss_cl.item())
                valid_rb_loss.append(loss_rb.item())

        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)

        self.model.train()
        return vali_loss, valid_cl_loss, valid_rb_loss

    def train(self, setting):
        """Fine-tuning for forecasting"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # To device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Loss
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            end_time = time.time()
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, Time: {end_time - start_time:.2f}s | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_loader, criterion):
        """Validation"""
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Encoder
                outputs = self.model(batch_x, batch_x_mark)

                # Loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # Record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self):
        """Testing"""
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        # Fix: Concatenate along batch dimension instead of creating nested arrays
        preds = np.concatenate(preds, axis=0)  # Shape: (total_samples, pred_len, n_vars)
        trues = np.concatenate(trues, axis=0)  # Shape: (total_samples, pred_len, n_vars)

        # Verify shapes match
        print(f'Test shapes - Predictions: {preds.shape}, Ground Truth: {trues.shape}')

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'{self.args.data} {self.args.seq_len}->{self.args.pred_len}, mse:{mse:.3f}, mae:{mae:.3f}')

        f = open("./outputs/score.txt", 'a')
        f.write(f'{self.args.data} {self.args.seq_len}->{self.args.pred_len}, {mse:.3f}, {mae:.3f}\n')
        f.close()
