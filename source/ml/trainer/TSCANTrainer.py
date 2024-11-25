"""Trainer for DeepPhys."""
import numpy as np
import os
import torch
import torch.optim as optim

from source.evaluation.metrics_ppg import calculate_metrics_ppg
from source.evaluation.metrics_bp import calculate_metrics_bp
from source.ml.models.TSCAN import TSCAN
from source.ml.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class TSCANTrainer(BaseTrainer):
    def __init__(self, config, data_loader, trainer_params):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.cv_split = trainer_params['cv_split']
        self.ax_all = trainer_params['ax_all']
        self.model_dir = trainer_params['model_dir']
        self.log_training_dir = trainer_params['log_training_dir']
        self.log_testing_dir = trainer_params['log_testing_dir']
        self.test_metrics = trainer_params['test_metrics']
        device_ids_list = list(range(int(config.DEVICE[-1]), int(config.DEVICE[-1]) + config.NUM_OF_GPU_TRAIN))
        self.model = TSCAN(w=config.TRAIN.DATA.PREPROCESS.RESIZE.W, h=config.TRAIN.DATA.PREPROCESS.RESIZE.H,
                           in_channels=1, frame_depth=self.frame_depth).to(self.device)
        if len(device_ids_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids_list)

        if config.TOOLBOX_MODE == "train_and_test":
            # self.loss_model = Neg_Pearson()
            self.loss_model = torch.nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=len(data_loader["train"]))
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                label = batch[1][0].to(self.device)
                input_data = batch[0][0].to(self.device)
                N, D, C, H, W = input_data.shape
                input_data = input_data.view(N * D, C, H, W)
                input_data = input_data[:(N * D) // self.frame_depth * self.frame_depth]
                label = label.view(-1, 1)
                label = label[:(N * D) // self.frame_depth * self.frame_depth]
                pred = self.model(input_data)
                loss = self.loss_model(pred, label)
                running_loss += loss.item()
                if idx % 50 == 49:  # print every 50 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Optimizer and learning rate
                lrs.append(self.scheduler.get_last_lr())
                loss.backward()  # compute gradients
                self.optimizer.step()  # tell optimizer the gradients
                self.optimizer.zero_grad()  # reset gradients
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())

            # Append mean training loss and save model of current epoch
            mean_training_losses.append(np.mean(train_loss))
            self.save_model(self.model_dir, self.model, self.cv_split, epoch)

            # Calculate validation loss if not just last epoch is used for testing
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("Best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config, self.log_training_dir,
                                     self.cv_split)

        # Save file with best epoch
        best_epochs_path = self.log_training_dir + f'/cv{self.cv_split}_best_epoch.npy'
        np.save(best_epochs_path, np.asarray(self.best_epoch))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("====Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            if self.config.LABEL_SIGNALS[self.config.LABEL_VALID] in ['ppg_finger', 'ppg_ear', 'ppg_nose']:
                vbar = tqdm(data_loader["valid"], ncols=80)
                for idx, batch in enumerate(vbar):
                    vbar.set_description("Validation")
                    label = batch[1][0].to(self.device)
                    input_data = batch[0][0].to(self.device)
                    N, D, C, H, W = input_data.shape
                    input_data = input_data.view(N * D, C, H, W)
                    input_data = input_data[:(N * D) // self.frame_depth * self.frame_depth]
                    label = label.view(-1, 1)
                    label = label[:(N * D) // self.frame_depth * self.frame_depth]
                    pred = self.model(input_data)
                    loss = self.loss_model(pred, label)
                    valid_loss.append(loss.item())
                    valid_step += 1
                    vbar.set_postfix(loss=loss.item())
            else:
                raise ValueError("Label signal error! Please check label signal.")

            valid_loss = np.asarray(valid_loss)

        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("\n===Testing===")

        # Load model specified in config file
        model_path, used_epoch = self.get_model_path(self.config, self.model_dir, self.log_training_dir, self.cv_split,
                                                     self.max_epoch_num, self.best_epoch)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

        # Predict
        predictions = dict()
        labels = dict()
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        chunk_len_test = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
        with torch.no_grad():
            for _, batch in enumerate(data_loader['test']):
                batch_size = batch[0][0].shape[0]
                label = batch[1][0].to(self.config.DEVICE)
                input_data = batch[0][0].to(self.config.DEVICE)
                N, D, C, H, W = input_data.shape
                input_data = input_data.view(N * D, C, H, W)
                input_data = input_data[:(N * D) // self.frame_depth * self.frame_depth]
                label = label.view(-1, 1)
                label = label[:(N * D) // self.frame_depth * self.frame_depth]
                pred = self.model(input_data)

                for idx in range(batch_size):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred[idx * chunk_len_test:(idx + 1) * chunk_len_test]
                    labels[subj_index][sort_index] = label[idx * chunk_len_test:(idx + 1) * chunk_len_test]

        # Evaluate predicted outputs
        results = {label_signal: None for label_signal in self.config.LABEL_SIGNALS}
        for label_signal in self.config.LABEL_SIGNALS:
            if label_signal in ['ppg_finger', 'ppg_ear', 'ppg_nose']:
                results[label_signal] = calculate_metrics_ppg(predictions, labels, self.config, self.log_testing_dir +
                                                              f'/cv{self.cv_split}_Evaluation.txt', used_epoch,
                                                              self.test_metrics[label_signal])
            elif label_signal in ['sys', 'dia']:
                results[label_signal] = calculate_metrics_bp(predictions, labels, self.config, self.log_testing_dir +
                                                             f'/cv{self.cv_split}_Evaluation.txt', used_epoch,
                                                             self.ax_all, self.test_metrics[label_signal])
            else:
                raise ValueError("Label signal error! Please check label signal.")
        return results
