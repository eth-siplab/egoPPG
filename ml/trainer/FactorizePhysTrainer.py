"""Trainer for DeepPhys."""
import numpy as np
import torch
import torch.optim as optim

from source.ml.loss.PhysNetNegPearsonLoss import Neg_Pearson
from source.ml.models.FactorizePhys.FactorizePhys import FactorizePhys
from source.ml.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class FactorizePhysTrainer(BaseTrainer):
    def __init__(self, config, data_loader, trainer_params):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.cv_split = trainer_params['cv_split']
        self.fig_all = trainer_params['fig_all']
        self.spec_all = trainer_params['spec_all']
        self.model_dir = trainer_params['model_dir']
        self.log_training_dir = trainer_params['log_training_dir']
        self.log_testing_dir = trainer_params['log_testing_dir']
        self.test_metrics = trainer_params['test_metrics']
        self.dropout_rate = config.MODEL.DROP_RATE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.no_labels = config.INFERENCE.NO_LABELS
        device_ids_list = list(range(int(config.DEVICE[-1]), int(config.DEVICE[-1]) + config.NUM_OF_GPU_TRAIN))

        # Model
        frames = self.config.MODEL.FactorizePhys.FRAME_NUM
        in_channels = self.config.MODEL.FactorizePhys.CHANNELS

        md_config = {}
        md_config["FRAME_NUM"] = self.config.MODEL.FactorizePhys.FRAME_NUM
        md_config["MD_TYPE"] = self.config.MODEL.FactorizePhys.MD_TYPE
        md_config["MD_FSAM"] = self.config.MODEL.FactorizePhys.MD_FSAM
        md_config["MD_TRANSFORM"] = self.config.MODEL.FactorizePhys.MD_TRANSFORM
        md_config["MD_S"] = self.config.MODEL.FactorizePhys.MD_S
        md_config["MD_R"] = self.config.MODEL.FactorizePhys.MD_R
        md_config["MD_STEPS"] = self.config.MODEL.FactorizePhys.MD_STEPS
        md_config["MD_INFERENCE"] = self.config.MODEL.FactorizePhys.MD_INFERENCE
        md_config["MD_RESIDUAL"] = self.config.MODEL.FactorizePhys.MD_RESIDUAL

        self.md_infer = self.config.MODEL.FactorizePhys.MD_INFERENCE
        self.use_fsam = self.config.MODEL.FactorizePhys.MD_FSAM

        self.model = FactorizePhys(frames=frames, md_config=md_config, in_channels=in_channels,
                                   dropout=self.dropout_rate, device=self.device).to(self.device)  # [3, T, 72,72]
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
            appx_error_list = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                label = batch[1][0].to(self.device)

                data = batch[0][0].to(self.device)
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                if self.use_fsam:
                    pred, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred, vox_embed = self.model(data)

                # pred = self.model(batch[0][0].to(self.device))

                if self.config.LABEL_SIGNALS[0] != 'hr':
                    label = (label - torch.mean(label)) / torch.std(label)  # normalize
                    pred = (pred - torch.mean(pred)) / torch.std(pred)  # normalize

                loss = self.loss_model(pred, label)
                running_loss += loss.item()
                if idx % 50 == 49:  # print every 50 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                if self.use_fsam:
                    appx_error_list.append(appx_error.item())

                # Adjust optimizer and learning rate
                lrs.append(self.scheduler.get_last_lr())
                loss.backward()  # compute gradients
                self.optimizer.step()  # tell optimizer the gradients
                self.optimizer.zero_grad()  # reset gradients
                self.scheduler.step()
                if self.use_fsam:
                    tbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                else:
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
            if self.config.LABEL_SIGNALS[self.config.LABEL_VALID] in ['ppg_finger', 'ppg_ear', 'ppg_nose', 'hr', 'ppg',
                                                                      'eda_raw', 'eda_filtered', 'eda_tonic', 'rr']:
                vbar = tqdm(data_loader["valid"], ncols=80)
                for idx, batch in enumerate(vbar):
                    vbar.set_description("Validation")
                    label = batch[1][0].to(self.device)

                    data = batch[0][0].to(self.device)
                    last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                    data = torch.cat((data, last_frame), 2)

                    if self.use_fsam:
                        pred, vox_embed, factorized_embed, appx_error = self.model(data)
                    else:
                        pred, vox_embed = self.model(data)


                    if self.config.LABEL_SIGNALS[0] != 'hr':
                        label = (label - torch.mean(label)) / torch.std(label)  # normalize
                        pred = (pred - torch.mean(pred)) / torch.std(pred)  # normalize
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
        predictions = {label_signal: {} for label_signal in self.config.LABEL_SIGNALS}
        labels = {label_signal: {} for label_signal in self.config.LABEL_SIGNALS}
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(data_loader['test']):
                batch_size = batch[0][0].shape[0]

                data = batch[0][0].to(self.device)
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                if self.use_fsam:
                    pred, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred, vox_embed = self.model(data)

                if self.no_labels:
                    label = pred
                else:
                    label = batch[1][0].to(self.config.DEVICE)

                for idx in range(batch_size):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    if subj_index not in predictions[self.config.LABEL_SIGNALS[0]].keys():
                        for label_signal in self.config.LABEL_SIGNALS:
                            predictions[label_signal][subj_index] = dict()
                            labels[label_signal][subj_index] = dict()
                    predictions[self.config.LABEL_SIGNALS[0]][subj_index][sort_index] = pred[idx]
                    labels[self.config.LABEL_SIGNALS[0]][subj_index][sort_index] = label[idx]

        # Evaluate predicted outputs
        results = self.evaluate_predictions(predictions, labels, self.config, self.log_testing_dir, self.cv_split,
                                            used_epoch, self.test_metrics, self.fig_all, self.spec_all)

        return results
