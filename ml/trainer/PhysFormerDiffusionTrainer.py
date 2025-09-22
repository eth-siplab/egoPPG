"""Trainer for DeepPhys."""
import math
import numpy as np
import torch
import torch.optim as optim

from source.evaluation.metrics_bp import calculate_metrics_bp
from source.evaluation.metrics_hr import calculate_metrics_hr
from source.evaluation.metrics_ppg_rr import calculate_metrics_ppg_rr
from source.ml.loss.PhysNetNegPearsonLoss import Neg_Pearson
from source.ml.loss.PhysFormerLossComputer import TorchLossComputer
from source.ml.models.PhysFormerDiffusion import PhysFormerDiffusionUNet, PhysFormerDiffusionCNN
from source.ml.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch
from scipy.signal import butter, detrend, find_peaks, filtfilt, periodogram


class PhysFormerDiffusionTrainer(BaseTrainer):
    def __init__(self, config, data_loader, trainer_params):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.cv_split = trainer_params['cv_split']
        self.spec_all = trainer_params['spec_all']
        self.fig_all = trainer_params['fig_all']
        self.model_dir = trainer_params['model_dir']
        self.log_training_dir = trainer_params['log_training_dir']
        self.log_testing_dir = trainer_params['log_testing_dir']
        self.test_metrics = trainer_params['test_metrics']
        self.no_labels = config.INFERENCE.NO_LABELS

        # PhysFormer specific
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.dropout_rate = config.MODEL.DROP_RATE
        self.patch_size = config.MODEL.PHYSFORMER.PATCH_SIZE
        self.dim = config.MODEL.PHYSFORMER.DIM
        self.ff_dim = config.MODEL.PHYSFORMER.FF_DIM
        self.num_heads = config.MODEL.PHYSFORMER.NUM_HEADS
        self.num_layers = config.MODEL.PHYSFORMER.NUM_LAYERS
        self.theta = config.MODEL.PHYSFORMER.THETA
        self.frame_rate = config.TRAIN.DATA.FS
        if config.LABEL_SIGNALS[0] == 'rr':
            self.min_hr_rr = 8
            self.max_hr_rr = 36
        else:
            self.min_hr_rr = 30
            self.max_hr_rr = 180

        device_ids_list = list(range(int(config.DEVICE[-1]), int(config.DEVICE[-1]) + config.NUM_OF_GPU_TRAIN))
        self.model = PhysFormerDiffusionUNet(
            image_size=(self.chunk_len, config.TRAIN.DATA.PREPROCESS.RESIZE.H, config.TRAIN.DATA.PREPROCESS.RESIZE.W),
            patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads,
            num_layers=self.num_layers, dropout_rate=self.dropout_rate, theta=self.theta,
            diffusion_steps=50, device=self.device).to(self.device)
        if len(device_ids_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids_list)

        if config.TOOLBOX_MODE == "train_and_test":
            self.criterion_reg = torch.nn.MSELoss()
            self.criterion_L1loss = torch.nn.L1Loss()
            self.criterion_class = torch.nn.CrossEntropyLoss()
            # self.criterion_Pearson = Neg_Pearson()
            self.criterion_Pearson = torch.nn.MSELoss()
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

        a_start = 1.0
        b_start = 1.0
        exp_a = 0.5  # Unused
        exp_b = 1.0

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []

            loss_rPPG_avg = []
            loss_peak_avg = []
            loss_kl_avg_test = []
            loss_hr_mae = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                hr = torch.tensor([self.get_hr_rr(i, self.config.TRAIN.DATA.FS)
                                   for i in batch[1][0]]).float().to(self.device)
                data = batch[0][0].to(self.device)
                label = batch[1][0].to(self.device)

                self.optimizer.zero_grad()  # reset gradients

                gra_sharp = 2.0
                pred, diffusion_loss = self.model(data, gra_sharp, compute_diff_loss=True)

                if self.config.LABEL_SIGNALS[0] != 'hr':
                    label = (label - torch.mean(label)) / torch.std(label)  # normalize
                    pred = (pred - torch.mean(pred)) / torch.std(pred)  # normalize

                loss_rPPG = self.criterion_Pearson(pred, label)

                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(data.shape[0]):
                    loss_distribution_kl, fre_loss_temp, train_mae_temp = (
                        TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(pred[bb], hr[bb], self.frame_rate,
                                                                                     1.0, self.device, self.min_hr_rr,
                                                                                     self.max_hr_rr))
                    fre_loss = fre_loss+fre_loss_temp
                    kl_loss = kl_loss+loss_distribution_kl
                    train_mae = train_mae+train_mae_temp
                fre_loss /= data.shape[0]
                kl_loss /= data.shape[0]
                train_mae /= data.shape[0]

                if epoch > 10:
                    a = 0.05
                    b = 5.0
                else:
                    a = a_start
                    # exp ascend
                    b = b_start*math.pow(exp_b, epoch/10.0)

                loss = a*loss_rPPG + b*(fre_loss+kl_loss)

                # Define weighting hyperparameters for the losses
                lambda1 = 1.0  # weight for reconstruction loss
                lambda2 = 0.2  # weight for diffusion loss (adjust as needed)

                # Combine the losses into a total loss
                # total_loss = lambda1 * loss + lambda2 * diffusion_loss
                total_loss = diffusion_loss

                total_loss.backward()
                self.optimizer.step()

                n = data.size(0)
                loss_rPPG_avg.append(float(loss_rPPG.data))
                loss_peak_avg.append(float(fre_loss.data))
                loss_kl_avg_test.append(float(kl_loss.data))
                loss_hr_mae.append(float(train_mae))
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'\nepoch:{epoch}, batch:{idx + 1}, '
                        f'lr:0.0001, sharp:{gra_sharp:.3f}, a:{a:.3f}, NegPearson:{np.mean(loss_rPPG_avg[-2000:]):.4f}, '
                        f'\nb:{b:.3f}, kl:{np.mean(loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(loss_peak_avg[-2000:]):.3f}, '
                        f'hr_mae:{np.mean(loss_hr_mae[-2000:]):.3f}')

                """running_loss += loss.item()
                if idx % 50 == 49:  # print every 50 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Adjust optimizer and learning rate
                loss.backward()  # compute gradients
                self.optimizer.step()  # tell optimizer the gradients
                tbar.set_postfix(loss=loss.item())"""
            lrs.append(self.scheduler.get_last_lr())
            self.scheduler.step()

            # Append mean training loss and save model of current epoch
            mean_training_losses.append(np.mean(loss_rPPG_avg))
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
            if self.config.LABEL_SIGNALS[self.config.LABEL_VALID] in ['ppg_finger', 'ppg_ear', 'ppg_nose', 'hr', 'rr',
                                                                      'ppg']:
                hrs = []
                vbar = tqdm(data_loader["valid"], ncols=80)
                for idx, batch in enumerate(vbar):
                    vbar.set_description("Validation")
                    data = batch[0][0].to(self.device)
                    label = batch[1][0].to(self.device)
                    gra_sharp = 2.0
                    pred = self.model(data, gra_sharp)
                    if self.config.LABEL_SIGNALS[0] != 'hr':
                        label = (label - torch.mean(label)) / torch.std(label)  # normalize
                        pred = (pred - torch.mean(pred)) / torch.std(pred)  # normalize
                    # loss = self.loss_model(pred, label)
                    # valid_loss.append(loss.item())
                    # valid_step += 1
                    # vbar.set_postfix(loss=loss.item())
                    for _1, _2 in zip(pred, label):
                        hrs.append((self.get_hr_rr(_1.cpu().detach().numpy(), self.config.TRAIN.DATA.FS),
                                    self.get_hr_rr(_2.cpu().detach().numpy(), self.config.TRAIN.DATA.FS)))
                RMSE = np.mean([(i - j) ** 2 for i, j in hrs]) ** 0.5
            else:
                raise ValueError("Label signal error! Please check label signal.")

            # valid_loss = np.asarray(valid_loss)

        return RMSE

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
            for _, batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = batch[0][0].shape[0]
                data = batch[0][0].to(self.config.DEVICE)
                gra_sharp = 2.0
                pred = self.model(data, gra_sharp)
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

    # HR calculation based on ground truth label
    def get_hr_rr(self, y, fs):
        p, q = welch(y, fs, nfft=1e5/fs, nperseg=np.min((len(y)-1, 256)))
        return p[(p > self.min_hr_rr/60) & (p < self.max_hr_rr/60)][np.argmax(q[(p > self.min_hr_rr/60) & (p < self.max_hr_rr/60)])] * 60
