"""Trainer for DeepPhys."""
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from source.ml.loss.PhysNetNegPearsonLoss import Neg_Pearson
from source.ml.models.VQVAEIMU import VQVAEIMU
from source.ml.loss.FreqLoss import FreqLoss
from source.ml.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


def generate_soft_labels(targets, num_classes, sigma=2.0, temperature=1.0):
    """
    Generate soft labels with optional temperature scaling.

    Args:
        targets (torch.Tensor): Ground truth HR values (shape: [B]).
        num_classes (int): Number of classes (e.g., 141 for HR classes 40-180 BPM).
        sigma (float): Standard deviation for the Gaussian.
        temperature (float): Temperature parameter to adjust sharpness.

    Returns:
        torch.Tensor: Soft labels of shape [B, num_classes].
    """
    device = targets.device
    B = targets.size(0)
    class_indices = torch.arange(num_classes, device=device).float().unsqueeze(0).expand(B, num_classes)
    targets = targets.unsqueeze(1).float()
    # Create a Gaussian distribution over the classes.
    soft_labels = torch.exp(- ((class_indices - targets) ** 2) / (2 * sigma ** 2))
    # Apply temperature scaling: lower temperature makes distribution sharper.
    if temperature != 1.0:
        soft_labels = soft_labels ** (1.0 / temperature)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
    return soft_labels


class VQVAEIMUTrainer(BaseTrainer):
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
        self.min_hr = 40
        self.num_embeddings = 141
        self.temperature = 1.25
        self.sigma = 2.0
        device_ids_list = list(range(int(config.DEVICE[-1]), int(config.DEVICE[-1]) + config.NUM_OF_GPU_TRAIN))
        self.model = VQVAEIMU(frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
                              out_length=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
                              num_embeddings=self.num_embeddings,
                              commitment_cost=0.5).to(self.device)
        if len(device_ids_list) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids_list)

        if config.TOOLBOX_MODE == "train_and_test":
            # self.loss_model = Neg_Pearson()
            # self.loss_model = torch.nn.MSELoss()
            if self.config.LABEL_SIGNALS[0] in ['ppg_finger', 'ppg_ear', 'ppg_nose', 'sineppg']:
                self.loss_model = FreqLoss('ppg', config.TRAIN.DATA.FS, self.device)
            elif self.config.LABEL_SIGNALS[0] in ['rr']:
                self.loss_model = FreqLoss('rr', config.TRAIN.DATA.FS, self.device)
            else:
                raise ValueError("Label signal error! Please check label signal.")
            self.hr_loss_model = torch.nn.KLDivLoss(reduction='batchmean')
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

                pred, vq_loss, class_logits, encoding_indices = self.model(batch[0][0].to(self.device),
                                                                           batch[0][1].to(self.device))
                mean_gt_hr = torch.mean(batch[1][1], dim=1).to(self.device)
                # target_class = torch.clamp(torch.round(mean_gt_hr).long(), 0, self.num_embeddings - 1)
                # hr_loss = F.cross_entropy(class_logits, target_class)

                soft_labels = generate_soft_labels(mean_gt_hr, self.num_embeddings, sigma=self.sigma,
                                                   temperature=self.temperature)
                log_probs = F.log_softmax(class_logits, dim=1)
                hr_loss = self.hr_loss_model(log_probs, soft_labels)

                if self.config.LABEL_SIGNALS[0] != 'hr':
                    label = (label - torch.mean(label)) / torch.std(label)  # normalize
                    pred = (pred - torch.mean(pred)) / torch.std(pred)  # normalize
                signal_loss = self.loss_model(pred, label)
                loss = signal_loss + hr_loss + vq_loss

                running_loss += loss.item()
                if idx % 50 == 49:  # print every 50 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 50:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Adjust optimizer and learning rate
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
            if self.config.LABEL_SIGNALS[self.config.LABEL_VALID] in ['ppg_finger', 'ppg_ear', 'ppg_nose', 'hr',
                                                                      'eda_raw', 'eda_filtered', 'eda_tonic', 'rr',
                                                                      'sineppg']:
                vbar = tqdm(data_loader["valid"], ncols=80)
                for idx, batch in enumerate(vbar):
                    vbar.set_description("Validation")
                    label = batch[1][0].to(self.device)
                    pred, vq_loss, class_logits, encoding_indices = self.model(batch[0][0].to(self.device),
                                                                               batch[0][1].to(self.device))
                    mean_gt_hr = torch.mean(batch[1][1], dim=1).to(self.device)
                    # target_class = torch.clamp(torch.round(mean_gt_hr).long(), 0, self.num_embeddings - 1)
                    # hr_loss = F.cross_entropy(class_logits, target_class)

                    soft_labels = generate_soft_labels(mean_gt_hr, self.num_embeddings, self.sigma,
                                                       temperature=self.temperature)
                    log_probs = F.log_softmax(class_logits, dim=1)
                    hr_loss = self.hr_loss_model(log_probs, soft_labels)

                    if self.config.LABEL_SIGNALS[0] != 'hr':
                        label = (label - torch.mean(label)) / torch.std(label)  # normalize
                        pred = (pred - torch.mean(pred)) / torch.std(pred)  # normalize
                    signal_loss = self.loss_model(pred, label)
                    loss = signal_loss + hr_loss
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
                label = batch[1][0].to(self.config.DEVICE)
                pred, vq_loss, class_logits, encoding_indices = self.model(batch[0][0].to(self.config.DEVICE),
                                                                           batch[0][1].to(self.device))

                # Get HR classes
                predicted_hr_class = torch.argmax(class_logits, dim=1)
                predicted_hr_class = predicted_hr_class.unsqueeze(1).expand(-1, label.shape[1])
                gt_hr_class = batch[1][1].to(self.config.DEVICE)

                for idx in range(batch_size):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    if subj_index not in predictions[self.config.LABEL_SIGNALS[0]].keys():
                        for label_signal in self.config.LABEL_SIGNALS:
                            predictions[label_signal][subj_index] = dict()
                            labels[label_signal][subj_index] = dict()
                    predictions[self.config.LABEL_SIGNALS[0]][subj_index][sort_index] = pred[idx]
                    labels[self.config.LABEL_SIGNALS[0]][subj_index][sort_index] = label[idx]
                    predictions[self.config.LABEL_SIGNALS[1]][subj_index][sort_index] = torch.Tensor(predicted_hr_class[idx])
                    labels[self.config.LABEL_SIGNALS[1]][subj_index][sort_index] = torch.Tensor(gt_hr_class[idx])

        # Evaluate predicted outputs
        results = self.evaluate_predictions(predictions, labels, self.config, self.log_testing_dir, self.cv_split,
                                            used_epoch, self.test_metrics, self.fig_all, self.spec_all)

        return results
