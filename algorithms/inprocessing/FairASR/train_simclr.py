import wandb
import os
import copy
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from nemo.utils import exp_manager
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import ModelCheckpoint
import soundfile as sf

import json
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import argparse
try:
    torchaudio.set_audio_backend("sox_io")
except AttributeError:
    pass

torch.set_float32_matmul_precision("high")

_THIS_DIR   = os.path.dirname(__file__)
_DATA_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, 'metadata'))

def safe_load_audio(path):
        try:
            return torchaudio.load(path)
        except Exception:
            wav, sr = sf.read(path)
            if len(wav.shape) == 1:
                wav = wav[None, :]
            return torch.tensor(wav, dtype=torch.float32), sr
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)
    

class SpecAugment:
    def __init__(self):
        self.time_mask = T.TimeMasking(time_mask_param=30)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)

    def __call__(self, x):
        x_aug = self.time_mask(self.freq_mask(x))
        return x_aug

class SimCLRDataset(Dataset):
    def __init__(self, manifest_path, sample_rate=16000, augment_fn=None):
        self.data = []
        self.sample_rate = sample_rate
        self.augment_fn = augment_fn

        with open(manifest_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)
    
    
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        audio_path = sample["audio_filepath"]
        cls_label = sample["ethnicity"]
        # waveform, sr = torchaudio.load(audio_path)
        waveform, sr = safe_load_audio(audio_path)
        return waveform, cls_label

def simclr_collate_fn(batch):
    mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=80)
    spec_augment = T.TimeMasking(time_mask_param=30)

    spectrograms, aug_spectrograms, input_lengths, cls_labels = [], [], [], []

    for waveform, label in batch:
        spec = mel_transform(waveform)  # (1, 80, T)
        spec_aug = spec_augment(spec)  # Augmentation 적용

        spectrograms.append(spec.squeeze(0).T)  # (80, T) → (T, 80) 변환
        aug_spectrograms.append(spec_aug.squeeze(0).T)  # (80, T) → (T, 80)
        input_lengths.append(spec.shape[-1])  # T 값 저장 (시간축 길이)
        cls_labels.append(label)

    spectrograms_padded = pad_sequence(spectrograms, batch_first=True, padding_value=0.0)  # (B, T, 80)
    aug_spectrograms_padded = pad_sequence(aug_spectrograms, batch_first=True, padding_value=0.0)  # (B, T, 80)

    spectrograms_padded = spectrograms_padded.permute(0, 2, 1)  # (B, 80, T)
    aug_spectrograms_padded = aug_spectrograms_padded.permute(0, 2, 1)  # (B, 80, T)
    
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    cls_labels = torch.tensor(cls_labels)
    
    return spectrograms_padded, aug_spectrograms_padded, input_lengths, cls_labels


class SimCLR_Encoder(pl.LightningModule):
    def __init__(self, asr_model, independent_space=False, balance_param=0.1):
        super().__init__()

        self.encoder = asr_model.encoder  
        self.independent_space = independent_space
        self.balance_param = balance_param
        
        self.projection_head = nn.Sequential(
            nn.Linear(asr_model.encoder.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # Independent space를 사용하면 별도의 projection head 생성
        if independent_space:
            self.projection_head_sup = nn.Sequential(
                nn.Linear(asr_model.encoder.d_model, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )

        self.simclr_criterion = torch.nn.CrossEntropyLoss()
        self.contrast_mode = 'all'
        self.temperature = 0.2
        self.margin=0.5
        self.GRL = GradientReversalLayer()
        
    def forward(self, x, x_len):
        encoded, lens = self.encoder(audio_signal=x, length=x_len)  
        #print(encoded)
        #print(lens)
        encoded_rev = self.GRL(encoded)
        
        mp_enc = torch.mean(encoded, dim=2)
        mp_enc_rev = torch.mean(encoded_rev, dim=2)
        
        z = self.projection_head(mp_enc)
        # Independent space면 별도 projection head 사용, 아니면 같은 projection head 사용
        if self.independent_space:
            z_rev = self.projection_head_sup(mp_enc_rev)
        else:
            z_rev = self.projection_head(mp_enc_rev)
        return z, z_rev

    def training_step(self, batch, batch_idx):
        x, x_aug, x_len, cls_labels = batch  

        z1, z1_rev = self.forward(x, x_len)  
        z2, z2_rev = self.forward(x_aug, x_len)  
        
        simclr_loss = self.info_nce_loss(z1, z2)
        supcon_loss = self.supcon_loss(z1_rev,z2_rev, cls_labels)
        #loss = simclr_loss-self.balance_param * supcon_loss
        #cls_rep_loss = self.class_repulsion_loss(z1, z2, cls_labels)
        loss = simclr_loss + self.balance_param * supcon_loss
        
        current_lr = self.optimizers().param_groups[0]['lr']
        if batch_idx % 30 ==0:
            self.log("simclr_loss", simclr_loss, prog_bar=True, logger=True)
            self.log("supcon_loss", supcon_loss, prog_bar=True, logger=True)

            self.log("train_loss", loss, prog_bar=True, logger=True)
            self.log("learning_rate", current_lr, prog_bar=True)
            print(f"Step {batch_idx}: Train Loss = {loss.item()} | LR = {current_lr}")

        return loss

    def info_nce_loss(self, z1, z2, temperature=0.5):
        features = torch.cat([z1,z2], dim=0)
        N = z1.shape[0]
        labels = torch.cat([torch.arange(N) for i in range (2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z1.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)

        logits = logits / temperature
        loss = self.simclr_criterion(logits, labels)
        return loss

    def supcon_loss(self, z1, z2, labels=None, mask=None):
        features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            0.2)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        #loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    def class_repulsion_loss(self, z1, z2, labels):
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  
        labels = torch.cat([labels, labels], dim=0)  

        z = F.normalize(z, dim=1)  
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)  
        mask.fill_diagonal_(False)  

        positive_sim = similarity_matrix * mask.float()
        loss_positive = torch.relu(self.margin - positive_sim)  

        negative_sim = similarity_matrix * (~mask).float()
        loss_negative = torch.relu(negative_sim - (1 - self.margin))  

        loss = loss_positive.mean() + loss_negative.mean()  
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [{"scheduler":scheduler}]

def worker_init_fn(worker_id):
    import torchaudio
    try:
        torchaudio.set_audio_backend("sox_io")
    except AttributeError:
        pass

def main(independent_space=False, balance_param=0.1, resume_ckpt=None):
    """Main training function"""
    # Load dataset
    # train_manifest = "metadata/train_manifest_converted_small.json"
    train_manifest = os.path.join(_DATA_ROOT, "train_manifest_converted_small.json")
    train_dataset = SimCLRDataset(train_manifest)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, collate_fn=simclr_collate_fn, worker_init_fn=worker_init_fn)

    # Check data loading
    for batch in train_dataloader:
        x, x_aug, x_len, label = batch
        print(f"원본 데이터: {x.shape}, Augmented 데이터: {x_aug.shape}, label: {label}")
        #print(x_len)
        break

    # Independent space 옵션: True면 별도 projection head 사용, False면 같은 projection head 공유
    # independent_space = args.independent_space

    # Load ASR model and create SimCLR encoder
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_small", map_location='cpu')
    encoder_model = SimCLR_Encoder(
        asr_model, 
        independent_space=independent_space, 
        balance_param=balance_param)

    cfg = copy.deepcopy(asr_model.cfg)
    
    # Independent space 여부에 따라 experiment name 변경
    if independent_space:
        exp_name = 'simclr_supconGRL_diffspace'
    else:
        exp_name = 'simclr_supconGRL_samespace'

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"experiments/nemo_fairaudio_pretrain/{exp_name}_checkpoints",
        filename="asr_checkpoint-{epoch:02d}",  
        every_n_epochs=10,  
        save_top_k=-1,  
        save_last=True,  
        monitor="train_loss",
        mode="min"
    )

    # Setup WandB logger and trainer
    trainer = pl.Trainer(
        max_epochs=50,
        devices=1,
        strategy='auto',
        accelerator="gpu",
        #precision=16,
        logger=False,
        callbacks=[checkpoint_callback]
    )

    # Resume from checkpoint if provided
    # ckpt_path = args.resume_ckpt if args.resume_ckpt else None
    ckpt_path = resume_ckpt
    trainer.fit(encoder_model, train_dataloader, ckpt_path=ckpt_path)

    # Save pretrained encoder
    torch.save(encoder_model.encoder.state_dict(), "pretrained_encoder.pt")

def run_train_simclr(
    independent_space=False,
    balance_param=0.1,
    resume_ckpt=None
):
    main(
        independent_space=independent_space,
        balance_param=balance_param,
        resume_ckpt=resume_ckpt
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()

    run_train_simclr(
        independent_space=args.independent_space,
        balance_param=args.balance_param,
        resume_ckpt=args.resume_ckpt
    )
