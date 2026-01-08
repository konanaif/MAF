import os
import copy
import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from nemo.utils import exp_manager

import torch
torch.set_float32_matmul_precision("high")

_THIS_DIR   = os.path.dirname(__file__)
_DATA_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, 'metadata'))
_MODEL_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, 'experiments'))

def run_train_asr():
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_small", map_location='cpu')
    path = _MODEL_ROOT + '/nemo_fairaudio_pretrain/simclr_supconGRL_samespace_checkpoints/last.ckpt'
    simclr_checkpoint = torch.load(path, map_location='cpu')
    simclr_encoder_state_dict = {k.replace("encoder.", ""): v for k, v in simclr_checkpoint["state_dict"].items() if k.startswith("encoder.")}
    asr_model.encoder.load_state_dict(simclr_encoder_state_dict)

    cfg = copy.deepcopy(asr_model.cfg)

    # train_manifest = "metadata/train_manifest.json"
    # test_manifest = "metadata/test_manifest.json"
    
    train_manifest = os.path.join(_DATA_ROOT, "train_manifest.json")
    test_manifest = os.path.join(_DATA_ROOT, "test_manifest.json")
    
    with open(train_manifest, "r") as f:
        train_vocab = set()
        for line in f:
            text = eval(line)["text"]
            train_vocab.update(text)

    with open_dict(cfg):
        cfg.labels = list(train_vocab) 

        cfg.train_ds = OmegaConf.create({}) 
        cfg.validation_ds = OmegaConf.create({})  
        cfg.train_ds.sample_rate = 16000  

        cfg.train_ds.manifest_filepath = train_manifest
        cfg.train_ds.labels = cfg.labels
        cfg.train_ds.batch_size = 32
        cfg.train_ds.num_workers = 8
        cfg.train_ds.shuffle = True
        cfg.train_ds.pin_memory = True
        cfg.train_ds.trim_silence = True  
        
        cfg.validation_ds.sample_rate = 16000  
        cfg.validation_ds.manifest_filepath = test_manifest
        cfg.validation_ds.labels = cfg.labels
        cfg.validation_ds.batch_size =32
        cfg.validation_ds.num_workers = 8
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True
        cfg.validation_ds.shuffle = False

    asr_model.setup_training_data(cfg.train_ds)
    asr_model.setup_validation_data(cfg.validation_ds)

    trainer = pl.Trainer(devices=1,
        max_epochs=50,
        accelerator="gpu",
        strategy="auto",
        logger=False,
        log_every_n_steps=10,  
        check_val_every_n_epoch=1,
        callbacks=[],
        enable_checkpointing=False
    )

    asr_model.cfg = cfg

    with open_dict(asr_model.cfg.optim):
        asr_model.cfg.optim.lr = 1e-4  
        asr_model.cfg.optim.betas = [0.9, 0.999]
        asr_model.cfg.optim.weight_decay = 0.01
        asr_model.cfg.optim.sched = {
            "name": "CosineAnnealing",  
            "warmup_steps": None,  
            "warmup_ratio": 0.05,  
            "min_lr": 1e-5,  
            "max_steps": None,  
        }
        
    exp_config = exp_manager.ExpManagerConfig(
        exp_dir='experiments/nemo_fairaudio',
        name="simclr_supconGRL_1e-1_diffspace",
        checkpoint_callback_params=exp_manager.CallbackParams(
            monitor="val_wer",
            mode="min",
            always_save_nemo=True,
            save_best_model=True,
        ),
    )

    config = OmegaConf.structured(exp_config)

    logdir = exp_manager.exp_manager(trainer, config)

    asr_model.set_trainer(trainer)  

    trainer.fit(asr_model)

    asr_model.save_to("fine_tuned_asr.nemo")


if __name__ == "__main__":
    run_train_asr()