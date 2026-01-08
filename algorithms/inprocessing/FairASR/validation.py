import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
import torch
import copy
torch.set_float32_matmul_precision("high")
from omegaconf import OmegaConf, open_dict
from collections import OrderedDict
import os

_THIS_DIR   = os.path.dirname(__file__)
_DATA_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, 'metadata'))
_MODEL_ROOT  = os.path.abspath(os.path.join(_THIS_DIR, 'experiments'))

def run_validation():

    # ✅ 학습된 모델 경로 (파일명을 본인이 저장한 경로로 수정)
    model_path = _MODEL_ROOT + "/nemo_fairaudio/simclr_supconGRL_1e-1_diffspace/2026-01-07_14-46-16/checkpoints/simclr_supconGRL_1e-1_diffspace.nemo"
    # ✅ 여러 개의 Validation Manifest 파일 리스트
    val_manifests = [
    os.path.join(_DATA_ROOT, "age", "test_manifest_18.json"),
    # os.path.join(_DATA_ROOT, "gender", "test_manifest.json"),
    ]
    print("val_manifests", val_manifests)
    metrics_results={}

    # ✅ 학습된 모델 불러오기
    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path)
    cfg = copy.deepcopy(asr_model.cfg)
    with open_dict(cfg):
        cfg.validation_ds = OmegaConf.create({})  
        cfg.validation_ds.sample_rate = 16000  
        cfg.validation_ds.labels = cfg.labels
        cfg.validation_ds.batch_size =64
        cfg.validation_ds.num_workers = 8
        cfg.validation_ds.pin_memory = True
        cfg.validation_ds.trim_silence = True
        cfg.validation_ds.shuffle = False

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=16,
        logger=False
    )

    for val_manifest in val_manifests:
        manifest_name = os.path.basename(val_manifest)
        print(f"Evaluating on: {manifest_name}")

        with open_dict(cfg):
            cfg.validation_ds.manifest_filepath = val_manifest

        asr_model.setup_validation_data(val_data_config=cfg.validation_ds)
        results = trainer.validate(asr_model)

        # NeMo validation 결과를 그대로 OrderedDict로 변환
        metrics_results = OrderedDict([
            ('global_step', results[0].get("global_step", 0.0)),
            ('val_loss', results[0].get("val_loss", float('nan'))),
            ('val_wer', results[0].get("val_wer", float('nan')))
        ])

    return metrics_results
        
if __name__ == "__main__":
    metrics = run_validation()