import torch

def FairASR():
    from MAF.algorithms.inprocessing.FairASR.train_simclr import run_train_simclr
    from MAF.algorithms.inprocessing.FairASR.train_hf import run_train_asr
    from MAF.algorithms.inprocessing.FairASR.validation import run_validation 
    
    print("[1/3] SimCLR Pretraining")
    run_train_simclr(
        independent_space=True,
        balance_param=0.1
    )
    torch.cuda.empty_cache()

    print("[2/3] ASR Fine-tuning")
    run_train_asr()
    torch.cuda.empty_cache()

    print("[3/3] Validation")
    metrics = run_validation()
    print('results:', metrics)
    print("✅ FairASR Pipeline Finished")
    return metrics


if __name__ == "__main__":
    metrics = FairASR()