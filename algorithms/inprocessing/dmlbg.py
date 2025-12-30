import torch, math, time, argparse, os, random
import os, sys
# BASE_DIR = "/home/eunji.lee/work/IITP/2025/MAF-DEMO/MAF/algorithms/inprocessing"
# DMLBG_PATH = os.path.join(BASE_DIR, "DMLBG")

from MAF.algorithms.inprocessing.DMLBG import dataset, utils, losses, net
import numpy as np

from MAF.algorithms.inprocessing.DMLBG.dataset.Inshop import Inshop_Dataset
from MAF.algorithms.inprocessing.DMLBG.net.resnet import *
from MAF.algorithms.inprocessing.DMLBG.net.googlenet import *
from MAF.algorithms.inprocessing.DMLBG.net.bn_inception import *
from MAF.algorithms.inprocessing.DMLBG.dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import argparse
import pdb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# --- ADD: 프록시 텐서 추출 유틸 ---
def _extract_proxies_tensor(crit):
    """
    Proxy-Anchor 등 criterion에서 프록시(클래스 센터) 텐서를 최대한 안전하게 꺼낸다.
    반환: torch.Tensor(shape: [C,D]) 또는 None
    """
    # 가장 흔한 패턴: nn.Embedding 형태로 보관
    if hasattr(crit, "proxies"):
        P = getattr(crit, "proxies")
        # nn.Embedding, nn.Linear.weight, nn.Parameter 모두 커버
        if hasattr(P, "weight") and isinstance(P.weight, torch.Tensor):
            return P.weight.detach().cpu()
        if isinstance(P, torch.Tensor):
            return P.detach().cpu()
        if isinstance(P, torch.nn.Parameter):
            return P.data.detach().cpu()

    # 다른 키워드 힌트 (혹시 클래스 이름이 다르거나 변수명이 다른 경우)
    for name in ["proxy", "centers", "class_centers", "proxies_mu"]:
        if hasattr(crit, name):
            obj = getattr(crit, name)
            if hasattr(obj, "weight") and isinstance(obj.weight, torch.Tensor):
                return obj.weight.detach().cpu()
            if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
                return obj.detach().cpu() if isinstance(obj, torch.Tensor) else obj.data.detach().cpu()
    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOG_DIR', default='../logs', help = 'Path to log folder')
    parser.add_argument('--dataset', default='dep', help = 'Training dataset, e.g. cub, cars, SOP, Inshop')
    parser.add_argument('--embedding-size', default = 512, type = int, dest = 'sz_embedding', help = 'Size of embedding that is appended to backbone model.')
    parser.add_argument('--batch-size', default = 180, type = int, dest = 'sz_batch', help = 'Number of samples per batch.')
    parser.add_argument('--epochs', default = 60, type = int, dest = 'nb_epochs', help = 'Number of training epochs.')
    parser.add_argument('--gpu-id', default = 0, type = int, help = 'ID of GPU that is used for training.')
    parser.add_argument('--workers', default = 4, type = int, dest = 'nb_workers', help = 'Number of workers for dataloader.')
    parser.add_argument('--model', default = 'bn_inception', help = 'Model for training')
    parser.add_argument('--loss', default = 'Proxy_Anchor', help = 'Criterion for training')
    parser.add_argument('--optimizer', default = 'adamw', help = 'Optimizer setting')
    parser.add_argument('--lr', default = 1e-4, type =float, help = 'Learning rate setting')
    parser.add_argument('--weight-decay', default = 1e-4, type =float, help = 'Weight decay setting')
    parser.add_argument('--lr-decay-step', default = 10, type =int, help = 'Learning decay step setting')
    parser.add_argument('--lr-decay-gamma', default = 0.5, type =float, help = 'Learning decay gamma setting')
    parser.add_argument('--alpha', default = 32, type = float, help = 'Scaling Parameter setting')
    parser.add_argument('--mrg', default = 0.1, type = float, help = 'Margin parameter setting')
    parser.add_argument('--IPC', type = int, help = 'Balanced sampling, images per class')
    parser.add_argument('--warm', default = 1, type = int, help = 'Warmup training epochs')
    parser.add_argument('--bn-freeze', default = 1, type = int, help = 'Batch normalization parameter freeze')
    parser.add_argument('--l2-norm', default = 1, type = int, help = 'L2 normlization')
    parser.add_argument('--remark', default = '', help = 'Any remark')
    parser.add_argument('--disent', default = False, type = bool, help = 'Attach the DDML module')
    parser.add_argument('--al', default = 0.0, type = float, help = 'DDML hyperparameter alpha setting')
    parser.add_argument('--be', default = 0.0, type = float, help = 'DDML hyperparameter beta setting')
    parser.add_argument('--gam', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--de', default = 0.0, type = float, help = 'DDML hyperparameter beta setting')
    parser.add_argument('--eps', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--ze', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--eta', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--theta', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--iota', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--kap', default = 0.0, type = float, help = 'DDML hyperparameter lambda setting')
    parser.add_argument('--bg-start', default=0.0, type=int, help='이 에폭부터 z_bg 학습/분리 손실 활성화 (그 전에는 꺼짐)')
    return parser.parse_args()

def dmlbg_main(args):
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        
    LOG_DIR = os.path.join(args.LOG_DIR, f'logs_{args.dataset}_{args.model}_{args.loss}_embedding{args.sz_embedding}_al{args.al}_be{args.be}_gam{args.gam}_{args.optimizer}_lr{args.lr}_batch{args.sz_batch}{args.remark}')
        
    # Dataset directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    os.chdir(DATA_DIR)
    data_root = os.getcwd()
    
    # Dataset Loader and Sampler
    if args.dataset != 'Inshop':
        trn_dataset = dataset.load(
                name = args.dataset,
                root = data_root,
                mode = 'train',
                transform = dataset.utils.make_transform(
                    is_train = True, 
                    is_inception = (args.model == 'bn_inception')
                ))
    else:
        trn_dataset = Inshop_Dataset(
                root = data_root,
                mode = 'train',
                transform = dataset.utils.make_transform(
                    is_train = True, 
                    is_inception = (args.model == 'bn_inception')
                ))

    if args.IPC:
        balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class = args.IPC)
        batch_sampler = BatchSampler(balanced_sampler, batch_size = args.sz_batch, drop_last = True)
        dl_tr = torch.utils.data.DataLoader(trn_dataset, num_workers = args.nb_workers, pin_memory = True, batch_sampler = batch_sampler)
    else:
        dl_tr = torch.utils.data.DataLoader(trn_dataset, batch_size = args.sz_batch, shuffle = True, num_workers = args.nb_workers, drop_last = True, pin_memory = True)

    if args.dataset != 'Inshop':
        ev_dataset = dataset.load(name = args.dataset, root = data_root, mode = 'eval', transform = dataset.utils.make_transform(is_train = False, is_inception = (args.model == 'bn_inception')))
        dl_ev = torch.utils.data.DataLoader(ev_dataset, batch_size = args.sz_batch, shuffle = False, num_workers = args.nb_workers, pin_memory = True)
        print("ev_dataset", ev_dataset)
        print("dl_ev", dl_ev)
        
    else:
        query_dataset = Inshop_Dataset(root = data_root, mode = 'query', transform = dataset.utils.make_transform(is_train = False, is_inception = (args.model == 'bn_inception')))
        dl_query = torch.utils.data.DataLoader(query_dataset, batch_size = args.sz_batch, shuffle = False, num_workers = args.nb_workers, pin_memory = True)
        gallery_dataset = Inshop_Dataset(root = data_root, mode = 'gallery', transform = dataset.utils.make_transform(is_train = False, is_inception = (args.model == 'bn_inception')))
        dl_gallery = torch.utils.data.DataLoader(gallery_dataset, batch_size = args.sz_batch, shuffle = False, num_workers = args.nb_workers, pin_memory = True)

    nb_classes = trn_dataset.nb_classes()
    # ---- best recall tracking ----
    if args.dataset == 'Inshop':
        Ks = [1, 10, 20, 30, 40, 50]
    elif args.dataset != 'SOP' :
        Ks = [1, 2, 4, 8, 16, 32]
    else:
        Ks = [1, 10, 100, 1000]

    best_recalls = np.zeros(len(Ks), dtype=np.float32)  # 각 K 별 최고값
    best_epoch_k = [-1] * len(Ks)                       # 각 K 최고가 나온 epoch
    best_epoch = -1                                     # (top-1 기준) 최고 epoch
    
    # Backbone Model
    if args.model.find('googlenet')+1:
        model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
    elif args.model.find('bn_inception')+1:
        model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze, disent= args.disent)
    elif args.model.find('resnet18')+1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
    elif args.model.find('resnet50')+1:
        model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze, disent= args.disent)
    elif args.model.find('resnet101')+1:
        model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = args.bn_freeze)
    model = model.cuda()

    if args.gpu_id == -1:
        model = nn.DataParallel(model)

    # DML Losses
    if args.loss == 'Proxy_Anchor':
        criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args.sz_embedding, mrg = args.mrg, alpha = args.alpha, al=args.al, be=args.be, gam=args.gam, de = args.de, eps = args.eps, ze = args.ze, eta=args.eta, theta=args.theta).cuda()
    elif args.loss == 'Proxy_NCA':
        criterion = losses.Proxy_NCA(nb_classes = nb_classes, sz_embed = args.sz_embedding).cuda()
    elif args.loss == 'MS':
        criterion = losses.MultiSimilarityLoss().cuda()
    elif args.loss == 'Contrastive':
        criterion = losses.ContrastiveLoss().cuda()
    elif args.loss == 'Triplet':
        criterion = losses.TripletLoss().cuda()
    elif args.loss == 'NPair':
        criterion = losses.NPairLoss().cuda()

    # Train Parameters
    param_groups = [
        {
            'params':
                list(set(model.parameters()).difference(
                    set(list(model.model.embedding.parameters())
                        + list(model.model.embedding_bg.parameters())
                        + list(model.model.embedding_specific.parameters()))
                )) if args.gpu_id != -1 else
                list(set(model.module.parameters()).difference(
                    set(list(model.module.model.embedding.parameters())
                        + list(model.module.model.embedding_bg.parameters())
                        + list(model.module.model.embedding_specific.parameters()))
                ))
        },
        {'params': (model.model.embedding if args.gpu_id != -1 else model.module.model.embedding).parameters(), 'lr': float(args.lr) * 1},
        {'params': (model.model.embedding_bg if args.gpu_id != -1 else model.module.model.embedding_bg).parameters(), 'lr': float(args.lr) * 1},
        {'params': (model.model.embedding_specific if args.gpu_id != -1 else model.module.model.embedding_specific).parameters(), 'lr': float(args.lr) * 1},
    ]
    if args.loss == 'Proxy_Anchor':
        param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr) * 100})
    elif args.loss == 'Proxy_NCA':
        param_groups.append({'params': criterion.parameters(), 'lr':float(args.lr)})

    # Optimizer Setting
    if args.optimizer == 'sgd': 
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay = args.weight_decay, momentum = 0.9, nesterov=True)
    elif args.optimizer == 'adam': 
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay = args.weight_decay, momentum = 0.9)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
        
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma = args.lr_decay_gamma)

    print("Training parameters: {}".format(vars(args)))
    print("Training for {} epochs.".format(args.nb_epochs))
    losses_list = []
    best_recall=[0]
    best_epoch = 0

    for epoch in range(0, args.nb_epochs):
        model.train()
        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            for m in modules: 
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # >>> ADDED: z_bg 하드 스타트 (n 에폭 이후에만 켜기)
        # param_groups 구성 순서(위에서 정의한 것과 동일해야 함):
        # 0: backbone 나머지, 1: embedding(z), 2: embedding_bg(z_bg), 3: embedding_specific(z_s), 4+: criterion
        IDX_EMB_BG = 2
        bg_on = (epoch >= args.bg_start)

        # (A) z_bg 헤드 freeze/unfreeze
        emb_bg = (model.model.embedding_bg if args.gpu_id != -1
                else model.module.model.embedding_bg)
        for p in emb_bg.parameters():
            p.requires_grad = bg_on

        # (B) z_bg/분리 손실 가중치 on/off
        #     losses.Proxy_Anchor 내부 속성명이 아래와 동일해야 함(de, eps, ze, eta).
        #     다르면 해당 이름으로 바꿔줘.
        target_de  = args.de   if bg_on else 0.0   # z_bg agnostic
        target_eps = args.eps  if bg_on else 0.0   # z_bg KL
        target_ze  = args.ze   if bg_on else 0.0   # split/HSIC
        target_eta = args.eta  if bg_on else 0.0   # ortho
        target_theta = args.theta  if bg_on else 0.0
        target_iota = args.iota if bg_on else 0.0
        target_kap = args.kap if bg_on else 0.0
        for name, val in [('de', target_de), ('eps', target_eps),
                        ('ze', target_ze), ('eta', target_eta),('theta', target_theta),('iota',target_iota),('kap',target_kap)]:
            if hasattr(criterion, name):
                setattr(criterion, name, val)

        # (C) 옵티마에서 z_bg 파라미터 그룹 LR on/off
        base_lr_bg = float(args.lr) * 1.0   # 너의 param_groups에서 z_bg head LR 배수 1로 설정되어 있었음
        opt.param_groups[IDX_EMB_BG]['lr'] = (base_lr_bg if bg_on else 0.0)

        # (D) (선택) epoch 경계에서만 로그
        if epoch == 0 or epoch == args.bg_start:
            print(f"[z_bg] epoch {epoch}: bg_on={bg_on} | "
                f"de={target_de} eps={target_eps} ze={target_ze} eta={target_eta} theta={target_theta} iota={target_iota} kap={target_kap}| "
                f"lr_bg={opt.param_groups[IDX_EMB_BG]['lr']}")

        losses_per_epoch = []
        
        # Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = (
                    list(model.model.embedding.parameters())
                    + list(model.model.embedding_bg.parameters())
                    + list(model.model.embedding_specific.parameters())
                    + list(criterion.parameters())
                )
            else:
                unfreeze_model_param = (
                    list(model.module.model.embedding.parameters())
                    + list(model.module.model.embedding_bg.parameters())
                    + list(model.module.model.embedding_specific.parameters())
                    + list(criterion.parameters())
                )
            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        pbar = tqdm(enumerate(dl_tr))

        for batch_idx, (x, y) in pbar:
            if args.disent:
                output_emb, output_emb_bg, output_emb_specific = model(x.squeeze().cuda())
                # ↓ criterion 시그니처가 세 임베딩을 받도록 구현되어 있어야 함
                loss, logit, logit_bg, logit_spcf, loss_AGReg = criterion(
                    output_emb, output_emb_bg, output_emb_specific,
                    y.squeeze().cuda(), args.disent
                )                
        
            else :
                m = model(x.squeeze().cuda())
                loss = criterion(m, y.squeeze().cuda())
                
            log_payload = {'loss/total': loss.item()}
            if args.disent:
                names = [
                    'loss/agnostic',   # 0
                    'loss/specific',   # 1
                    'loss/zs_kl',      # 2
                    'loss/bg',         # 3
                    'loss/bg_kl',      # 4
                    'loss/split',      # 5
                    'loss/ortho',      # 6
                    'loss/bg_ent',     # 7
                    'loss/bg_repulsion', #8
                    'loss/bg_prx_ortho', #9
                ]
                for i, v in enumerate(loss_AGReg):
                    key = names[i] if i < len(names) else f'loss/extra_{i}'
                    log_payload[key] = v.item()
            # replace wandb.log with print
            print(f"[LOG] step={epoch * len(dl_tr) + batch_idx} payload={log_payload}")
            opt.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == 'Proxy_Anchor':
                torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

            losses_per_epoch.append(loss.data.cpu().numpy())
            opt.step()

            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}- [agn: {:.6f}, spe: {:.6f}, spli: {:.6f}]'.format(
                    epoch, batch_idx + 1, len(dl_tr),
                    100. * batch_idx / len(dl_tr),
                    loss.item(), loss_AGReg[0].item(), loss_AGReg[1].item(), loss_AGReg[2].item()))
            
        losses_list.append(np.mean(losses_per_epoch))
        # replaced wandb.log({'loss': ...}) with print
        print(f"[EPOCH] epoch={epoch} mean_loss={losses_list[-1]}")
        scheduler.step()

        ... 

        
        if(epoch >= 0):
            with torch.no_grad():
                print("**Evaluating...**")
                if args.dataset == 'Inshop':
                    Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
                elif args.dataset != 'SOP':
                    Recalls = utils.evaluate_cos(model, dl_ev)
                else:
                    Recalls = utils.evaluate_cos_SOP(model, dl_ev)

            # ---- 현재 epoch의 R@K 로깅 ----
            for i, K in enumerate(Ks):
                # replaced wandb.log with print
                print(f"[R@K] epoch={epoch} R@{K}={Recalls[i]}")

            # ---- 현재 vs. 최고치 콘솔 출력 (업데이트 전, '이전 최고치'와 비교) ----
            lines = []
            for i, K in enumerate(Ks):
                cur = Recalls[i] * 100.0
                prev_best = best_recalls[i] * 100.0
                prev_ep = best_epoch_k[i] if best_epoch_k[i] != -1 else "-"
                lines.append(f"R@{K}: {cur:.2f}% | best {prev_best:.2f}% (epoch {prev_ep})")
            print(" | ".join(lines))

            # ---- 최고치 갱신 시 즉시 기록 ----
            prev_best_top1 = best_recalls[0]
            improved_top1 = Recalls[0] > prev_best_top1

            for i, K in enumerate(Ks):
                if Recalls[i] > best_recalls[i]:
                    best_recalls[i] = Recalls[i]
                    best_epoch_k[i] = epoch

                    payload = {
                        f"best/R@{K}": float(best_recalls[i]),
                        f"best/epoch_R@{K}": int(best_epoch_k[i]),
                        "epoch": int(epoch)
                    }
                    # replaced wandb.log with print
                    print(f"[BEST LOG] payload={payload}")

                    # replaced wandb.run.summary updates with print
                    print(f"[BEST SUMMARY] best/R@{K}={best_recalls[i]}, best/epoch_R@{K}={best_epoch_k[i]}")

                    # 갱신 알림 출력
                    print(f"[NEW BEST] R@{K} -> {best_recalls[i]*100:.2f}% at epoch {epoch}")


            # ---- R@1 기준 Best 모델 저장 ----
            if improved_top1:
                best_recall = Recalls  # 파일로 쓰기 위해 전체 벡터 저장(기존 코드 호환)
                best_epoch = epoch
                if not os.path.exists('{}'.format(LOG_DIR)):
                    os.makedirs('{}'.format(LOG_DIR))
                # --- REPLACE: 베스트 체크포인트 저장 (프록시 포함) ---
                save_path = '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model)

                # 프록시 텐서 추출
                P_tensor = _extract_proxies_tensor(criterion)

                ckpt = {
                    'model_state_dict': (model.state_dict() if args.gpu_id != -1 else model.module.state_dict()),
                    'criterion_state_dict': criterion.state_dict(),
                    # 찾기 쉬운 키 이름들을 함께 저장 (gradcam 스크립트가 자동 탐색)
                    'proxies': P_tensor,                 # 널 수도 있음
                    'proxies.weight': P_tensor,          # 중복 저장(탐색 편의)
                    'criterion.proxies.weight': P_tensor # 중복 저장(탐색 편의)
                }
                torch.save(ckpt, save_path)

                # 프록시만 별도 파일로도 저장(선택: gradcam에서 직접 로드 가능)
                if P_tensor is not None:
                    os.makedirs(f"{LOG_DIR}/aux", exist_ok=True)
                    proxies_pt  = f"{LOG_DIR}/aux/{args.dataset}_{args.model}_proxies.pth"
                    proxies_npy = f"{LOG_DIR}/aux/{args.dataset}_{args.model}_proxies.npy"
                    torch.save(P_tensor, proxies_pt)
                    np.save(proxies_npy, P_tensor.numpy())
                    print(f"[SAVE] Proxies saved: {proxies_pt}  ({tuple(P_tensor.shape)})")
                else:
                    print("[WARN] Could not extract proxies from criterion; saved model & criterion only.")

                print(f"[SAVE] Improved R@1: {Recalls[0]*100:.2f}% "
                    f"(prev {prev_best_top1*100:.2f}%). Saved to {save_path}")

                # 저장/개선 상황 출력
                print(f"[SAVE] Improved R@1: {Recalls[0]*100:.2f}% "
                    f"(prev {prev_best_top1*100:.2f}%). Saved to {save_path}")

                with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                    f.write('Best Epoch: {}\n'.format(best_epoch))
                    if args.dataset == 'Inshop':
                        for i, K in enumerate([1,10,20,30,40,50]):
                            f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100))
                    elif args.dataset != 'SOP':
                        for i in range(6):
                            f.write("Best Recall@{}: {:.4f}\n".format(2**i, best_recall[i] * 100))
                    else:
                        for i in range(4):
                            f.write("Best Recall@{}: {:.4f}\n".format(10**i, best_recall[i] * 100))

                # replaced wandb.summary updates with print
                print(f"[SUMMARY] @1_overall={float(best_recalls[0])}, best_epoch_overall={int(best_epoch)}")
    return Ks, best_recalls, best_epoch

def migitage_dmlbg(**kwargs):
    class Args:
        def __init__(self):
            # 기본값
            self.LOG_DIR = "../logs"
            self.dataset = "dep"
            self.sz_embedding = 512
            self.sz_batch = 180
            self.nb_epochs = 60
            self.gpu_id = 0
            self.nb_workers = 4
            self.model = "bn_inception"
            self.loss = "Proxy_Anchor"
            self.optimizer = "adamw"
            self.lr = 1e-4
            self.weight_decay = 1e-4
            self.lr_decay_step = 10
            self.lr_decay_gamma = 0.5
            self.alpha = 32
            self.mrg = 0.1
            self.IPC = None
            self.warm = 1
            self.bn_freeze = 1
            self.l2_norm = 1
            self.remark = ""
            self.disent = True
            self.al = 0.0
            self.be = 0.0
            self.gam = 0.0
            self.de = 0.0
            self.eps = 0.0
            self.ze = 0.0
            self.eta = 0.0
            self.theta = 0.0
            self.iota = 0.0
            self.kap = 0.0
            self.bg_start = 0
            # kwargs override
            for k,v in kwargs.items():
                setattr(self, k, v)
    Ks, best_recalls, best_epoch = dmlbg_main(Args())
    return {"Ks": Ks,  "best_recalls": best_recalls, "best_epoch": best_epoch}


if __name__ == "__main__":
    args = get_args()
    Ks, best_recalls, best_epoch = dmlbg_main(args)
    print(f"Best Epoch: {best_epoch}, Best Recalls: {best_recalls}")