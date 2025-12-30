import numpy as np
import torch
import logging
from . import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math

import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

def evaluation(X, Y, Kset, args):

    def get_recallK(Y_query, YNN, Kset):
        recallK = np.zeros(len(Kset))
        num = Y_query.shape[0]
        for i in range(0, len(Kset)):
            pos = 0.
            for j in range(0, num):
                if Y_query[j] in YNN[j, :Kset[i]]:
                    pos += 1.
            recallK[i] = pos/num
        return recallK

    def get_Rstat(Y_query, YNN, test_class_dict):
        '''
        test_class_dict:
            key = class_idx, value = the number of images
        '''
        RP_list = []
        MAP_list = []

        for gt, knn in zip(Y_query, YNN):
            n_imgs = test_class_dict[gt] - 1 # - 1 for query.
            selected_knn = knn[:n_imgs]
            correct_array = (selected_knn == gt).astype('float32')
            
            RP = np.mean(correct_array)
            
            MAP = 0.0
            sum_correct = 0.0
            for idx, correct in enumerate(correct_array):
                if correct == 1.0:
                    sum_correct += 1.0
                    MAP += sum_correct / (idx + 1.0)
            MAP = MAP / n_imgs
            
            RP_list.append(RP)
            MAP_list.append(MAP)
        
        return np.mean(RP_list), np.mean(MAP_list)

    def evaluation_faiss(X, Y, Kset, args):
        if args.data_name.lower() != 'inshop':
            kmax = np.max(Kset + [args.max_r]) # search K
        else:
            kmax = np.max(Kset)
        
        test_class_dict = args.test_class_dict

        # compute NMI
        if args.do_nmi:
            classN = np.max(Y)+1
            kmeans = KMeans(n_clusters=classN).fit(X)
            nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
        else:
            nmi = 0.0

        if args.data_name.lower() != 'inshop':
            offset = 1
            X_query = X
            X_gallery = X
            Y_query = Y
            Y_gallery = Y

        else: # inshop
            offset = 0
            len_gallery = len(args.gallery_labels)
            X_gallery = X[:len_gallery, :]
            X_query = X[len_gallery:, :]
            Y_query = args.query_labels
            Y_gallery = args.gallery_labels

        nq, d = X_query.shape
        ng, d = X_gallery.shape
        I = np.empty([nq, kmax + offset], dtype='int64')
        D = np.empty([nq, kmax + offset], dtype='float32')
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        faiss.bruteForceKnn(res, faiss.METRIC_INNER_PRODUCT,
                            faiss.swig_ptr(X_gallery), True, ng,
                            faiss.swig_ptr(X_query), True, nq,
                            d, int(kmax + offset), faiss.swig_ptr(D), faiss.swig_ptr(I))

        indices = I[:,offset:]

        YNN = Y_gallery[indices]
        
        recallK = get_recallK(Y_query, YNN, Kset)
        
        if args.data_name.lower() != 'inshop':
            RP, MAP = get_Rstat(Y_query, YNN, test_class_dict)
        else: # inshop
            RP = 0
            MAP = 0

        return nmi, recallK, RP, MAP

    return evaluation_faiss(X, Y, Kset, args)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())[0][:,:512]

                for j in J:
                    A[i].append(j.cpu())
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall


def evaluate_cos_baseline(embeddings, labels, Ks=[1,2,4,8,16,32]):
    """
    embeddings: (N, D) tensor
    labels: (N,) tensor
    Ks: list of K values for Recall@K
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    sim_matrix = embeddings @ embeddings.t()  # cosine similarity
    N = labels.size(0)
    recalls = []

    for K in Ks:
        correct = 0
        for i in range(N):
            sim_scores = sim_matrix[i]
            sim_scores[i] = -1  # exclude self
            topk = torch.topk(sim_scores, K).indices
            if labels[i] in labels[topk]:
                correct += 1
        recalls.append(correct / N)
    return recalls


def evaluate_cos_Inshop_baseline(query_embeddings, query_labels, gallery_loader, Ks=[1,10,20,30,40,50]):
    """
    query_embeddings: (Nq, D)
    query_labels: (Nq,)
    gallery_loader: DataLoader for gallery set
    """
    # 1. Collect gallery embeddings and labels
    gallery_embeddings, gallery_labels = [], []
    for x, y in gallery_loader:
        x = x.to(query_embeddings.device)
        emb = x if isinstance(x, torch.Tensor) else x[0]
        gallery_embeddings.append(emb.cpu())
        gallery_labels.append(y)
    gallery_embeddings = torch.cat(gallery_embeddings)
    gallery_labels = torch.cat(gallery_labels)

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=1)

    sim_matrix = query_embeddings @ gallery_embeddings.t()  # (Nq, Ng)
    Nq = query_labels.size(0)
    recalls = []

    for K in Ks:
        correct = 0
        for i in range(Nq):
            topk = torch.topk(sim_matrix[i], K).indices
            if query_labels[i] in gallery_labels[topk]:
                correct += 1
        recalls.append(correct / Nq)
    return recalls
