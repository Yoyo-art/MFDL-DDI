import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import torch.nn as nn
from dataset import load_ddi_dataset
from train_logger import TrainLogger
from utils import *
from metrics import *
from model import MFDL_DDI
from data_preprocessing import CustomData


def val(model, criterion, dataloader, device, epoch, sa_fe):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in tqdm(dataloader,desc='val_epoch_{}'.format(epoch),leave=True):
        head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, head_smi, tail_smi, head_fp, tail_fp = [d.to(device) for d in data]
        batch_h_e = head_pairs_dgl.edata['feat']
        batch_t_e = tail_pairs_dgl.edata['feat']

        with torch.no_grad():
            # pred = model((head_pairs, tail_pairs, rel))
            pred = model(head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_smi, tail_smi, head_fp, tail_fp, sa_fe)
            loss = criterion(pred, label)

            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap


def test(model, criterion, dataloader, device, sa_fe):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []
    smi_list = []
    fp_list = []
    graph_list = []
    feature_list = []
    pair_list = []
    class_list = []
    rel_list = []

    for data in tqdm(dataloader,leave=True):
        head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, head_smi, tail_smi, head_fp, tail_fp = [d.to(device) for d in data]
        batch_h_e = head_pairs_dgl.edata['feat']
        batch_t_e = tail_pairs_dgl.edata['feat']

        with torch.no_grad():
            # pred = model((head_pairs, tail_pairs, rel))
            pred, pair, h, t = model(head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_smi, tail_smi, head_fp, tail_fp, sa_fe)
            loss = criterion(pred, label)

            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            # features_h = h.detach().cpu().numpy()
            # split_arrays = np.split(features_h, 3, axis=1)  # 在列方向 (axis=1) 上分割
            # features = np.vstack(split_arrays)
            # feature_list.append(features)
            # class_list.append(np.array(['SMILES'] * pred.shape[0] + ['2D Graph'] * pred.shape[0] + ['Fingerprint'] * pred.shape[0]))
            # features_t = torch.concat((t_smi_emb, t_fp_emb, t_graph_emb), -1)
            # feature_list.append(features_t.detach().cpu().numpy())
            # class_list.append(np.array(['SMILES'] * pred.shape[0] + ['2D Graph'] * pred.shape[0] + ['Fingerprint'] * pred.shape[0]))
            # pair_list.append(pair.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            rel_list.append(rel.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    relation = np.concatenate(rel_list, axis=0)
    np.save("yy.npy", pred_probs)
    np.save("lable.npy", label)
    np.save("rel.npy", relation)
    # features = np.concatenate(feature_list, axis=0)
    # np.save("features1.npy", features)
    # classes = np.concatenate(class_list, axis=0)
    # np.save("class1.npy", classes)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    # model.train()
    print("test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
    epoch_loss, acc, auroc, f1_score, precision, recall, ap))


    return epoch_loss, acc, auroc, f1_score, precision, recall, ap

def main():
    parser = argparse.ArgumentParser()
    # Add argument
    parser.add_argument('--n_iter', type=int, default=10, help='number of MPNN')
    parser.add_argument('--L', type=int, default=3, help='number of Graph Transformer')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight of decay')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()
    params = dict(
        model='MFDL_DDI',
        data_root='data/',
        save_dir='../save/transductive',
        dataset='drugbank',
        epochs=args.epochs,
        L=args.L,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay
    )
    logger = TrainLogger(params)
    logger.info(__file__)
    save_model = params.get('save_model')
    batch_size = params.get('batch_size')
    data_root = params.get('data_root')
    fold = params.get('fold')
    epochs = params.get('epochs')
    n_iter = params.get('n_iter')
    L = params.get('L')
    lr = params.get('lr')
    weight_decay = params.get('weight_decay')
    train_loader, val_loader, test_loader = load_ddi_dataset(root=data_root, batch_size=batch_size, fold=fold)
    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim = data[0].edge_attr.size(-1)
    print(node_dim)
    print(edge_dim)
    device = torch.device('cuda:0')
    model = MFDL_DDI(node_dim, edge_dim, n_iter=n_iter).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    running_loss = AverageMeter()
    running_acc = AverageMeter()
    max_acc = 0
    model.train()
    for epoch in range(epochs):
        label_list = []
        pair_list = []
        rel_list = []
        for data in tqdm(train_loader,desc='train_loader_epoch_{}'.format(epoch),leave=True):
            head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, head_smi, tail_smi, head_fp, tail_fp = [d.to(device) for d in data]
            batch_h_e = head_pairs_dgl.edata['feat']
            batch_t_e = tail_pairs_dgl.edata['feat']
            pred, pair, h, t = model(head_pairs, tail_pairs, rel, label, head_pairs_dgl, tail_pairs_dgl, batch_h_e, batch_t_e, head_smi, tail_smi, head_fp, tail_fp, sa_fe=True)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0))
            # pair_list.append(pair.detach().cpu().numpy())
            # rel_list.append(rel.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

        # feature_save_dir = f"epoch_{epoch}_features.npy"
        # label_save_dir = f"epoch_{epoch}_label.npy"
        # np.save(feature_save_dir, pair_list)
        # np.save(label_save_dir, rel_list)
        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap = val(model, criterion, val_loader, device, epoch, sa_fe=False)
        test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap = val(model, criterion, test_loader, device, epoch, sa_fe=False)
        if test_acc > max_acc:
            max_acc = test_acc
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (
            epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            save_model_dict(model, logger.get_model_dir(), msg)

            msg = "epoch-%d, test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
            epoch, test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap)
            logger.info(msg)


        msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (
        epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
        # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap)
        logger.info(msg)

        scheduler.step()

        if save_model:
            print('save model')
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            save_model_dict(model, logger.get_model_dir(), msg)
    # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap = val(model, criterion, test_loader, device, epoch, sa_fe=False)
    # msg = "test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (
    # test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap)
    # logger.info(msg)
    test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap = test(model, criterion,
                                                                                               test_loader, device, sa_fe=True)
    msg = "test_loss-%.4f, test_acc-%.4f, test_auroc-%.4f, test_f1_score-%.4f, test_prec-%.4f, test_rec-%.4f, test_ap-%.4f" % (test_loss, test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_ap)
    logger.info(msg)



if __name__ == "__main__":
    # main()
    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/', batch_size=512, fold=0)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda:0')
    test_model = MFDL_DDI(70, 6, n_iter=10).cuda()
    test_model.load_state_dict(torch.load(
        'data/epoch-117, train_loss-0.0110, train_acc-0.9991, val_loss-0.1267, val_acc-0.9657.pt'))
    # print(test_model.finger_encoding.fc2.weight)  # 打印权重
    #
    # print("fc2 偏置：")
    # print(test_model.finger_encoding.fc2.bias)
    test(test_model, criterion, test_loader, device, True)
