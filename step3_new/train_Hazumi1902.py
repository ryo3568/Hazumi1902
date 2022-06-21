import glob
import warnings
import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from model import LSTMModel, MaskedNLLLoss, FNNModel
from dataloader import IEMOCAPDataset, HazumiDataset

warnings.simplefilter('ignore')

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset(path=path)
    testset = IEMOCAPDataset(path=path, train=False)
    
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def get_Hazumi_loaders(testfile, batch_size=32, valid=0.1, num_workers=0, pin_memory=False, rate=1.0):
    trainset = HazumiDataset(testfile, rate=rate)
    testset = HazumiDataset(testfile,train=False)
    
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, umask, label =\
                [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
        log_prob, alpha, alpha_f, alpha_b = model(torch.cat((textf, acouf, visuf), dim=-1), umask)  
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1) 
        #CrossEntropyに変更
        loss = loss_function(lp_, labels_, umask)
        # loss = loss_function(lp_, labels_)

        pred_ = torch.argmax(lp_, 1) 
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--model', default='LSTM', help='using model')
    parser.add_argument('--rate', type=float, default=1.0, help='rate of using train data')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    cuda       = args.cuda
    n_epochs   = args.epochs
    
    n_classes  = 3
    D_m = 1417
    D_e = 100
    D_h = 100

    FILE_PATH = '../data/dumpfiles/*.csv'
    files = glob.glob(FILE_PATH)

    all_score = []

    for testfile in files:
        print('testfile : ', testfile)

        if args.model == 'LSTM':
            model = LSTMModel(D_m, D_e, D_h,
                      n_classes=n_classes,
                      dropout=args.dropout,
                      attention=args.attention)
        elif args.model == 'FNN':
            model = FNNModel(D_m, D_e, D_h, 
                        n_classes=n_classes,
                        dropout=args.dropout,
                        attention=args.attention)

        if cuda:
            model.cuda()
        
        loss_weights = torch.FloatTensor([3.45142857, 0.77977978, 0.66981943])
        # CrossEntropyに変更
        if args.class_weight:
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            loss_function = MaskedNLLLoss()
        # loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

        train_loader, valid_loader, test_loader = get_Hazumi_loaders(testfile,
                                                                  batch_size=batch_size,
                                                                  valid=0.1,
                                                                  rate=args.rate)


        best_loss, best_label, best_pred, best_mask = None, None, None, None

        for e in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function,
                                                train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, val_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
            test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)

            if best_loss == None or best_loss > test_loss:
                best_loss, best_label, best_pred, best_mask, best_attn =\
                        test_loss, test_label, test_pred, test_mask, attentions

            if args.tensorboard:
                writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
                writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
            # print('epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
            #         format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
            #                 test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

        if args.tensorboard:
            writer.close()

        print('Test performance..')
        print('Loss {} F1-score {}'.format(best_loss,
                                        round(f1_score(best_label, best_pred, sample_weight=best_mask, average='weighted')*100, 2)))
        print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
        print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
        print('accuracy(weight) : ',accuracy_score(best_label, best_pred, sample_weight=best_mask))
        all_score.append(accuracy_score(best_label, best_pred))
    

    print(all_score)
    print(np.array(all_score).mean())