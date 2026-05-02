import argparse
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Owlv2Model
import os.path as osp
from models.others.CNN import resnet12
from models.TeSMo_KAN import TeSMo_KAN
from models.distill import DistillKL
from dataset.datasets import DatasetLoader
from dataset.samplers import CategoriesSampler
from dataset.textual_desc import get_class_labels_from_split
from utils import seed_torch, set_gpu, ensure_path, Averager, count_acc, euclidean_metric, Timer, compute_confidence_interval

Omodel = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble").cuda()
Oprocessor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
    
def get_dataset(args):
    if args.dataset == 'cub':
        n_cls = 200
        print("=> CUB_200_2011...")
    elif args.dataset == 'dog':
        n_cls = 120
        print("=> Stanford_Dogs...")
    elif args.dataset == 'car':
        n_cls = 196
        print("=> Stanford_Cars...")
    else:
        print("Invalid dataset:", args.dataset)
        exit()
        
    trainset = DatasetLoader(dataset_name=args.dataset, phase='train', size=args.image_size)
    valset = DatasetLoader(dataset_name=args.dataset, phase='valid', size=args.image_size)
    testset = DatasetLoader(dataset_name=args.dataset, phase='test', size=args.image_size)
    
    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                        args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                num_workers=0, pin_memory=True)

    val_sampler = CategoriesSampler(valset.label, args.valid_batch,
                                    args.train_way, args.shot + args.train_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=True)
    
    test_sampler = CategoriesSampler(testset.label, args.test_batch,
                                    args.test_way, args.shot + args.test_query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                            num_workers=0, pin_memory=True)

    train_labels, valid_labels, test_labels = get_class_labels_from_split(dataset_name=args.dataset)
    
    return train_loader, val_loader, test_loader, n_cls, train_labels, valid_labels, test_labels

def main(args):
    ensure_path(args.save_path)

    train_loader, val_loader, test_loader, n_cls, train_labels, valid_labels, test_labels = get_dataset(args)
   
    teacher = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls).cuda()
    checkpoint_file = os.path.join(args.stage1_path, 'max-test-acc.pth')
    teacher.load_state_dict(torch.load(checkpoint_file))

    model = TeSMo_KAN(num_classes=n_cls).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1, verbose=True)
    criterion_kd = DistillKL(args.temperature).cuda()
    
    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(points, path, name='checkpoint'):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(points, os.path.join(path, '{}.pth.tar'.format(name)))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['test_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['test_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['maxtestacc'] = 0.0
    trlog['max_epoch'] = 0
    trlog['maxtestepoch'] = 0

    timer = Timer()
    best_epoch = 0
    start_epoch = 1
    cmi = [0.0, 0.0]
    cmi2 = [0.0, 0.0]
      
    # check resume point
    checkpoint_file = os.path.join(args.checkpoint_path, 'checkpoint.pth.tar')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        trlog = checkpoint['trlog']
        start_epoch = checkpoint['start_epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        trlog['maxtestacc'] = checkpoint['best_test_acc']
        trlog['maxtestepoch'] = checkpoint['best_test_epoch']
        trlog['max_acc'] = checkpoint['best_acc']
        trlog['max_epoch'] = checkpoint['best_epoch']
        print("=> Resume from epoch {} ...".format(start_epoch))
    
    
    for epoch in range(start_epoch, args.max_epoch + 1):

        tl, ta = train(args, model, train_loader, optimizer, teacher, criterion_kd, n_cls, train_labels)
        lr_scheduler.step()
        vl, va, aa, bb = validate(args, model, val_loader, n_cls, valid_labels)
        # Additional validation on test dataset
        test_loss, test_acc, test_acc_mean, test_acc_ci = validate(args, model, test_loader, n_cls, test_labels)

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
            trlog['max_epoch'] = epoch
            best_epoch = epoch
            cmi[0] = aa
            cmi[1] = bb
            
            # save best model
            save_checkpoint({
                'best_epoch': epoch,
                'model': model.state_dict()
            }, args.save_path, name='max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)
        trlog['test_loss'].append(test_loss)
        trlog['test_acc'].append(test_acc)
        
        # Update max test accuracy and epoch if necessary
        if test_acc > trlog['maxtestacc']:
            trlog['maxtestacc'] = test_acc
            trlog['maxtestepoch'] = epoch
            save_model('max-test-acc')
            cmi2[0] = test_acc_mean
            cmi2[1] = test_acc_ci
            
            # save best model
            save_checkpoint({
                'best_test_acc': test_acc,
                'best_test_epoch': epoch,
                'model': model.state_dict()
            }, args.save_path, name='max-test-acc')

        torch.save(trlog, osp.join(args.save_path, 'trlog'))
        
        # checkpoint saving
        save_checkpoint({
            'start_epoch': epoch,
            'best_test_acc': trlog['maxtestacc'],
            'best_test_epoch': trlog['maxtestepoch'],
            'best_acc': trlog['max_acc'],
            'best_epoch': trlog['max_epoch'],
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'trlog': trlog
        }, args.save_path)
        
        save_model('epoch-last')
        ot, ots = timer.measure()
        tt, _ = timer.measure(epoch / args.max_epoch)
        
        print('Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f} - max acc={:.4f} - ETA:{}/{}'.format(
            epoch, args.max_epoch, tl, ta, vl, va, trlog['max_acc'], ots, timer.tts(tt-ot)))
        print("Best Epoch is {} with acc={:.2f}±{:.2f}%...".format(best_epoch, cmi[0], cmi[1]))
        
        print('\nTest loss={:.4f} - acc={:.4f} - acc={:.2f}±{:.2f}%'.format(test_loss, test_acc, test_acc_mean, test_acc_ci))
        print("Best Test Accuracy: {:.2f}±{:.2f}%, achieved at epoch {}".format(cmi2[0], cmi2[1], trlog['maxtestepoch']))
        
        print("------------------------------------------------------\n")

def train(args, model, train_loader, optimizer, teacher, kd_loss, n_cls, class_labels):
    model.train()
    Omodel.eval()
    teacher.eval()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]
        
        label = torch.arange(args.train_way).repeat(args.train_query)
        label = label.type(torch.cuda.LongTensor)
        
        with torch.no_grad():
            tproto = teacher(data_shot)
            tproto = tproto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            tlogits = euclidean_metric(teacher(data_query), tproto)
        
        proto_image_features = model(data_shot)  # (30, 1600)
        proto_image_features = proto_image_features.reshape(args.shot, args.train_way, -1).mean(dim=0)
        query_image_features = model(data_query)

        inputs = Oprocessor(text=class_labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = Omodel.get_text_features(input_ids=inputs['input_ids'].cuda(),
                                                       attention_mask=inputs['attention_mask'].cuda())
            
        text_features = text_features[:, :n_cls] 

        proto_image_features = proto_image_features / proto_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        proto = ((1-args.delta) * proto_image_features) + (args.delta * text_features[:args.test_way, :])
        query = query_image_features
        
        logits = euclidean_metric(query, proto)
        acc = count_acc(logits, label)
        
        # knowledge distill loss
        kdloss = kd_loss(logits, tlogits)
        loss = ((1.0 - args.gamma) * F.cross_entropy(logits, label)) + (args.gamma * kdloss)
        
        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        proto = None; query = None; logits = None; loss = None
        tproto = None; tlogits = None; 
    return tl.item(), ta.item()

def validate(args, model, val_loader, n_cls, class_labels):
    model.eval()
    Omodel.eval()

    vl = Averager()
    va = Averager()
    acc_list = []

    with torch.no_grad():
        inputs = Oprocessor(text=class_labels, return_tensors="pt", padding=True)

        text_features = Omodel.get_text_features(input_ids=inputs['input_ids'].cuda(),
                                                       attention_mask=inputs['attention_mask'].cuda())
                
        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto_image_features  = model(data_shot)  # (30, 1600)
            proto_image_features  = proto_image_features .reshape(args.shot, args.test_way, -1).mean(dim=0)
            query_image_features  = model(data_query)
            
            text_features = Omodel.get_text_features(input_ids=inputs['input_ids'].cuda(),
                                                       attention_mask=inputs['attention_mask'].cuda())
            text_features = text_features[:, :n_cls]

            # Normalize features
            proto_image_features = proto_image_features / proto_image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Combine text and image features
            proto = ((1-args.delta) * proto_image_features) + (args.delta * text_features[:args.test_way, :])
            query = query_image_features  
            
            label = torch.arange(args.test_way).repeat(args.test_query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(query, proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            acc_list.append(acc*100)

            proto = None; query = None; logits = None; loss = None
    
    a,b = compute_confidence_interval(acc_list)
    return vl.item(), va.item(), a, b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--checkpoint-path', default='')
    parser.add_argument('--stage1-path', default='')
    parser.add_argument('--save-path', default='')
    parser.add_argument('--gpu', default='0')
    # few-shot setting
    parser.add_argument('--shot', type=int, default=1) 
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    # Multimodal Prototype
    parser.add_argument('--delta', type=float, default=0.1)
    # Knowledge-Distillation
    parser.add_argument('--temperature', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.7)
    # dataset
    parser.add_argument('--dataset', type=str, default='cub', choices=['cub','dog','car'])
    parser.add_argument('--image-size', type=int, default=84)
    # network
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.001)
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--train-batch', type=int, default=10000) #every epoch consist 1000 episode, total 30 epoch, so 30000 episode
    parser.add_argument('--valid-batch', type=int, default=1000)
    parser.add_argument('--test-batch', type=int, default=1000)
    args, _ = parser.parse_known_args()
    
    # Create the directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    start_time = datetime.datetime.now()

    # fix seed
    seed_torch(1)
    set_gpu(args.gpu)
    main(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)