import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F
from util.get_dataset import get_dataset
from util.CNNmodel import *
from util.augmentation import augmentations
from torchsummary import summary
from util.con_losses import SupConLoss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2023)

def get_param_value(model_size: str) -> int:
    """Returns the parameter value based on the input size: S, M, or L."""
    model_size_mapping = {'S': 8, 'M': 16, 'L': 32}
    if model_size in model_size_mapping:
        return model_size_mapping[model_size]
    else:
        raise ValueError(f"Invalid model_size: {model_size}. Use 'S', 'M', or 'L'.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Single-source Domain Generalization")
    parser.add_argument("--dataset_name", type=str, default="ORACLE", choices=["ORACLE", "WiSig"])   
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"])
    parser.add_argument("--model_size", type=str, default="S")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--main_aug_depth", type=int, nargs='+', default=[2])
    parser.add_argument("--aux_aug_depth", type=int, nargs='+', default=[1])
    parser.add_argument("--lambda_con", type=float, nargs='+', default=[1.0, 100.0])
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    return parser.parse_args()

def aug(iq_data, preprocess, main_aug_depth, aux_aug_depth):
    aug_list = augmentations
    iq_data_aug_list = []
    for i in range(len(aux_aug_depth)):
        iq_data_aug = iq_data
        if aux_aug_depth[i] != 0:
            sampled_ops = np.random.choice(aug_list, aux_aug_depth[i])
            for op in sampled_ops:
                iq_data_aug = op(iq_data_aug)
        iq_data_aug = np.squeeze(preprocess(iq_data_aug.astype(np.float32)))
        iq_data_aug_list.append(iq_data_aug)

    #main_aug
    iq_data_aug = iq_data
    if main_aug_depth[0] != 0:
        sampled_ops = np.random.choice(aug_list, main_aug_depth[0])
        for op in sampled_ops:
            iq_data_aug = op(iq_data_aug)
    iq_data_aug = np.squeeze(preprocess(iq_data_aug.astype(np.float32)))
    iq_data_aug_list.append(iq_data_aug)

    return iq_data_aug_list

class AugDataset(torch.utils.data.Dataset):
  def __init__(self, x_train, y_train, preprocess, main_aug_depth, aux_aug_depth):
    self.dataset = []
    for i in range(np.shape(x_train)[0]):
        self.dataset.append((np.squeeze(x_train[i,:,:]), y_train[i]))#
    self.preprocess = preprocess
    self.main_aug_depth = main_aug_depth
    self.aux_aug_depth = aux_aug_depth

  def __getitem__(self, idx):
    x, y = self.dataset[idx]
    return aug(x, self.preprocess, self.main_aug_depth, self.aux_aug_depth), y

  def __len__(self):
    return len(self.dataset)

def train(model, loss, train_dataloader, optimizer, epoch, conf):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target = data_nn
        target = target.long()
        domain_target = []
        target_all = []
        num_data= len(conf.main_aug_depth) + len(conf.aux_aug_depth)
        for i in range(num_data):
            domain_target.append(i*torch.ones(data[0].size(0)).long())   
            target_all.append(target)   

        if torch.cuda.is_available():
            data_all = torch.cat(data, 0).cuda()
            target = target.cuda()
            target_all = torch.cat(target_all, 0).cuda()
            domain_target= torch.cat(domain_target, 0).cuda()

        optimizer.zero_grad()
        embedding, output = model(data_all)
        prob = F.log_softmax(output, dim=1)
        porb_list = torch.split(prob, data[0].size(0))        
        cls_loss = loss[0](porb_list[num_data - 1], target)       
        con_loss = loss[1](embedding.unsqueeze(1), target_all, adv=False)
        adv_con_loss = loss[1](embedding.unsqueeze(1), domain_target, adv=True)
        result_loss = cls_loss + conf.lambda_con[0]*con_loss + conf.lambda_con[1]*adv_con_loss
        result_loss.backward()
        optimizer.step()
        all_loss += result_loss.item()*data[0].size(0)
        pred = porb_list[num_data -  1].argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )

def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss[0](output, target).item()*data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return test_loss

def test(model, test_dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            embedding, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(correct / len(test_dataloader.dataset))


def train_and_evaluate(model, loss, train_loader, val_loader, optimizer, epochs, save_path, conf):
    """Train and evaluate the model, saving the best model."""
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train(model, loss, train_loader, optimizer, epoch, conf)
        val_loss = evaluate(model, loss, val_loader, epoch)
        if val_loss < best_loss:
            print(f"Saving model at epoch {epoch} with loss {val_loss:.4f}")
            best_loss = val_loss
            torch.save(model, save_path)

def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda

    num_classes = 16 if conf.dataset_name == "ORACLE" else 6
    input_shape = (2, 6000) if conf.dataset_name == "ORACLE" else (2, 256)

    save_path = (
        f"weight/Dataset={conf.dataset_name}_"
        f"Model={conf.model_size}_"
        f"main_aug_depth={','.join(map(str, conf.main_aug_depth))}_"
        f"aux_aug_depth={','.join(map(str, conf.aux_aug_depth))}_"
        f"lambda={','.join(map(str, conf.lambda_con))}.pth"
    )
    
    dataset = get_dataset(conf.dataset_name)
    train_loader = DataLoader(AugDataset(dataset['train'][0], dataset['train'][1], transforms.ToTensor(), conf.main_aug_depth, conf.aux_aug_depth), 
                              batch_size=conf.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(dataset['val'][0]), torch.Tensor(dataset['val'][1])), 
                            batch_size=conf.test_batch_size, shuffle=True)

    model = MACNN(in_channels=2, channels=get_param_value(conf.model_size), num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    cls_loss = nn.NLLLoss()
    con_loss = SupConLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        cls_loss = cls_loss.cuda()
        con_loss = con_loss.cuda()
        summary(model, input_shape)

    loss = cls_loss, con_loss

    if conf.mode in ["train", "train_test"]:
        print("Starting training...")
        train_and_evaluate(model, loss, train_loader, val_loader, optimizer, conf.epochs, save_path, conf)

    if conf.mode in ["test", "train_test"]:
        print("Starting testing on source domain...")
        test_loader = DataLoader(TensorDataset(torch.Tensor(dataset['test_s'][0]), torch.Tensor(dataset['test_s'][1])), 
                                 batch_size=conf.test_batch_size, shuffle=False)
        model = torch.load(save_path)
        test(model, test_loader)

        print("Starting testing on target domain...")
        for i, (x_test, y_test) in enumerate(dataset['test_t']):
            test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), 
                             batch_size=conf.test_batch_size, shuffle=False)
            model = torch.load(save_path)
            test(model, test_loader)

if __name__ == "__main__":
    main()