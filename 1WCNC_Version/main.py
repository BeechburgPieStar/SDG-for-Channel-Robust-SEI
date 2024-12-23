import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F
from util.get_dataset import TrainDataset, TestDataset
from util.CNNmodel import MACNN
from util.augmentation import augmentations
from torchsummary import summary

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
    parser = argparse.ArgumentParser(description="Random overlay augmentation")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"],
                        help="Choose mode: 'train', 'test', or 'train_test'.")
    parser.add_argument("--model_size", type=str, default="S", help="MSACN-S/M/L")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sd_time_ft", type=int, nargs='+', default=[1, 2], help="Source domain, [run1, ft2]")
    parser.add_argument("--td_time_ft", type=int, nargs='+', default=[2, 2], help="Target domain, [run2, ft2]")
    parser.add_argument("--aug_depth", type=int, default=5, help="The depth of (I)")
    parser.add_argument("--wd", type=float, default=0, help="Weight decay")
    parser.add_argument("--cuda", type=str, default="0", help="GPU for training")
    return parser.parse_args()

def aug(iq_data, aug_depth, preprocess):
    """Perform data augmentation."""
    iq_data_aug = iq_data
    sampled_ops = np.random.choice(augmentations, aug_depth)
    for op in sampled_ops:
        iq_data_aug = op(iq_data_aug)
    return np.squeeze(preprocess(iq_data_aug.astype(np.float32)))

class AugDataset(torch.utils.data.Dataset):
    """Custom dataset with augmentation support."""
    def __init__(self, x_data, y_data, aug_depth, preprocess):
        self.dataset = [(np.squeeze(x_data[i]), y_data[i]) for i in range(len(x_data))]
        self.aug_depth = aug_depth
        self.preprocess = preprocess

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return aug(x, self.aug_depth, self.preprocess), y

    def __len__(self):
        return len(self.dataset)

def train(model, loss_fn, dataloader, optimizer, epoch):
    """Train the model."""
    model.train()
    total_loss, correct = 0, 0
    for data, target in dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        _, output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        total_loss / len(dataloader.dataset),
        correct,
        len(dataloader.dataset),
        100.0 * correct / len(dataloader.dataset))
    )

def evaluate(model, loss_fn, dataloader, epoch):
    """Evaluate the model."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            _, output = model(data)
            output = F.log_softmax(output, dim=1)
            total_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            total_loss/len(dataloader.dataset),
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )
    return total_loss/len(dataloader.dataset)

def test(model, dataloader):
    """Test the model."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            _, output = model(data)
            output = F.log_softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(dataloader.dataset)
    print(f"Test: Accuracy {accuracy:.2f}%")


def train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, epochs, save_path):
    """Train and evaluate the model, saving the best model."""
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train(model, loss_fn, train_loader, optimizer, epoch)
        val_loss = evaluate(model, loss_fn, val_loader, epoch)
        if val_loss < best_loss:
            print(f"Saving model at epoch {epoch} with loss {val_loss:.4f}")
            best_loss = val_loss
            torch.save(model, save_path)

def main():
    conf = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.cuda

    save_path = f"weight/{conf.model_size}_ROA_sd={conf.sd_time_ft}_depth={conf.aug_depth}.pth"

    # Load datasets
    x_train, x_val, y_train, y_val = TrainDataset(conf.sd_time_ft)
    train_loader = DataLoader(AugDataset(x_train, y_train, conf.aug_depth, transforms.ToTensor()), 
                              batch_size=conf.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val)), 
                            batch_size=conf.test_batch_size, shuffle=False)

    model = MACNN(in_channels=2, channels=get_param_value(conf.model_size), num_classes=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.wd)
    loss_fn = torch.nn.NLLLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        summary(model, (2, 6000))

    if conf.mode in ["train", "train_test"]:
        print("Starting training...")
        train_and_evaluate(model, loss_fn, train_loader, val_loader, optimizer, conf.epochs, save_path)

    if conf.mode in ["test", "train_test"]:
        print("Starting testing on source domain...")
        x_test, y_test = TestDataset(conf.sd_time_ft)
        test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), 
                                 batch_size=conf.test_batch_size, shuffle=False)
        model = torch.load(save_path)
        test(model, test_loader)
        print("Starting testing on target domain...")
        x_test, y_test = TestDataset(conf.td_time_ft)
        test_loader = DataLoader(TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), 
                                 batch_size=conf.test_batch_size, shuffle=False)
        model = torch.load(save_path)
        test(model, test_loader)

if __name__ == "__main__":
    main()