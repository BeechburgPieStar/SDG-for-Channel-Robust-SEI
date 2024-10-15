import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F
from util.get_dataset import TrainDataset, TestDataset
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

# Key function
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