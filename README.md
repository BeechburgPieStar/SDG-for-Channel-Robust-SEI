# Towards Channel-Robust-SEI

## (WCNC2025 Accept!) More is Better: Channel-Robust Radio Frequency Fingerprinting with Random Overlay Augmentation

### File directory description

```
filetree 
├── /dataset
├── util
│  ├── augmentation.py
│  ├── CNNmodel.py
|  └── get_dataset.py
├── weight
└── main.py
```

### How to run?

```
python main.py --mode train_test --model_size S --aug_depth 4
```

## (TWC2025 Accept!) Avoiding Shortcuts: Enhancing Channel-Robust Specific Emitter Identification via Single-Source Domain Generalization

```
filetree 
├── Dataset_ORALCE
├── Dataset_WiSig
├── util
│  ├── augmentation.py
│  ├── con_losses.py
│  ├── CNNmodel.py
|  └── get_dataset.py
├── weight
└── main.py
```

### Dataset

```
百度网盘： https://pan.baidu.com/s/1ilpykvcLWpfLjKHd03S4xA?pwd=7wd7

Google：https://drive.google.com/drive/folders/1vLa3p5uX45aJE5IziRCbMIOR5D3YgvkC?usp=sharing
```

### How to run?

```
ORACEL Dataset

python main.py --dataset_name ORACLE --mode train_test --model_size S --epochs 1000 --main_aug_depth 3 --aux_aug_depth 1 2 --lambda_con 0.01 0.1 --cuda 0
python main.py --dataset_name ORACLE --mode train_test --model_size M --epochs 1000 --main_aug_depth 3 --aux_aug_depth 1 2 --lambda_con 1.0 1.0 --cuda 0
python main.py --dataset_name ORACLE --mode train_test --model_size L --epochs 1000 --main_aug_depth 4 --aux_aug_depth 1 2 3 --lambda_con 1.0 1.0 --cuda 0

WiSig Dataset

python main.py --dataset_name WiSig --mode train_test --model_size S --epochs 200 --main_aug_depth 2 --aux_aug_depth 1 --lambda_con 1.0 100.0 --cuda 0
```

### Training and testing logs

```
/2TWC_Version/log
```

### License / 许可证

```
本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途。

This project is distributed under a custom non-commercial license. Any form of commercial use is prohibited.
```


