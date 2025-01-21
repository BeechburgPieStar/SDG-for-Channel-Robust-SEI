# Robust-SEI

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
### How to run?

```
ORACEL Dataset

python main.py --dataset_name ORACLE --mode train_test --model_size S --epochs 1000 --main_aug_depth 3 --aux_aug_depth 1 2 --lambda_con 0.01 0.1 --cuda 0
python main.py --dataset_name ORACLE --mode train_test --model_size M --epochs 1000 --main_aug_depth 3 --aux_aug_depth 1 2 --lambda_con 1.0 1.0 --cuda 0
python main.py --dataset_name ORACLE --mode train_test --model_size L --epochs 1000 --main_aug_depth 4 --aux_aug_depth 1 2 3 --lambda_con 1.0 1.0 --cuda 0

WiSig Dataset

python main.py --dataset_name WiSig --mode train_test --model_size S --epochs 200 --main_aug_depth 2 --aux_aug_depth 1 --lambda_con 1.0 100.0 --cuda 0
```


