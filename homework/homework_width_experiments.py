import torch
import time
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from itertools import product
from sklearn.model_selection import ParameterGrid
from datasets import get_mnist_loaders, get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters
from torch.utils.data import random_split, DataLoader


def create_width_architectures():
    return {
        "narrow_decreasing": [64, 32, 16],
        "narrow_constant": [64, 64, 64], 
        "narrow_increasing": [16, 32, 64],
        "medium_decreasing": [256, 128, 64],
        "medium_constant": [256, 256, 256],
        "medium_increasing": [64, 128, 256],
        "wide_decreasing": [1024, 512, 256],
        "wide_constant": [1024, 1024, 1024],
        "wide_increasing": [256, 512, 1024],
        "xwide_decreasing": [2048, 1024, 512],
        "xwide_constant": [2048, 2048, 2048],
        "xwide_increasing": [512, 1024, 2048],
    }


def build_layers(widths, dropout_rate, use_batchnorm):
    layers = []
    for i, width in enumerate(widths):
        layers.append({"type": "linear", "size": width})
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if dropout_rate > 0 and i < len(widths) - 1:
            layers.append({"type": "dropout", "rate": dropout_rate})
    layers.append({"type": "linear", "size": 10})
    return layers


def run_width_grid_search(dataset_name="mnist"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset_name == "mnist":
        full_train_loader, test_loader = get_mnist_loaders(batch_size=64)
        input_size = 784
        epochs = 5
    else:
        full_train_loader, test_loader = get_cifar_loaders(batch_size=64)
        input_size = 3072
        epochs = 10

    full_train_dataset = full_train_loader.dataset
    val_size = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    param_grid = {
        'width_scheme': list(create_width_architectures().items()),
        'dropout_rate': [0.0, 0.2, 0.4],
        'use_batchnorm': [False, True]
    }

    results = {}
    best_val_acc = 0.0
    best_config = None

    for params in ParameterGrid(param_grid):
        scheme_name, widths = params['width_scheme']
        dropout_rate = params['dropout_rate']
        use_batchnorm = params['use_batchnorm']

        model_name = f"3_layers_{scheme_name}"
        if dropout_rate > 0:
            model_name += f"_dropout{dropout_rate}"
        if use_batchnorm:
            model_name += "_batchnorm"

        print(f"\n[Grid Search] Training: {model_name}")

        layers = build_layers(widths, dropout_rate, use_batchnorm)
        model = FullyConnectedModel(
            input_size=input_size,
            num_classes=10,
            layers=layers
        ).to(device)

        start_time = time.time()
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=epochs,
            device=device
        )
        training_time = time.time() - start_time

        final_val_acc = history['test_accs'][-1]  # на самом деле — val accuracy

        results[model_name] = {
            'width_scheme': scheme_name,
            'widths': widths,
            'dropout_rate': dropout_rate,
            'use_batchnorm': use_batchnorm,
            'final_val_acc': final_val_acc,
            'final_train_acc': history['train_accs'][-1],
            'val_accs': history['test_accs'],
            'train_accs': history['train_accs'],
            'params_count': count_parameters(model),
            'training_time_sec': training_time
        }

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_config = model_name

    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.JSON')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Лучшая конфигурация: {best_config}")
    print(f"Валидационная точность: {best_val_acc:.4f}")

    return results, best_config


run_width_grid_search()