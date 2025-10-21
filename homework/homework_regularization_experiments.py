import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from trainer import train_model
from datasets import get_cifar_loaders, get_mnist_loaders
from models import FullyConnectedModel
from utils import count_parameters


DATASET = "cifar"
ARCHITECTURE = [256, 256, 256]
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def build_layers_with_config(dropout_rate=None, use_batchnorm=False):
    layers = []
    widths = ARCHITECTURE
    
    for i, width in enumerate(widths):
        layers.append({"type": "linear", "size": width})
        if use_batchnorm:
            layers.append({"type": "batch_norm"})
        layers.append({"type": "relu"})
        if dropout_rate is not None and dropout_rate > 0:
            layers.append({"type": "dropout", "rate": dropout_rate})
    
    layers.append({"type": "linear", "size": 10})
    return layers


def train_with_regularization(model, train_loader, test_loader, weight_decay=0.0, device='cpu', epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    
    train_accs, test_accs = [], []
    train_losses, test_losses = [], []
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)
        
        train_acc = 100. * correct / total
        train_loss = loss_sum / total
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                loss_sum += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        test_acc = 100. * correct / total
        test_loss = loss_sum / total
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    return {
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses
    }


def run_regularization_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, test_loader = get_cifar_loaders(batch_size=BATCH_SIZE)
    input_size = 3072

    experiments = [
        ("no_regularization", None, False, 0.0),
        ("dropout_0.1", 0.1, False, 0.0),
        ("dropout_0.3", 0.3, False, 0.0),
        ("dropout_0.5", 0.5, False, 0.0),
        ("batchnorm_only", None, True, 0.0),
        ("dropout_0.3_batchnorm", 0.3, True, 0.0),
        ("l2_regularization", None, False, 1e-4),
    ]

    results = {}

    for name, dropout_rate, use_batchnorm, weight_decay in experiments:
        print(f"Dropout: {dropout_rate}, BatchNorm: {use_batchnorm}, Weight Decay: {weight_decay}")
        
        layers = build_layers_with_config(dropout_rate=dropout_rate, use_batchnorm=use_batchnorm)
        model = FullyConnectedModel(input_size=input_size, num_classes=10, layers=layers)
        
        start_time = time.time()
        history = train_with_regularization(
            model, train_loader, test_loader,
            weight_decay=weight_decay,
            device=device,
            epochs=EPOCHS
        )
        training_time = time.time() - start_time
        
        results[name] = {
            'dropout_rate': dropout_rate,
            'use_batchnorm': use_batchnorm,
            'weight_decay': weight_decay,
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'overfitting_gap': history['train_accs'][-1] - history['test_accs'][-1],
            'train_accs': history['train_accs'],
            'test_accs': history['test_accs'],
            'train_losses': history['train_losses'],
            'test_losses': history['test_losses'],
            'params_count': count_parameters(model),
            'training_time_sec': training_time
        }
        
        print(f"Завершено. Test Acc: {results[name]['final_test_acc']:.2f}%, "
              f"Overfitting gap: {results[name]['overfitting_gap']:.2f}%")

    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.JSON')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results


class AdaptiveDropout(nn.Module):
    def __init__(self, start_p=0.5, end_p=0.1, total_epochs=10):
        super().__init__()
        self.start_p = start_p
        self.end_p = end_p
        self.total_epochs = total_epochs
        self.current_p = start_p

    def set_epoch(self, epoch):
        progress = min(epoch / self.total_epochs, 1.0)
        self.current_p = self.start_p + (self.end_p - self.start_p) * progress

    def forward(self, x):
        if self.training:
            return nn.functional.dropout(x, p=self.current_p, training=True)
        return x


class AdaptiveBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, start_momentum=0.1, end_momentum=0.01, total_epochs=10):
        super().__init__(num_features, momentum=start_momentum)
        self.start_momentum = start_momentum
        self.end_momentum = end_momentum
        self.total_epochs = total_epochs

    def set_epoch(self, epoch):
        progress = min(epoch / self.total_epochs, 1.0)
        new_mom = self.start_momentum + (self.end_momentum - self.start_momentum) * progress
        self.momentum = new_mom


class AdaptiveModel(nn.Module):
    def __init__(self, input_size, num_classes, arch, 
                 use_adaptive_dropout=False, dropout_config=None,
                 use_adaptive_bn=False, bn_config=None,
                 total_epochs=10):
        super().__init__()
        self.total_epochs = total_epochs
        layers = []
        prev_size = input_size
        
        for i, width in enumerate(arch):
            layers.append(nn.Linear(prev_size, width))
            prev_size = width
            
            if use_adaptive_bn:
                cfg = bn_config or {}
                start_mom = cfg.get('start_momentum', 0.1)
                end_mom = cfg.get('end_momentum', 0.01)
                layers.append(AdaptiveBatchNorm1d(width, start_momentum=start_mom, end_momentum=end_mom, total_epochs=total_epochs))
            
            layers.append(nn.ReLU())
            
            if use_adaptive_dropout:
                cfg = dropout_config or {}
                start_p = cfg.get('start_p', 0.5)
                end_p = cfg.get('end_p', 0.1)
                layers.append(AdaptiveDropout(start_p=start_p, end_p=end_p, total_epochs=total_epochs))
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.layers = nn.Sequential(*layers)

    def set_epoch(self, epoch):
        for layer in self.modules():
            if isinstance(layer, (AdaptiveDropout, AdaptiveBatchNorm1d)):
                layer.set_epoch(epoch)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
    

def extract_layer_weights(model):
    weights_by_layer = {}
    linear_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            module_path = '.'.join(name.split('.')[:-1])
            try:
                module = model.get_submodule(module_path)
                if isinstance(module, nn.Linear):
                    weights_by_layer[f"layer_{linear_idx}"] = param.data.cpu().numpy().flatten().tolist()
                    linear_idx += 1
            except:
                continue
    return weights_by_layer


def train_adaptive_model(model, train_loader, test_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    
    for epoch in range(epochs):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(epoch)
        
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_acc = 100. * correct / total
    return test_acc


def run_adaptive_techniques_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    DATASET = "cifar"
    if DATASET == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=64)
        input_size = 784
    else:
        train_loader, test_loader = get_cifar_loaders(batch_size=64)
        input_size = 3072
    
    ARCH = [256, 256, 256]
    EPOCHS = 15

    experiments = [
        ("static_dropout", False, None, False, None),
        ("adaptive_dropout", True, {"start_p": 0.5, "end_p": 0.1}, False, None),
        ("adaptive_bn", False, None, True, {"start_momentum": 0.1, "end_momentum": 0.01}),
        ("adaptive_both", True, {"start_p": 0.4, "end_p": 0.2}, 
                         True, {"start_momentum": 0.1, "end_momentum": 0.01}),
        ("adaptive_dropout_high", True, {"start_p": 0.7, "end_p": 0.3}, False, None),
        ("adaptive_bn_slow", False, None, True, {"start_momentum": 0.2, "end_momentum": 0.05}),
    ]

    results = {}

    for name, use_drop, drop_cfg, use_bn, bn_cfg in experiments:
        print(f"\nЗапуск: {name}")
        
        # Создаём модель
        if name == "static_dropout":
            layers = []
            for w in ARCH:
                layers.append({"type": "linear", "size": w})
                layers.append({"type": "relu"})
                layers.append({"type": "dropout", "rate": 0.3})
            layers.append({"type": "linear", "size": 10})
            model = FullyConnectedModel(input_size=input_size, num_classes=10, layers=layers)
        else:
            model = AdaptiveModel(
                input_size=input_size,
                num_classes=10,
                arch=ARCH,
                use_adaptive_dropout=use_drop,
                dropout_config=drop_cfg,
                use_adaptive_bn=use_bn,
                bn_config=bn_cfg,
                total_epochs=EPOCHS
            )
        
        param_count = count_parameters(model)
        print(f"  Параметров: {param_count:,}")
        
        start_time = time.time()
        test_acc = train_adaptive_model(model, train_loader, test_loader, EPOCHS, device)
        training_time = time.time() - start_time
        
        layer_weights = extract_layer_weights(model)
        
        results[name] = {
            "config": {
                "use_adaptive_dropout": use_drop,
                "dropout_config": drop_cfg,
                "use_adaptive_bn": use_bn,
                "bn_config": bn_cfg,
                "architecture": ARCH,
                "epochs": EPOCHS
            },
            "metrics": {
                "final_test_acc": test_acc,
                "params_count": param_count,
                "training_time_sec": training_time
            },
            "layer_weights": layer_weights
        }
        
        print(f"  Test Acc: {test_acc:.2f}%")

    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.2.JSON')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    return results


run_regularization_study()
run_adaptive_techniques_study()
