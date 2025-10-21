import torch
import time
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'base'))
from datasets import get_mnist_loaders, get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters, plot_training_history


base_architectures = {
    "1_layer": [
        {"type": "linear", "size": 10}
    ],
    "2_layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ],
    "3_layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ],
    "5_layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "linear", "size": 64},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ],
    "7_layers": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ]
}

dropout_architectures = {
    "1_layer_dropout": [
        {"type": "linear", "size": 10}
    ],
    "2_layers_dropout": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "size": 10}
    ],
    "3_layers_dropout": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 10}
    ],
    "5_layers_dropout": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 64},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.1},
        {"type": "linear", "size": 10}
    ],
    "7_layers_dropout": [
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.2},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.1},
        {"type": "linear", "size": 128},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.1},
        {"type": "linear", "size": 10}
    ]
}

batchnorm_architectures = {
    "1_layer_batchnorm": [
        {"type": "linear", "size": 10}
    ],
    "2_layers_batchnorm": [
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ],
    "3_layers_batchnorm": [
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ],
    "5_layers_batchnorm": [
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 64},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ],
    "7_layers_batchnorm": [
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 512},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 128},
        {"type": "batch_norm"},
        {"type": "relu"},
        {"type": "linear", "size": 10}
    ]
}

all_architectures = {
    **base_architectures,
    **dropout_architectures,
    **batchnorm_architectures
}

def test_with_mnist():
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    for name, layers in all_architectures.items():
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=layers
        ).to("cuda")
        
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device="cuda")
        training_time = time.time() - start_time
        
        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        
        save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')

        results_data = {
            "1_layer": {
                'train_accs': history['train_accs'],
                'test_accs': history['test_accs'],
                'train_losses': history['train_losses'],
                'test_losses': history['test_losses'],
                'final_train': final_train_acc,
                'final_test': final_test_acc,
                'params': count_parameters(model),
                'time': training_time
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)

        plot_training_history(history)


def test_with_cifar():
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    for name, layers in all_architectures.items():
        model = FullyConnectedModel(
            input_size=3072,
            num_classes=10,
            layers=layers
        ).to("cuda")
        
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, device="cuda")
        training_time = time.time() - start_time
        
        final_train_acc = history['train_accs'][-1]
        final_test_acc = history['test_accs'][-1]
        
        save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')

        results_data = {
            "1_layer": {
                'train_accs': history['train_accs'],
                'test_accs': history['test_accs'],
                'train_losses': history['train_losses'],
                'test_losses': history['test_losses'],
                'final_train': final_train_acc,
                'final_test': final_test_acc,
                'params': count_parameters(model),
                'time': training_time
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)

        plot_training_history(history)


test_with_mnist()
test_with_cifar()