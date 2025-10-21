import os
import json
import matplotlib.pyplot as plt
import numpy as np


def define_best_depth_mnist():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    models = []
    models.append(("1_layer", (data["1_layer"]["final_test"])))
    models.append(("2_layers", (data["2_layers"]["final_test"])))
    models.append(("3_layers", (data["3_layers"]["final_test"])))
    models.append(("5_layers", (data["5_layers"]["final_test"])))
    models.append(("7_layers", (data["7_layers"]["final_test"])))
    best_model = ('', -1)
    for model in models:
        if best_model[1] < model[1]:
            best_model = model
    print("Лучшая глубина для MNIST:", best_model[0])


def define_best_depth_cifar():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    models = []
    models.append(("1_layer", (data["1_layer"]["final_test"])))
    models.append(("2_layers", (data["2_layers"]["final_test"])))
    models.append(("3_layers", (data["3_layers"]["final_test"])))
    models.append(("5_layers", (data["5_layers"]["final_test"])))
    models.append(("7_layers", (data["7_layers"]["final_test"])))
    best_model = ('', -1)
    for model in models:
        if best_model[1] < model[1]:
            best_model = model
    print("Лучшая глубина для CIFAR:", best_model[0])


def define_retraining_mnist():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for layers in data:
        for epoch in range(1, len(data[layers]["test_accs"])):
            if data[layers]["test_accs"][epoch - 1] > data[layers]["test_accs"][epoch] and data[layers]["test_losses"][epoch - 1] < data[layers]["test_losses"][epoch]:
                print(f"Переобучение модели {layers} на датасете MNIST началось на {epoch + 1} эпохе")
                break


def define_retraining_cifar():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for layers in data:
        for epoch in range(1, len(data[layers]["test_accs"])):
            if data[layers]["test_accs"][epoch - 1] > data[layers]["test_accs"][epoch] and data[layers]["test_losses"][epoch - 1] < data[layers]["test_losses"][epoch]:
                print(f"Переобучение модели {layers} на датасете CIFAR началось на {epoch + 1} эпохе")
                break


def compare_models_on_mnist():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_MNIST.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    overfitting_count = {}
    for layers in data:
        overfitting_count[layers] = 0
        for epoch in range(1, len(data[layers]["test_accs"])):
            if data[layers]["test_accs"][epoch - 1] > data[layers]["test_accs"][epoch] and data[layers]["test_losses"][epoch - 1] < data[layers]["test_losses"][epoch]:
                overfitting_count[layers] = 1
                break

    layers_count = [1, 2, 3, 5, 7]

    for i in layers_count:
        if i == 1:
            basic_acc = [data[f"{i}_layer"]["final_test"]]
            dropout_acc = [data[f"{i}_layer_dropout"]["final_test"]]
            batchnorm_acc = [data[f"{i}_layer_batchnorm"]["final_test"]]
        else:
            basic_acc = [data[f"{i}_layers"]["final_test"]]
            dropout_acc = [data[f"{i}_layers_dropout"]["final_test"]]
            batchnorm_acc = [data[f"{i}_layers_batchnorm"]["final_test"]]

    for i in layers_count:
        if i == 1:
            basic_time = [data[f"{i}_layer"]["time"]]
            dropout_time = [data[f"{i}_layer_dropout"]["time"]]
            batchnorm_time = [data[f"{i}_layer_batchnorm"]["time"]]
        else:
            basic_time = [data[f"{i}_layers"]["time"]]
            dropout_time = [data[f"{i}_layers_dropout"]["time"]]
            batchnorm_time = [data[f"{i}_layers_batchnorm"]["time"]]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    x = np.arange(len(layers_count))
    width = 0.25

    ax1.bar(x - width, basic_acc, width, label='Базовые', alpha=0.8)
    ax1.bar(x, dropout_acc, width, label='Dropout', alpha=0.8)
    ax1.bar(x + width, batchnorm_acc, width, label='BatchNorm', alpha=0.8)

    ax1.set_xlabel('Количество слоев')
    ax1.set_ylabel('Final Test Accuracy')
    ax1.set_title('Сравнение Final Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers_count)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(x - width, basic_time, width, label='Базовые', alpha=0.8)
    ax2.bar(x, dropout_time, width, label='Dropout', alpha=0.8)
    ax2.bar(x + width, batchnorm_time, width, label='BatchNorm', alpha=0.8)

    ax2.set_xlabel('Количество слоев')
    ax2.set_ylabel('Время обучения (сек)')
    ax2.set_title('Сравнение времени обучения')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers_count)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    basic_overfit = []
    dropout_overfit = []
    batchnorm_overfit = []

    for i in layers_count:
        if i == 1:
            basic_overfit.append(overfitting_count[f"{i}_layer"])
            dropout_overfit.append(overfitting_count[f"{i}_layer_dropout"])
            batchnorm_overfit.append(overfitting_count[f"{i}_layer_batchnorm"])
        else:
            basic_overfit.append(overfitting_count[f"{i}_layers"])
            dropout_overfit.append(overfitting_count[f"{i}_layers_dropout"])
            batchnorm_overfit.append(overfitting_count[f"{i}_layers_batchnorm"])

    ax3.bar(x - width, basic_overfit, width, label='Базовые', alpha=0.8)
    ax3.bar(x, dropout_overfit, width, label='Dropout', alpha=0.8)
    ax3.bar(x + width, batchnorm_overfit, width, label='BatchNorm', alpha=0.8)

    ax3.set_xlabel('Количество слоев')
    ax3.set_ylabel('Наличие переобучения')
    ax3.set_title('Наличие переобучения по моделям')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers_count)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Нет', 'Есть'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_models_on_cifar():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#1_CIFAR.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    overfitting_count = {}
    for layers in data:
        overfitting_count[layers] = 0
        for epoch in range(1, len(data[layers]["test_accs"])):
            if data[layers]["test_accs"][epoch - 1] > data[layers]["test_accs"][epoch] and data[layers]["test_losses"][epoch - 1] < data[layers]["test_losses"][epoch]:
                overfitting_count[layers] = 1
                break

    layers_count = [1, 2, 3, 5, 7]

    for i in layers_count:
        if i == 1:
            basic_acc = [data[f"{i}_layer"]["final_test"]]
            dropout_acc = [data[f"{i}_layer_dropout"]["final_test"]]
            batchnorm_acc = [data[f"{i}_layer_batchnorm"]["final_test"]]
        else:
            basic_acc = [data[f"{i}_layers"]["final_test"]]
            dropout_acc = [data[f"{i}_layers_dropout"]["final_test"]]
            batchnorm_acc = [data[f"{i}_layers_batchnorm"]["final_test"]]

    for i in layers_count:
        if i == 1:
            basic_time = [data[f"{i}_layer"]["time"]]
            dropout_time = [data[f"{i}_layer_dropout"]["time"]]
            batchnorm_time = [data[f"{i}_layer_batchnorm"]["time"]]
        else:
            basic_time = [data[f"{i}_layers"]["time"]]
            dropout_time = [data[f"{i}_layers_dropout"]["time"]]
            batchnorm_time = [data[f"{i}_layers_batchnorm"]["time"]]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    x = np.arange(len(layers_count))
    width = 0.25

    ax1.bar(x - width, basic_acc, width, label='Базовые', alpha=0.8)
    ax1.bar(x, dropout_acc, width, label='Dropout', alpha=0.8)
    ax1.bar(x + width, batchnorm_acc, width, label='BatchNorm', alpha=0.8)

    ax1.set_xlabel('Количество слоев')
    ax1.set_ylabel('Final Test Accuracy')
    ax1.set_title('Сравнение Final Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers_count)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(x - width, basic_time, width, label='Базовые', alpha=0.8)
    ax2.bar(x, dropout_time, width, label='Dropout', alpha=0.8)
    ax2.bar(x + width, batchnorm_time, width, label='BatchNorm', alpha=0.8)

    ax2.set_xlabel('Количество слоев')
    ax2.set_ylabel('Время обучения (сек)')
    ax2.set_title('Сравнение времени обучения')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers_count)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    basic_overfit = []
    dropout_overfit = []
    batchnorm_overfit = []

    for i in layers_count:
        if i == 1:
            basic_overfit.append(overfitting_count[f"{i}_layer"])
            dropout_overfit.append(overfitting_count[f"{i}_layer_dropout"])
            batchnorm_overfit.append(overfitting_count[f"{i}_layer_batchnorm"])
        else:
            basic_overfit.append(overfitting_count[f"{i}_layers"])
            dropout_overfit.append(overfitting_count[f"{i}_layers_dropout"])
            batchnorm_overfit.append(overfitting_count[f"{i}_layers_batchnorm"])

    ax3.bar(x - width, basic_overfit, width, label='Базовые', alpha=0.8)
    ax3.bar(x, dropout_overfit, width, label='Dropout', alpha=0.8)
    ax3.bar(x + width, batchnorm_overfit, width, label='BatchNorm', alpha=0.8)

    ax3.set_xlabel('Количество слоев')
    ax3.set_ylabel('Наличие переобучения')
    ax3.set_title('Наличие переобучения по моделям')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layers_count)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Нет', 'Есть'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# define_best_depth_mnist()
# define_best_depth_cifar()
# define_retraining_mnist()
# define_retraining_cifar()
# compare_models_on_mnist()
# compare_models_on_cifar()