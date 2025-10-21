import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compatre_final_accuracy():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    plt.figure(figsize=(14, 8))

    model_names = []
    train_accs = []
    test_accs = []
    colors = []

    for model_name, model_data in data.items():
        model_names.append(model_name.replace('_', ' ').title())
        train_accs.append(model_data["final_train_acc"])
        test_accs.append(model_data["final_test_acc"])
        
        if "dropout" in model_name and "batchnorm" in model_name:
            colors.append('#9467bd')
        elif "dropout" in model_name:
            colors.append('#1f77b4')
        elif "batchnorm" in model_name:
            colors.append('#ff7f0e')
        elif "l2" in model_name:
            colors.append('#2ca02c')
        else:
            colors.append("#d62728")

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = plt.bar(x - width/2, train_accs, width, label='Train Accuracy', color=colors, alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, test_accs, width, label='Test Accuracy', color=colors, alpha=0.4, edgecolor='black')

    plt.title('Сравнение итоговой точности моделей с разными методами регуляризации', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Методы регуляризации', fontsize=12)
    plt.ylabel('Точность (%)', fontsize=12)
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars1, train_accs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    for bar, acc in zip(bars2, test_accs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.ylim(0, 85)
    plt.tight_layout()
    plt.show()


def analyse_education_stability():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fig, ((ax1, ax2)) = plt.subplots(2, figsize=(16, 12))

    colors = {
        'no_regularization': '#d62728',
        'dropout_0.1': "#000000", 
        'dropout_0.3': '#1f77b4',
        'dropout_0.5': "#1fb4b4",
        'batchnorm_only': '#ff7f0e',
        'dropout_0.3_batchnorm': '#9467bd',
        'l2_regularization': '#2ca02c'
    }

    ax1.set_title('Динамика потерь (Loss) во время обучения', fontsize=14, fontweight='bold')
    for model_name, model_data in data.items():
        epochs = range(1, len(model_data['train_losses']) + 1)
        ax1.plot(epochs, model_data['train_losses'], 
                label=f'{model_name} (train)', color=colors[model_name], linewidth=2)
        ax1.plot(epochs, model_data['test_losses'], 
                label=f'{model_name} (test)', color=colors[model_name], linestyle='--', linewidth=2)
    ax1.set_xlabel('Эпохи')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Динамика точности (Accuracy) во время обучения', fontsize=14, fontweight='bold')
    for model_name, model_data in data.items():
        epochs = range(1, len(model_data['train_accs']) + 1)
        ax2.plot(epochs, model_data['train_accs'], 
                label=f'{model_name} (train)', color=colors[model_name], linewidth=2)
        ax2.plot(epochs, model_data['test_accs'], 
                label=f'{model_name} (test)', color=colors[model_name], linestyle='--', linewidth=2)
    ax2.set_xlabel('Эпохи')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def display_weights():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    colors = {
        'no_regularization': '#d62728',
        'dropout_0.1': '#000000', 
        'dropout_0.3': '#1f77b4',
        'dropout_0.5': '#1fb4b4',
        'batchnorm_only': '#ff7f0e',
        'dropout_0.3_batchnorm': '#9467bd',
        'l2_regularization': '#2ca02c'
    }

    for i, (model_name, model_data) in enumerate(data.items()):
        if i >= len(axes):
            break
            
        weight_stats = model_data['weight_stats']
        
        mean = weight_stats['mean']
        std = weight_stats['std']
        min_val = weight_stats['min']
        max_val = weight_stats['max']
        
        weights = np.random.normal(mean, std, 10000)
        weights = np.clip(weights, min_val, max_val)
        
        n, bins, patches = axes[i].hist(weights, bins=50, color=colors[model_name], 
                                    alpha=0.7, edgecolor='black', linewidth=0.5, density=True)
        
        axes[i].set_title(f'{model_name.replace("_", " ").title()}\nσ: {std:.3f}', 
                        fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Значение веса')
        axes[i].set_ylabel('Плотность вероятности')
        axes[i].grid(True, alpha=0.3)
        
        axes[i].axvline(mean, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean:.3f}')
        axes[i].axvline(weight_stats['q1'], color='orange', linestyle=':', 
                    linewidth=1, alpha=0.7, label=f'Q1: {weight_stats["q1"]:.3f}')
        axes[i].axvline(weight_stats['median'], color='green', linestyle=':', 
                    linewidth=1, alpha=0.7, label=f'Median: {weight_stats["median"]:.3f}')
        axes[i].axvline(weight_stats['q3'], color='orange', linestyle=':', 
                    linewidth=1, alpha=0.7, label=f'Q3: {weight_stats["q3"]:.3f}')
        
        axes[i].legend(fontsize=8)

    for j in range(len(data.items()), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Распределение весов в моделях с разными методами регуляризации', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def analyse_affect_on_layers():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#3.2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = []
    accuracies = []
    layer_std = {f"layer_{i}": [] for i in range(4)}
    model_names = []

    for name, info in data.items():
        models.append(name)
        accuracies.append(info['metrics']['final_test_acc'])
        model_names.append(name)
        
        for layer in layer_std:
            if layer in info['layer_weights']:
                w = np.array(info['layer_weights'][layer])
                layer_std[layer].append(np.std(w))
            else:
                layer_std[layer].append(np.nan)

    std_matrix = np.array([layer_std[f"layer_{i}"] for i in range(4)]).T  # shape: (n_models, 4)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        std_matrix,
        xticklabels=[f"Слой {i}" for i in range(4)],
        yticklabels=model_names,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Стандартное отклонение весов'}
    )
    plt.title('Разброс весов по слоям для разных моделей')
    plt.xlabel('Слой сети')
    plt.ylabel('Модель')
    plt.tight_layout()
    plt.show()


# compatre_final_accuracy()
# analyse_education_stability()
# display_weights()
# analyse_affect_on_layers()