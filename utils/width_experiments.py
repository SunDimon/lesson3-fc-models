import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def compare_accuracy_and_time():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_models = {}
    for model_name, model_data in data.items():
        if model_data["dropout_rate"] == 0.0 and not model_data["use_batchnorm"]:
            widths = model_data["widths"]
            
            if widths == [64, 32, 16]:
                category = "Узкие слои"
            elif widths == [256, 128, 64]:
                category = "Средние слои"
            elif widths == [1024, 512, 256]:
                category = "Широкие слои"
            elif widths == [2048, 1024, 512]:
                category = "Очень широкие слои"
            else:
                continue
                
            filtered_models[model_name] = {
                "category": category,
                "final_train_acc": model_data["final_train_acc"],
                "training_time_sec": model_data["training_time_sec"],
                "widths": widths
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    categories = ["Узкие слои", "Средние слои", "Широкие слои", "Очень широкие слои"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    train_accs = []
    times = []

    for category in categories:
        category_models = [model for model in filtered_models.values() if model["category"] == category]
        if category_models:
            train_acc = category_models[0]["final_train_acc"]
            time = category_models[0]["training_time_sec"]
            train_accs.append(train_acc)
            times.append(time)

    bars1 = ax1.bar(categories, train_accs, color=colors, alpha=0.7)
    ax1.set_title('Final Training Accuracy по категориям ширины слоев', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Final Training Accuracy')
    ax1.set_ylim(0.95, 0.99)
    ax1.grid(True, alpha=0.3)

    for bar, acc in zip(bars1, train_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(categories, times, color=colors, alpha=0.7)
    ax2.set_title('Training Time по категориям ширины слоев', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.grid(True, alpha=0.3)

    for bar, time in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def compare_parameters_count():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories_data = {
        "Узкие слои": None,
        "Средние слои": None,
        "Широкие слои": None,
        "Очень широкие слои": None
    }

    for model_name, model_data in data.items():
        if model_data["dropout_rate"] == 0.0 and not model_data["use_batchnorm"]:
            widths = model_data["widths"]
            
            if widths == [64, 32, 16]:
                categories_data["Узкие слои"] = model_data["params_count"]
            elif widths == [256, 128, 64]:
                categories_data["Средние слои"] = model_data["params_count"]
            elif widths == [1024, 512, 256]:
                categories_data["Широкие слои"] = model_data["params_count"]
            elif widths == [2048, 1024, 512]:
                categories_data["Очень широкие слои"] = model_data["params_count"]

    plt.figure(figsize=(12, 8))
    categories = list(categories_data.keys())
    params_counts = list(categories_data.values())

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(categories, params_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    plt.title('Сравнение количества параметров по категориям ширины слоев', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Количество параметров', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, params_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                f'{count:,}'.replace(',', ' '), 
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()


def display_all_models():
    save_path = os.path.join(os.path.dirname(__file__), '..', 'logs', '#2.JSON')
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width_schemes = set()
    dropout_rates = set()
    batchnorm_flags = set()

    for model_name, model_data in data.items():
        width_schemes.add(model_data["width_scheme"])
        dropout_rates.add(model_data["dropout_rate"])
        batchnorm_flags.add(model_data["use_batchnorm"])

    width_schemes = sorted(list(width_schemes))
    dropout_rates = sorted(list(dropout_rates))
    batchnorm_flags = sorted(list(batchnorm_flags))

    heatmap_data = np.zeros((len(width_schemes), len(dropout_rates) * len(batchnorm_flags)))

    for i, width_scheme in enumerate(width_schemes):
        for j, dropout_rate in enumerate(dropout_rates):
            for k, use_batchnorm in enumerate(batchnorm_flags):
                for model_name, model_data in data.items():
                    if (model_data["width_scheme"] == width_scheme and 
                        model_data["dropout_rate"] == dropout_rate and 
                        model_data["use_batchnorm"] == use_batchnorm):
                        
                        col_index = j * len(batchnorm_flags) + k
                        heatmap_data[i, col_index] = model_data["final_train_acc"]
                        break

    column_labels = []
    for dropout_rate in dropout_rates:
        for use_batchnorm in batchnorm_flags:
            label = f"DR:{dropout_rate}\nBN:{use_batchnorm}"
            column_labels.append(label)

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=column_labels,
                yticklabels=width_schemes,
                cbar_kws={'label': 'Final Train Accuracy'})

    plt.title('Heatmap: Final Train Accuracy по конфигурациям моделей', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Конфигурация (Dropout Rate + BatchNorm)', fontsize=12)
    plt.ylabel('Схема ширины слоев', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# compare_accuracy_and_time()
# compare_parameters_count()
# display_all_models()