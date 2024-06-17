import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.transforms import v2

import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator     # построение сетки на графиках
import numpy as np

import json
from tqdm import tqdm   # progressbar
from PIL import Image


plt.style.use("dark_background")
device = "cuda" if torch.cuda.is_available() else "cpu"

class MNISTDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.len_dataset = 0
        self.data_list = []

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split('\\')[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        file_path, target = self.data_list[index]
        sample = Image.open(file_path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class MyModel(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, output)
        )

    def forward(self, x):
        return self.model(x)



transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, ), std=(0.5, ))
    ]
)
train_data = MNISTDataset("C:\\My\\Projects\\MNIST_test\\data\\training", transform=transform)
test_data = MNISTDataset("C:\\My\\Projects\\MNIST_test\\data\\testing", transform=transform)

train_data, val_data = random_split(train_data, [0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


model = MyModel(784, 10).to(device)

loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode='min',
    factor=0.1,
    patience=5,
    threshold=0.0001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-08
)


EPOCHS = 40
train_loss = []
train_acc = []
val_loss = []
val_acc = []
lr_list = []
best_loss = None
temp_state_dict = None
count = 0       # Счетчик эпох в которых модель не улучшалась
old_best = None

for epoch in range(EPOCHS):
    # Тренировка модели
    model.train()
    running_train_loss = []
    true_answer = 0     # кол-во правильных ответов
    train_loop = tqdm(train_loader, leave=False)       # прогрессбар состояния обучения НС
    for x, targets in train_loop:
        # Данные
        # (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.reshape(-1, 28*28).to(device)
        # (batch_size, int) -> (batch_size, 10), dtype=float32
        targets = targets.reshape(-1).to(torch.int32)
        targets = torch.eye(10)[targets].to(device)

        # Прямой проход + расчет ошибки модели
        pred = model(x)
        loss = loss_model(pred, targets)

        # Обратный проход
        opt.zero_grad()
        loss.backward()

        # Шаг оптимизации
        opt.step()

        running_train_loss.append(loss.item())
        mean_train_loss = sum(running_train_loss)/len(running_train_loss)

        true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        # train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}")

    # Расчет значения метрики
    running_train_acc = true_answer / len(train_data)

    # Сохранение значения функции потерь и метрики
    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)


    # Проверка модели (валидация)
    model.eval()
    running_val_loss = []
    true_answer = 0
    with torch.no_grad():   # Запрет вычисления градиентов
        for x, targets in val_loader:
            # Данные
            # (batch_size, 1, 28, 28) -> (batch_size, 784)
            x = x.reshape(-1, 28 * 28).to(device)
            # (batch_size, int) -> (batch_size, 10), dtype=float32
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)

            # Прямой проход + расчет ошибки модели
            pred = model(x)
            loss = loss_model(pred, targets)

            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)

            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

        # Расчет значения метрики
        running_val_acc = true_answer / len(val_data)

        # Сохранение значения функции потерь и метрики
        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

        lr_scheduler.step(mean_val_loss)
        lr = lr_scheduler._last_lr[0]
        lr_list.append(lr)

        print(f"Epoch [{epoch+1}/{EPOCHS}], train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, "
                f"val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}")     # .4f - округление значения до 4 знаков после запятой

        if best_loss is None:
            best_loss = mean_val_loss

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            count = 0
            # temp_state_dict = model.state_dict()
            best_name = f"model state_dict_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), best_name)
            print(f"На эпохе - {epoch+1}, сохранена модель со значением функции потерь на валидации - {mean_val_loss:.4f}", end="\n\n")

            # Сохранение состояния модели
            checkpoint = {
                "class_to_idx": train_data.class_to_idx,    # сохр. словаря, который отвечает за позицию класса датасета в ванхот-векторе
                "state_model": model.state_dict(),
                "state_opt": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict(),
                "loss": {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_loss": best_loss
                },
                "metric": {
                    "train_acc": train_acc,
                    "val_acc": val_acc
                },
                "lr": lr_list,
                "epoch": {
                    "EPOCHS": EPOCHS,
                    "save_epoch": epoch
                }
            }


            # Удаление предыдущего состояния модели
            try:
                if old_best != None and old_best != best_name:
                    os.remove(old_best)
            except OSError as e:
                print(f"Ошибка при удалении файла: {e.strerror}")
            old_best = best_name

    # Кол-во эпох без улучшения должно быть меньше, чем кол-во эпох без изменения скорости обучения (patience = 5)
    if count >= 10:
        print(f"\033[31mОбучение остановлено на {epoch+1} эпохе.\033[0m")
        break
    count += 1



plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['loss_train', 'loss_val'])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['acc_train', 'acc_val'])
plt.show()

plt.plot(lr_list)
plt.legend(['Learn speed'])
plt.show()