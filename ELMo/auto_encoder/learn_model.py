import os
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from auto_encoder import AutoEncoder


class EmbeddingsDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        embedding = torch.FloatTensor(self.data[index, :])
        return embedding, embedding

    def __len__(self):
        return self.data.shape[0]


# Станадратный метод обучения нейронной сети с помощью градиентного оптимизатора. 
# В эпоху обучения минимизируем функцию потерь, в эпоху валидации ищем лучшие параметры модели по какой-либо метрике (в данном случае accuracy).
def fit_model(data_loaders, model, criterion, optimizer, lr_scheduler=None, num_epochs=25, device='cuda'):
    loss_history = {"train": [], "val": []}
    best_params = model.state_dict()
    best_loss = 0.0

    progress_bar = trange(num_epochs, desc="Epoch:")

    for _ in progress_bar:
        for mode in ["train", "val"]:
            running_loss = 0.0
            processed_size = 0

            if mode == "train":
                model.train()
            else:
                model.eval()

            for x_batch, y_batch in tqdm(data_loaders[mode], leave=False, desc=f"{mode} iter:"):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if mode == "train":
                    optimizer.zero_grad()
                    output = model(x_batch)
                else:
                    with torch.no_grad():
                        output = model(x_batch)

                loss = criterion(output, y_batch)
                running_loss += loss.item()
                processed_size += 1

                if mode == "train":
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / processed_size
            loss_history[mode].append(epoch_loss)
            progress_bar.set_description('{} Loss: {:.4f}'.format(
                mode, epoch_loss
            ))

            if mode == "train" and lr_scheduler is not None:
                lr_scheduler.step()

            if mode == "val" and epoch_loss < best_loss:
                best_params = model.state_dict()
                best_loss = epoch_loss

    return model, best_params, loss_history


sns.set(style="whitegrid", font_scale=1.4)

# Выбираем место для работы с тензорами.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Обучаемся на каждом датасете по отдельности.
for path in os.listdir("train"):
    data = np.load(path)
    
    # Разделяем выборку на обучение и валидацию.
    train_data, test_data = train_test_split(data, train_size=0.7, shuffle=False)
    data_loaders = {
        "train": DataLoader(EmbeddingsDataset(train_data), batch_size=1024, shuffle=True),
        "val": DataLoader(EmbeddingsDataset(test_data), batch_size=1024)
    }

    # Загружем автоэнкодер для обучения.
    model = AutoEncoder(input_dim=1024, hidden_dim=8)
    if os.path.isfile(rf"auto_encoder_params"):
        model.load_state_dict(torch.load(rf"auto_encoder_params"))
    model.to(device)

    # Средства оптимизации loss.
    optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # НЕпосредственно процесс обучения.
    model, best_params, loss_history = fit_model(data_loaders, model, criterion, optimizer, num_epochs=15)

    torch.save(model.state_dict(), "auto_encoder_params")

    plt.figure(figsize=(12, 8))
    plt.plot(loss_history['train'], label="train")
    plt.plot(loss_history['val'], label="val")
    plt.legend()
    plt.show()
