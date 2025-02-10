import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np

torch.set_float32_matmul_precision('high')

# Config
BATCH_SIZE = 256
NUM_WORKERS = 2
SIZE_H = SIZE_W = 128
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Albumentations Augmentations
train_transforms = A.Compose([
    A.Resize(height=SIZE_H, width=SIZE_W),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(height=SIZE_H, width=SIZE_W),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Dataset Wrapper
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)  # Преобразуем в numpy массив для Albumentations
        image = self.transform(image=image)['image']
        return image, label
    
    def __len__(self):
        return len(self.dataset)

# Model Definition
class CNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Размер изображения уменьшается в два раза
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Уменьшаем еще раз
             
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Еще одно уменьшение
        )
        
        # Для вычисления правильного размера, проходим через сверточные слои
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, SIZE_H, SIZE_W).to(DEVICE)
            output = self.conv_layers(dummy_input)
            conv_output_size = output.numel()  # Получаем количество элементов в тензоре после свертки
            print(f"Output shape after convolutions: {output.shape}, numel: {conv_output_size}")
        
        # Используем вычисленный размер для первого линейного слоя
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 256),  # Здесь правильно размер после свертки
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES),
        )
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(DEVICE)  # Перемещаем входные данные на GPU, если необходимо
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(DEVICE)  # Перемещаем входные данные на GPU, если необходимо
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

if __name__ == "__main__":
    DATA_PATH = Path("data").resolve()
    train_dataset = ImageFolder(DATA_PATH / "train_11k")
    val_dataset = ImageFolder(DATA_PATH / "val")

    train_dataset = AlbumentationsDataset(train_dataset, train_transforms)
    val_dataset = AlbumentationsDataset(val_dataset, val_transforms)

    # Используем pin_memory=True и persistent_workers=True для улучшения загрузки данных
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    model = CNNModel().to(DEVICE)  # Перемещаем модель на нужное устройство
    trainer = pl.Trainer(max_epochs=30, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)
    print("Готово! Модель обучена 🚀")

