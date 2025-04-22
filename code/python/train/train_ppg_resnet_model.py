# train_resnet_model.py
# This script trains a ResNet-based model for classifying blood pressure changes from PPG signals.
# The current version supports training using a small released dataset of 10 subjects. Full datasets will be released progressively.

import torch
import torch.nn.functional as F
import h5py
import numpy as np
import os
import gc
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold

# ---------------- Dataset Definition ----------------
class MemoryEfficientPPGDataset(Dataset):
    def __init__(self, file_pattern, num_files=6):
        self.file_pattern = file_pattern
        self.num_files = num_files
        self.file_handles = []
        self.cumulative_lengths = [0]
        self.total_length = 0

        for i in range(1, num_files + 1):
            file = h5py.File(file_pattern.format(i), 'r')
            self.file_handles.append(file)
            self.total_length += len(file['inputs'])
            self.cumulative_lengths.append(self.total_length)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        idx_in_file = idx - self.cumulative_lengths[file_idx]

        file = self.file_handles[file_idx]
        sample = file['inputs'][idx_in_file]
        ppg1 = sample[:875]
        ppg2 = sample[875:1750]
        sbp = sample[-1]
        ppg_stacked = np.stack([ppg1, ppg2])
        label = np.argmax(file['label'][idx_in_file])

        return (torch.from_numpy(ppg_stacked).float(), torch.tensor(sbp).float()), torch.tensor(label, dtype=torch.long)

    def get_subject_idx(self):
        return np.concatenate([file['subject_idx'][:].flatten() for file in self.file_handles])

    def __del__(self):
        for file in self.file_handles:
            file.close()

# ---------------- ResNet Model Definition ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.norm1 = nn.InstanceNorm1d(out_channels)
        self.norm2 = nn.InstanceNorm1d(out_channels)
        self.norm3 = nn.InstanceNorm1d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.norm1(x)
        out = self.prelu(self.norm1(self.conv1(x)))
        out = self.prelu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += residual
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes, seq_len=875):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels + 1, 64, kernel_size=7, padding=3)
        self.norm1 = nn.InstanceNorm1d(64)
        self.res_block1 = ResidualBlock(65, 65, [9, 5, 3])
        self.res_block2 = ResidualBlock(66, 66, [9, 5, 3])
        self.res_block3 = ResidualBlock(67, 67, [9, 5, 3])
        self.fc_sbp = nn.Linear(1, seq_len)
        self.prelu = nn.PReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(67, num_classes)

    def forward(self, x, sbp):
        sbp_feature = self.fc_sbp(sbp.unsqueeze(1))
        x = torch.cat((x, sbp_feature.unsqueeze(1)), dim=1)
        x = self.prelu(self.norm1(self.conv1(x)))
        x = torch.cat((x, sbp_feature.unsqueeze(1)), dim=1)
        x = self.res_block1(x)
        x = torch.cat((x, sbp_feature.unsqueeze(1)), dim=1)
        x = self.res_block2(x)
        x = torch.cat((x, sbp_feature.unsqueeze(1)), dim=1)
        x = self.res_block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# ---------------- Training Utilities ----------------
def l1_regularization(model, lambda_l1):
    return lambda_l1 * sum(torch.norm(p, 1) for p in model.parameters())

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.02):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        elif loss > (self.best_loss + self.min_delta):
            self.counter += 1
            return self.counter >= self.patience
        return False

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path):
    best_val_loss = float('inf')
    early_stopper = EarlyStopper()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)

            outputs = model(inputs[0], inputs[1])
            loss = criterion(outputs, labels) + l1_regularization(model, 1e-5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                outputs = model(inputs[0], inputs[1])
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={avg_val_loss:.4f}")

        if early_stopper.early_stop(avg_val_loss):
            print("Early stopping triggered.")
            break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")

    return best_val_loss

def cross_validate(dataset, subject_idx, model_cls, criterion, num_epochs, model_path_base):
    results = []
    gkf = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(dataset)), groups=subject_idx)):
        print(f"Fold {fold+1}/5")
        model = model_cls(input_channels=2, num_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=2048, shuffle=True, num_workers=4)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=2048, shuffle=False, num_workers=4)

        model_path = f"{model_path_base}_fold{fold+1}.pth"
        val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path)
        results.append({"fold": fold+1, "val_loss": val_loss})
    return results

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model_output_dir = "./models"
    os.makedirs(model_output_dir, exist_ok=True)

    datasets = {
        "SBP": ("././output/sbp_trn_30_part{}.h5", os.path.join(model_output_dir, "resnet_sbp"))
        # "DBP": ("././output/dbp_trn_15_part{}.h5", os.path.join(model_output_dir, "resnet_dbp")),
        # "MBP": ("././output/mbp_trn_20_part{}.h5", os.path.join(model_output_dir, "resnet_mbp"))
    }

    for name, (pattern, model_path) in datasets.items():
        print(f"Training {name} model (ResNet)...")
        dataset = MemoryEfficientPPGDataset(pattern)
        subject_idx = dataset.get_subject_idx()
        results = cross_validate(dataset, subject_idx, ResNet, criterion, num_epochs=200, model_path_base=model_path)

        for res in results:
            print(f"{name} Fold {res['fold']} - Best Val Loss: {res['val_loss']:.4f}")

        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("All ResNet models trained.")