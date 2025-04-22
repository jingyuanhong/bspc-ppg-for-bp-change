# train_ppg_model.py
# This script trains classification models to predict blood pressure changes from PPG signals.
# Data is expected in HDF5 format, prepared via MATLAB preprocessing, and placed in "data/".

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GroupKFold
import gc

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

        ppg1_d2 = np.diff(np.diff(ppg1, prepend=ppg1[0]), prepend=ppg1[0])
        ppg2_d2 = np.diff(np.diff(ppg2, prepend=ppg2[0]), prepend=ppg2[0])

        def min_max_normalize(feature):
            return (feature - feature.min()) / (feature.max() - feature.min())

        ppg1_d2 = min_max_normalize(ppg1_d2)
        ppg2_d2 = min_max_normalize(ppg2_d2)

        ppg_stacked = np.stack([ppg1, ppg1_d2, ppg2, ppg2_d2])
        label = np.argmax(file['label'][idx_in_file])

        return (torch.from_numpy(ppg_stacked).float(), torch.tensor(sbp).float()), torch.tensor(label, dtype=torch.long)

    def get_subject_idx(self):
        return np.concatenate([f['subject_idx'][:].flatten() for f in self.file_handles])

    def __del__(self):
        for f in self.file_handles:
            f.close()

# ---------------- Model Architecture ----------------
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.dense = nn.Linear(256, 256)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_weights = self.softmax(x)
        return torch.sigmoid(self.dense(attn_weights * x))

class Encoder(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels + 1, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(65, 128, 11, padding=5)
        self.conv3 = nn.Conv1d(129, 256, 21, padding=10)

        self.norm1 = nn.InstanceNorm1d(64)
        self.norm2 = nn.InstanceNorm1d(128)
        self.norm3 = nn.InstanceNorm1d(256)

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.2)
        self.attention = Attention()

        self.classifier = nn.Linear(256 * 218, num_classes)

        self.fc_sbp1 = nn.Linear(1, 875)
        self.fc_sbp2 = nn.Linear(1, 437)
        self.fc_sbp3 = nn.Linear(1, 218)

    def forward(self, x, sbp):
        sbp_feat = self.fc_sbp1(sbp.unsqueeze(1))
        x = torch.cat((x, sbp_feat.unsqueeze(1)), 1)
        x = self.prelu(self.norm1(self.conv1(x)))
        x = self.dropout(F.max_pool1d(x, 2))

        sbp_feat = self.fc_sbp2(sbp.unsqueeze(1))
        x = torch.cat((x, sbp_feat.unsqueeze(1)), 1)
        x = self.prelu(self.norm2(self.conv2(x)))
        x = self.dropout(F.max_pool1d(x, 2))

        sbp_feat = self.fc_sbp3(sbp.unsqueeze(1))
        x = torch.cat((x, sbp_feat.unsqueeze(1)), 1)
        x = self.prelu(self.norm3(self.conv3(x)))
        x = self.dropout(x)

        x = self.attention(x).permute(0, 2, 1)
        x = self.norm3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)

            outputs = model(inputs[0], inputs[1])
            loss = criterion(outputs, labels) + l1_regularization(model, 1e-5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                outputs = model(inputs[0], inputs[1])
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Val Loss={avg_val_loss:.4f}")

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
        model = model_cls(num_classes=3, in_channels=4).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=2048, shuffle=True, num_workers=4)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=2048, shuffle=False, num_workers=4)

        model_path = f"{model_path_base}_fold{fold+1}.pth"
        best_val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_path)

        results.append({"fold": fold+1, "val_loss": best_val_loss})
    return results

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model_output_dir = "./models"
    os.makedirs(model_output_dir, exist_ok=True)

    datasets = {
        "SBP": ("././output/sbp_trn_30_part{}.h5", os.path.join(model_output_dir, "sbp"))
    }

    for name, (pattern, model_path) in datasets.items():
        print(f"Training {name} model...")
        dataset = MemoryEfficientPPGDataset(pattern)
        subject_idx = dataset.get_subject_idx()
        results = cross_validate(dataset, subject_idx, Encoder, criterion, num_epochs=200, model_path_base=model_path)

        for res in results:
            print(f"{name} Fold {res['fold']} - Best Val Loss: {res['val_loss']:.4f}")

        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("All Encoder models trained.")