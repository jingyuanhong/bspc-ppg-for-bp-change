# test_encoder_model.py
# This script evaluates pre-trained encoder models on test datasets.
# Compatible with models trained using train_ppg_model.py and preprocessed .h5 datasets.

import os
import h5py
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score

# ---------------- Dataset Definition ----------------
class PPGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        ppg1 = sample[:875]
        ppg2 = sample[875:1750]
        sbp = sample[-1]

        def min_max_normalize(feature):
            return (feature - feature.min()) / (feature.max() - feature.min())

        ppg1_d2 = min_max_normalize(np.diff(np.diff(ppg1, prepend=ppg1[0]), prepend=ppg1[0]))
        ppg2_d2 = min_max_normalize(np.diff(np.diff(ppg2, prepend=ppg2[0]), prepend=ppg2[0]))

        ppg_stacked = np.stack([ppg1, ppg1_d2, ppg2, ppg2_d2])
        label = np.argmax(self.labels[idx])
        return (torch.tensor(ppg_stacked, dtype=torch.float32), torch.tensor(sbp, dtype=torch.float32)), torch.tensor(label, dtype=torch.long)

# ---------------- Data Loading ----------------
def load_data(file_path, label_key='label'):
    with h5py.File(file_path, 'r') as f:
        inputs = f['inputs'][:]
        labels = f[label_key][:]
    return inputs, labels

def create_dataloader(file_path, batch_size=2048, threshold=None, num_workers=4):
    label_key = f'label{threshold}' if threshold else 'label'
    data, labels = load_data(file_path, label_key)
    dataset = PPGDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ---------------- Model Architecture ----------------
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        weights = self.softmax(x)
        return torch.sigmoid(self.fc(weights * x))

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

# ---------------- Evaluation ----------------
def evaluate_model(model, loader, description):
    model.eval()
    correct, total, preds, targets = 0, 0, [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = [x.to(device) for x in inputs]
            labels = labels.to(device)
            outputs = model(inputs[0], inputs[1])
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    f1 = f1_score(targets, preds, average='weighted') * 100
    print(f"{description} Accuracy: {acc:.2f}%, F1 Score: {f1:.2f}%")

# ---------------- Main ----------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2048

    # Paths
    test_paths = [
        ("././output/sbp_tst_i_30.h5", "././models/encoder_ppg_sdppg/sbp_fold_best.pth", None, "SBP")
        # ("./data/dbp_tst_15.h5", "./models/dbp_fold5.pth", None, "DBP"),
        # ("./data/mbp_tst_20.h5", "./models/mbp_fold5.pth", None, "MBP"),
        ("././output/data/sbp_tst_ii.h5", "././models/encoder_ppg_sdppg/sbp_fold_best.pth", 30, "SBP (26 subs)")
        # ("./data/dbp_tst_26subs.h5", "./models/dbp_fold5.pth", 15, "DBP (26 subs)"),
        # ("./data/mbp_tst_26subs.h5", "./models/mbp_fold5.pth", 20, "MBP (26 subs)")
    ]

    for path, model_path, threshold, name in test_paths:
        print(f"Evaluating: {name}")
        loader = create_dataloader(path, batch_size=batch_size, threshold=threshold)
        model = Encoder(num_classes=3, in_channels=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        evaluate_model(model, loader, name)

    print("Evaluation complete.")