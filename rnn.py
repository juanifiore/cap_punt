#%% Librerías
# Librerías 

import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from collections import Counter
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

#%% Funciones y clases
# Funciones y clases

class TextRestorationGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, shared_dim, num_caps_tags, num_punt_ini_tags, num_punt_fin_tags, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.shared_layer = nn.Linear(hidden_dim, shared_dim)
        
        self.cap_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, num_caps_tags)
        )
        self.punt_ini_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, num_punt_ini_tags)
        )
        self.punt_fin_head = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, num_punt_fin_tags)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim) 
        
    def forward(self, embeddings):
        rnn_out, _ = self.gru(embeddings)  # [batch, seq_len, hidden_dim]
        rnn_out = self.dropout(rnn_out)
        rnn_out = self.norm(rnn_out)
        
        # Capa común con activación ReLU
        shared_rep = F.relu(self.shared_layer(rnn_out))  # [batch, seq_len, shared_dim]
        
        # Salidas separadas para capitalización y puntuación
        return (
            self.cap_head(shared_rep),         # logits capitalización
            self.punt_ini_head(shared_rep),    # logits puntuación inicial
            self.punt_fin_head(shared_rep),    # logits puntuación final
        )

class EmbeddingSequenceDataset(Dataset):
    def __init__(self, X, y_cap, y_punt_ini, y_punt_fin):
        self.X = X
        self.y_cap = y_cap
        self.y_punt_ini = y_punt_ini
        self.y_punt_fin = y_punt_fin

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_cap[idx], dtype=torch.long),
            torch.tensor(self.y_punt_ini[idx], dtype=torch.long),
            torch.tensor(self.y_punt_fin[idx], dtype=torch.long),
        )

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: tensor de pesos por clase o None para peso uniforme
        gamma: parámetro de focalización
        reduction: 'mean', 'sum' o 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch, classes]
        # targets: [batch] con clase como índice
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_class_weights(y, num_classes=4):
    all_labels = np.concatenate(y)
    freqs = Counter(all_labels)
    total = sum(freqs.values())
    weights = torch.tensor([
        total / freqs[i] if freqs[i] > 0 else 0.0
        for i in range(num_classes)
    ], dtype=torch.float32).to(device)
    return weights

def preprocess_text(text):
    embedding_cols = [f"dim_red_{i}" for i in range(15)]
    grouped = df.groupby("instancia_id")

    X, y_cap, y_punt_ini, y_punt_fin = [], [], [], []

    for _, group in grouped:
        X.append(group[embedding_cols].values)                      # shape: [seq_len, 15]
        y_cap.append(group["capitalización"].values)                # [seq_len]
        y_punt_ini.append(group["i_punt_inicial"].values)           # [seq_len]
        y_punt_fin.append(group["i_punt_final"].values)             # [seq_len]

    return X, y_cap, y_punt_ini, y_punt_fin

def evaluate_model(model, dataloader, mode, device):
    true_cap = []
    pred_cap = []

    true_ini = []
    pred_ini = []

    true_fin = []
    pred_fin = []

    model.eval()

    with torch.no_grad():
        for X_batch, y_cap_batch, y_ini_batch, y_fin_batch in dataloader:
            X_batch = [x.to(device) for x in X_batch]
            y_cap_batch = [y.to(device) for y in y_cap_batch]
            y_ini_batch = [y.to(device) for y in y_ini_batch]
            y_fin_batch = [y.to(device) for y in y_fin_batch]

            for x, y_cap, y_ini, y_fin in zip(X_batch, y_cap_batch, y_ini_batch, y_fin_batch):
                x = x.unsqueeze(0)  # [1, seq_len, 15]
                logits_cap, logits_ini, logits_fin = model(x)

                pred_cap_logits = logits_cap.squeeze(0).argmax(dim=-1).cpu().numpy()
                pred_ini_logits = logits_ini.squeeze(0).argmax(dim=-1).cpu().numpy()
                pred_fin_logits = logits_fin.squeeze(0).argmax(dim=-1).cpu().numpy()

                true_cap.extend(y_cap.cpu().numpy())
                pred_cap.extend(pred_cap_logits)

                true_ini.extend(y_ini.cpu().numpy())
                pred_ini.extend(pred_ini_logits)

                true_fin.extend(y_fin.cpu().numpy())
                pred_fin.extend(pred_fin_logits)

    print(f"\n--- EVALUACIÓN DEL MODELO : {mode} ---")

    print("\n--- CAPITALIZACIÓN ---")
    print(classification_report(true_cap, pred_cap, target_names=["Minúscula", "Capitalizado", "Mixto", "Mayúscula"]))
    print("Accuracy:", accuracy_score(true_cap, pred_cap))
    print("F1-macro:", f1_score(true_cap, pred_cap, average='macro'))
    print("F1-weighted:", f1_score(true_cap, pred_cap, average='weighted'))
    print("F1-micro:", f1_score(true_cap, pred_cap, average='micro'))

    print("\n--- PUNTUACIÓN INICIAL ---")
    print(classification_report(true_ini, pred_ini))
    print("Accuracy:", accuracy_score(true_ini, pred_ini))
    print("F1-macro:", f1_score(true_ini, pred_ini, average='macro'))
    print("F1-weighted:", f1_score(true_ini, pred_ini, average='weighted'))
    print("F1-micro:", f1_score(true_ini, pred_ini, average='micro'))

    print("\n--- PUNTUACIÓN FINAL ---")
    print(classification_report(true_fin, pred_fin))
    print("Accuracy:", accuracy_score(true_fin, pred_fin))
    print("F1-macro:", f1_score(true_fin, pred_fin, average='macro'))
    print("F1-weighted:", f1_score(true_fin, pred_fin, average='weighted'))
    print("F1-micro:", f1_score(true_fin, pred_fin, average='micro'))

    # Matriz de confusión 1

    cm = confusion_matrix(true_fin, pred_fin, labels=[0,1,2,3,4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nada', '¿', '?', '.', ','])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Puntuación Final")
    plt.show()

    cm = confusion_matrix(true_ini, pred_ini, labels=[0,1,2,3,4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nada', '¿', '?', '.', ','])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Puntuación Inicial")
    plt.show()

    cm = confusion_matrix(true_cap, pred_cap, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Minúscula", "Capitalizado", "Mixto", "Mayúscula"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Capitalización")
    plt.show()

def train_model(model, dataloader, criterion, optimizer, device):
    if len(criterion) == 1:
        criterion_cap, criterion_ini, criterion_fin = criterion[0], criterion[0], criterion[0]
    else: 
        criterion_cap, criterion_ini, criterion_fin = criterion
    
    for epoch in range(50):
        model.train()
        total_loss = 0

        for X_batch, y_cap_batch, y_ini_batch, y_fin_batch in dataloader:
            X_batch = [x.to(device) for x in X_batch]
            y_cap_batch = [y.to(device) for y in y_cap_batch]
            y_ini_batch = [y.to(device) for y in y_ini_batch]
            y_fin_batch = [y.to(device) for y in y_fin_batch]

            optimizer.zero_grad()
            batch_loss = 0

            for x, y_cap, y_ini, y_fin in zip(X_batch, y_cap_batch, y_ini_batch, y_fin_batch):
                x = x.unsqueeze(0)  # [1, seq_len, 15]
                logits_cap, logits_ini, logits_fin = model(x)

                loss_cap = criterion_cap(logits_cap.squeeze(0), y_cap)
                loss_ini = criterion_ini(logits_ini.squeeze(0), y_ini)
                loss_fin = criterion_fin(logits_fin.squeeze(0), y_fin)

                loss = loss_cap + loss_ini + loss_fin
                loss.backward()
                batch_loss += loss.item()

            optimizer.step()
            total_loss += batch_loss

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

#%% Cargar datos
# Cargar datos

# Cargar el CSV

df = pd.read_csv("./tokens_etiquetados/tokens_etiquetados_or_fin1000_dim_152.csv")
p_inicial = ["", "¿"]
p_final = ["", ".", ",", "?"]

X, y_cap, y_punt_ini, y_punt_fin = preprocess_text(df)
dataset = EmbeddingSequenceDataset(X, y_cap, y_punt_ini, y_punt_fin)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: list(zip(*x)))

# Model
embed_dim = 15  # Dimensión de las embeddings
model = TextRestorationGRU(
    embed_dim=embed_dim,
    hidden_dim=64,
    shared_dim=32,
    num_caps_tags=4,
    num_punt_ini_tags=5,  
    num_punt_fin_tags=5,
).to(device)

# Pesos por clase

weights_cap = get_class_weights(y_cap)
weights_fin = get_class_weights(y_punt_fin, num_classes=5)
weights_ini = get_class_weights(y_punt_ini, num_classes=5)

# Criterios de pérdida

criterion = nn.CrossEntropyLoss()
criterion_cap = nn.CrossEntropyLoss(weight=weights_cap)
criterion_ini = nn.CrossEntropyLoss(weight=weights_ini)
criterion_fin = nn.CrossEntropyLoss(weight=weights_fin)

# Optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#%% Entrenamiento y evaluación del modelo
# Entrenamiento y evaluación del modelo

# Entrenamiento 1
print("Entrenamiento 1: Entrenamiento inicial")

train_model(model, dataloader, [criterion], optimizer, device)

print("Entrenamiento 1 completado")

# Evaluación del modelo 1

evaluate_model(model, dataloader, 'Base', device)

# Entrenamiento 2 / Finetuning con pesos para clases desbalanceadas
print("Entrenamiento 2: Finetuning con pesos para clases desbalanceadas")

train_model(model, dataloader, [criterion_cap, criterion_ini, criterion_fin], optimizer, device)

print("Entrenamiento 2 completado")

# Evaluación del modelo 2

evaluate_model(model, dataloader, 'Weights', device)

# %%
