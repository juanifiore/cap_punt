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
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        
    def forward(self, embeddings, lengths):
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)

        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [batch, seq_len, hidden_dim]

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

def collate_fn(batch):
    X_batch, y_cap_batch, y_ini_batch, y_fin_batch = zip(*batch)
    
    # Padding con 0.0 para X (inputs)
    X_pad = pad_sequence(X_batch, batch_first=True, padding_value=0.0)   # [batch, max_seq_len, embed_dim]
    
    print(f"X_pad shape: {X_pad.shape}")

    # Padding con -100 para etiquetas, que será ignore_index en la loss
    y_cap_pad = pad_sequence(y_cap_batch, batch_first=True, padding_value=-100)
    y_ini_pad = pad_sequence(y_ini_batch, batch_first=True, padding_value=-100)
    y_fin_pad = pad_sequence(y_fin_batch, batch_first=True, padding_value=-100)
    
    # Calculo longitudes originales (sin contar padding)
    lengths = torch.tensor([len(x) for x in X_batch], dtype=torch.long)
    
    return X_pad, y_cap_pad, y_ini_pad, y_fin_pad, lengths

def evaluate_model(model, dataloader, mode, device):
    model.eval()
    
    true_cap, pred_cap = [], []
    true_ini, pred_ini = [], []
    true_fin, pred_fin = [], []

    with torch.no_grad():
        for X_batch, y_cap_batch, y_ini_batch, y_fin_batch, lengths in dataloader:
            X_batch = X_batch.to(device)
            y_cap_batch = y_cap_batch.to(device)
            y_ini_batch = y_ini_batch.to(device)
            y_fin_batch = y_fin_batch.to(device)
            lengths = lengths.to(device)

            logits_cap, logits_ini, logits_fin = model(X_batch, lengths)
            
            pred_cap_batch = logits_cap.argmax(dim=-1).cpu().numpy()
            pred_ini_batch = logits_ini.argmax(dim=-1).cpu().numpy()
            pred_fin_batch = logits_fin.argmax(dim=-1).cpu().numpy()
            
            y_cap_batch = y_cap_batch.cpu().numpy()
            y_ini_batch = y_ini_batch.cpu().numpy()
            y_fin_batch = y_fin_batch.cpu().numpy()
            
            for i, length in enumerate(lengths):
                true_cap.extend(y_cap_batch[i][:length])
                pred_cap.extend(pred_cap_batch[i][:length])
                
                true_ini.extend(y_ini_batch[i][:length])
                pred_ini.extend(pred_ini_batch[i][:length])
                
                true_fin.extend(y_fin_batch[i][:length])
                pred_fin.extend(pred_fin_batch[i][:length])

    print(f"\n--- EVALUACIÓN DEL MODELO : {mode} ---")

    print("\n--- CAPITALIZACIÓN ---")
    print(classification_report(true_cap, pred_cap, target_names=["minúscula", "mayúscula", "Capitalizado", "Mixto"]))

    print("\n--- PUNTUACIÓN INICIAL ---")
    print(classification_report(true_ini, pred_ini))

    print("\n--- PUNTUACIÓN FINAL ---")
    print(classification_report(true_fin, pred_fin))

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["minúscula", "mayúscula", "Capitalizado", "Mixto"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Capitalización")
    plt.show()

def train_model(model, dataloader, optimizer, device):
    model.train()

    for epoch in range(20):
        total_loss = 0

        for X_batch, y_cap_batch, y_ini_batch, y_fin_batch, lengths in dataloader:
            X_batch = X_batch.to(device)
            y_cap_batch = y_cap_batch.to(device)
            y_ini_batch = y_ini_batch.to(device)
            y_fin_batch = y_fin_batch.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            
            logits_cap, logits_ini, logits_fin = model(X_batch, lengths)
            
            loss = loss_fn(logits_cap, logits_ini, logits_fin, y_cap_batch, y_ini_batch, y_fin_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

#%% Cargar datos
# Cargar datos

# Cargar el CSV

df = pd.read_csv("./tokens_etiquetados/tokens_etiquetados_or_fin1000_dim_152.csv")
p_inicial = ["", "¿"]
p_final = ["", ".", ",", "?"]

X, y_cap, y_punt_ini, y_punt_fin = preprocess_text(df)
dataset = EmbeddingSequenceDataset(X, y_cap, y_punt_ini, y_punt_fin)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Model

model = TextRestorationGRU(
    embed_dim=15,
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

criterion = nn.CrossEntropyLoss(ignore_index=-100)
criterion_cap = nn.CrossEntropyLoss(weight=weights_cap, ignore_index=-100)
criterion_ini = nn.CrossEntropyLoss(weight=weights_ini, ignore_index=-100)
criterion_fin = nn.CrossEntropyLoss(weight=weights_fin, ignore_index=-100)

# Optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_fn(logits_cap, logits_ini, logits_fin, y_cap, y_ini, y_fin):
    # logits: [batch, seq_len, num_classes]
    # targets: [batch, seq_len]
    loss_cap = criterion_cap(logits_cap.view(-1, logits_cap.size(-1)), y_cap.view(-1))
    loss_ini = criterion_ini(logits_ini.view(-1, logits_ini.size(-1)), y_ini.view(-1))
    loss_fin = criterion_fin(logits_fin.view(-1, logits_fin.size(-1)), y_fin.view(-1))
    return loss_cap + loss_ini + loss_fin

#%% Entrenamiento y evaluación del modelo
# Entrenamiento y evaluación del modelo

# Entrenamiento 1
print("Entrenamiento 1: Entrenamiento inicial")

train_model(model, dataloader, optimizer, device)

print("Entrenamiento 1 completado")

# Evaluación del modelo 1

evaluate_model(model, dataloader, 'Base', device)

# %%
