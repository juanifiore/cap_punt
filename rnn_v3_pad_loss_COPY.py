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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
import sympy
from sklearn.model_selection import train_test_split


seed = 42  
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

embed_dim = 256
atencion = False 
bi = False 
train_test = False

df = pd.read_csv(f"./tokens_etiquetados/dataset_final_DEV1.csv")
#df = df[df['instancia_id']==1]
df['i_punt_final'] = df['i_punt_final'].replace({0: 0, 2: 1, 3: 2, 4: 3})

instancias = df['instancia_id'].unique()
print(f"Total de instancias: {len(instancias)}")

if train_test: 
    train_ids, test_ids = train_test_split(instancias, test_size=0.15, random_state=42)

    df_train = df[df['instancia_id'].isin(train_ids)]
    df_test = df[df['instancia_id'].isin(test_ids)]

    print(f"Instancias en train: {df_train['instancia_id'].nunique()}")
    print(f"Instancias en test: {df_test['instancia_id'].nunique()}")

p_inicial = ["", "¿"]
p_final = ["", ".", ",", "?"]
batch_size = 32

#%% Funciones y clases
# Funciones y clases
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_out, mask=None):
        # rnn_out: [batch, seq_len, hidden_dim]
        attn_weights = self.attn(rnn_out).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)  # [batch, seq_len]
        attended_output = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_dim]
        return attended_output, attn_weights
    
class TextRestorationGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_caps_tags, num_punt_ini_tags, num_punt_fin_tags, dropout=0.1):
        super().__init__()

        if bi: 
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            hidden_dim_2 = hidden_dim * 2
        else: 
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
            hidden_dim_2 = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim_2)
        if atencion:
            self.attention = Attention(hidden_dim_2)

        # Heads (ahora actúan directamente sobre rnn_out o attended_seq)
        self.cap_head = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_caps_tags)
        )
        self.punt_ini_head = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_punt_ini_tags)
        )
        self.punt_fin_head = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_punt_fin_tags)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, lengths):
        packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        rnn_out = self.dropout(rnn_out)
        rnn_out = self.norm(rnn_out)

        if atencion:
            attended, _ = self.attention(rnn_out)
            attended_seq = attended.unsqueeze(1).expand(-1, rnn_out.size(1), -1)
            xx = attended_seq
        else:
            xx = rnn_out

        return (
            self.cap_head(xx),
            self.punt_ini_head(xx),
            self.punt_fin_head(xx)
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

def preprocess_text(df):
    embedding_cols = [f"dim_red_{i}" for i in range(embed_dim)]
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
    
    #print(f"X_pad shape: {X_pad.shape}")

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

    cm = confusion_matrix(true_fin, pred_fin, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nada','?', '.', ','])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Puntuación Final")
    plt.show()

    cm = confusion_matrix(true_ini, pred_ini, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nada', '¿'])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Puntuación Inicial")
    plt.show()

    cm = confusion_matrix(true_cap, pred_cap, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Minúscula", "Capitalizado", "Mixto", "Mayúscula"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Capitalización")
    plt.show()

def train_model(model, dataloader, optimizer, device, loss_func, epochs=20):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for X_batch, y_cap_batch, y_ini_batch, y_fin_batch, lengths in dataloader:
            X_batch = X_batch.to(device)
            y_cap_batch = y_cap_batch.to(device)
            y_ini_batch = y_ini_batch.to(device)
            y_fin_batch = y_fin_batch.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            
            logits_cap, logits_ini, logits_fin = model(X_batch, lengths)
            
            mask = torch.arange(logits_cap.size(1), device=lengths.device)[None, :] < lengths[:, None]

            loss = loss_func(logits_cap, logits_ini, logits_fin, y_cap_batch, y_ini_batch, y_fin_batch, mask)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0 or epoch == epochs-1: 
            print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

#%% Cargar datos
# Cargar datos

# Cargar el CSV
if train_test:
    X, y_cap, y_punt_ini, y_punt_fin = preprocess_text(df_train)
else: 
    X, y_cap, y_punt_ini, y_punt_fin = preprocess_text(df)
    df_test = pd.read_csv(f"./tokens_etiquetados/tokens_etiquetados_dim256PCA.csv")
    print(f"Instancias en test: {df_test['instancia_id'].nunique()}")
    X_test, y_cap_test, y_punt_ini_test, y_punt_fin_test = preprocess_text(df_test)
    dataset_test = EmbeddingSequenceDataset(X_test, y_cap_test, y_punt_ini_test, y_punt_fin_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Model
model = TextRestorationGRU(
    embed_dim=embed_dim,
    hidden_dim=64,
    num_caps_tags=4,
    num_punt_ini_tags=len(p_inicial),  
    num_punt_fin_tags=len(p_final),
).to(device)

# Pesos por clase

weights_cap = get_class_weights(y_cap)
weights_fin = get_class_weights(y_punt_fin, num_classes=len(p_final))
weights_ini = get_class_weights(y_punt_ini, num_classes=len(p_inicial))

# Criterios de pérdida

criterion = nn.CrossEntropyLoss(ignore_index=-100)
criterion_cap = nn.CrossEntropyLoss(weight=weights_cap, ignore_index=-100)
criterion_ini = nn.CrossEntropyLoss(weight=weights_ini, ignore_index=-100)
criterion_fin = nn.CrossEntropyLoss(weight=weights_fin, ignore_index=-100)

# Optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def masked_cross_entropy(logits, targets, mask, criterion):
    logits_flat = logits.view(-1, logits.size(-1))          # [batch * seq_len, num_classes]
    targets_flat = targets.view(-1)                         # [batch * seq_len]
    mask_flat = mask.view(-1)                               # [batch * seq_len]
    
    logits_valid = logits_flat[mask_flat]
    targets_valid = targets_flat[mask_flat]
    
    return criterion(logits_valid, targets_valid)  # devuelve escalar

def loss_fn(logits_cap, logits_ini, logits_fin, y_cap, y_ini, y_fin, mask):
    loss_cap = masked_cross_entropy(logits_cap, y_cap, mask, criterion_cap)
    loss_ini = masked_cross_entropy(logits_ini, y_ini, mask, criterion_ini)
    loss_fin = masked_cross_entropy(logits_fin, y_fin, mask, criterion_fin)
    return loss_cap + loss_ini + loss_fin

def loss_norm(logits_cap, logits_ini, logits_fin, y_cap, y_ini, y_fin, mask):
    loss_cap = masked_cross_entropy(logits_cap, y_cap, mask, criterion)
    loss_ini = masked_cross_entropy(logits_ini, y_ini, mask, criterion)
    loss_fin = masked_cross_entropy(logits_fin, y_fin, mask, criterion)
    return loss_cap + loss_ini + loss_fin
#%% # Entrenamiento 1
# Entrenamiento 1

train_losses, val_losses = [], []
fractions = np.linspace(0.1, 1.0, 5)

print("Entrenamiento 1: con aumento de dataset")

for frac in fractions: 
    total = 50000
    parcial = int(total*frac)
    batch_size = int(parcial * 0.2) +1 
    print(parcial)
    dataset = EmbeddingSequenceDataset(X[:parcial], y_cap[:parcial], y_punt_ini[:parcial], y_punt_fin[:parcial])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    train_model(model, dataloader, optimizer, device, loss_norm, epochs=30)
    evaluate_model(model, dataloader_test, 'Unidireccional', device)

print("Entrenamiento 1 completado")

#%% # Entrenamiento 2
# Entrenamiento 2

batch_size = 1000
dataset = EmbeddingSequenceDataset(X, y_cap, y_punt_ini, y_punt_fin)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


print("Entrenamiento 2: Dataset Completo")
train_model(model, dataloader, optimizer, device, loss_norm, epochs=100)
evaluate_model(model, dataloader_test, 'Unidireccional', device)

print("Entrenamiento 2 completado")
torch.save(model.state_dict(), 'pesos2.pth')

#%% # Entrenamiento 3
# Entrenamiento 3

print("Entrenamiento 3: Weight")

train_model(model, dataloader, optimizer, device, loss_fn, epochs=20)
evaluate_model(model, dataloader_test, 'Unidireccional', device)

print("Entrenamiento 3 completado")

torch.save(model.state_dict(), 'pesos3.pth')

#%% # Entrenamiento 4
# Entrenamiento 4

print("Entrenamiento 4: Dataset Completo")

batch_size = 100
X, y_cap, y_punt_ini, y_punt_fin = preprocess_text(df[(df['instancia_id'] >= 30000) & (df['instancia_id'] <= 35000)])
dataset = EmbeddingSequenceDataset(X, y_cap, y_punt_ini, y_punt_fin)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


train_model(model, dataloader, optimizer, device, loss_norm, epochs=300)
evaluate_model(model, dataloader_test, 'Unidireccional', device)

print("Entrenamiento 4 completado")

torch.save(model.state_dict(), 'pesos4.pth')
#%% Evaluación del modelo

# Evaluación del modelo

X_test, y_cap_test, y_punt_ini_test, y_punt_fin_test = preprocess_text(df_test)
dataset_test = EmbeddingSequenceDataset(X_test, y_cap_test, y_punt_ini_test, y_punt_fin_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

evaluate_model(model, dataloader_test, 'Unidireccional', device)

# %% Evaluación single 
## Evaluación single 

def evaluate_single_instance(model, df, instancia_id): 
    # Filtrar solo una instancia
    df = df[df['instancia_id'] == instancia_id]
    if df.empty:
        print(f"No hay datos para instancia_id={instancia_id}")
        return

    # Preprocesamiento
    X, y_cap, y_punt_ini, y_punt_fin = preprocess_text(df)

    # Dataset con una sola instancia
    dataset = EmbeddingSequenceDataset(X, y_cap, y_punt_ini, y_punt_fin)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    
    true_cap, pred_cap = [], []
    true_ini, pred_ini = [], []
    true_fin, pred_fin = [], []

    with torch.no_grad():
        for X_batch, y_cap_batch, y_ini_batch, y_fin_batch, lengths in dataloader:
            X_batch = X_batch.to(next(model.parameters()).device)
            lengths = lengths.to(next(model.parameters()).device)

            logits_cap, logits_ini, logits_fin = model(X_batch, lengths)

            pred_cap_batch = logits_cap.argmax(dim=-1).cpu().numpy()
            pred_ini_batch = logits_ini.argmax(dim=-1).cpu().numpy()
            pred_fin_batch = logits_fin.argmax(dim=-1).cpu().numpy()

            y_cap_batch = y_cap_batch.cpu().numpy()
            y_ini_batch = y_ini_batch.cpu().numpy()
            y_fin_batch = y_fin_batch.cpu().numpy()

            for i, length in enumerate(lengths.cpu().numpy()):
                true_cap.extend(y_cap_batch[i][:length])
                pred_cap.extend(pred_cap_batch[i][:length])

                true_ini.extend(y_ini_batch[i][:length])
                pred_ini.extend(pred_ini_batch[i][:length])

                true_fin.extend(y_fin_batch[i][:length])
                pred_fin.extend(pred_fin_batch[i][:length])

    print(f"\n--- EVALUACIÓN DE LA INSTANCIA : {instancia_id} ---")

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


    # Matriz de confusión 

    cm = confusion_matrix(true_fin, pred_fin, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nada','?', '.', ','])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Puntuación Final")
    plt.show()

    cm = confusion_matrix(true_ini, pred_ini, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Nada', '¿'])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Puntuación Inicial")
    plt.show()

    cm = confusion_matrix(true_cap, pred_cap, labels=[0,1,2,3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Minúscula", "Capitalizado", "Mixto", "Mayúscula"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de confusión 1 - Capitalización")
    plt.show()

evaluate_single_instance(model, df, instancia_id=1)