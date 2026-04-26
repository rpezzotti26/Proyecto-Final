
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargar los datasets
train_df = pd.read_csv("trainingData.csv")
test_df  = pd.read_csv("validationData.csv")

# Mostrar primeras filas
train_df.head()

# Tamaño del dataset
print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Distribución de clases
train_df["FLOOR"].value_counts().sort_index()




# Columnas a eliminar
cols_to_drop = [
    "LONGITUDE", "LATITUDE",
    "SPACEID", "RELATIVEPOSITION",
    "USERID", "PHONEID",
    "TIMESTAMP", "BUILDINGID"
]

train_df_clean = train_df.drop(columns=cols_to_drop)
test_df_clean  = test_df.drop(columns=cols_to_drop)

# Separar features y target
X_train_full = train_df_clean.drop(columns=["FLOOR"])
y_train_full = train_df_clean["FLOOR"]

X_test = test_df_clean.drop(columns=["FLOOR"])
y_test = test_df_clean["FLOOR"]

print(X_train_full.shape, y_train_full.shape)



X_train_full[X_train_full == 100] = -100
X_test[X_test == 100] = -100



def prepare_dataset(X, y, test_size=0.2, random_state=42):
    # División train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Normalización
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train, X_val, X_test_scaled,
        y_train.values, y_val.values, y_test.values
    )

X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(
    X_train_full, y_train_full
)

print(X_train.shape, X_val.shape, X_test.shape)



def create_loader(X, y, batch_size=32, shuffle=True):
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_loader(X_train, y_train)
val_loader   = create_loader(X_val, y_val, shuffle=False)
test_loader  = create_loader(X_test, y_test, shuffle=False)


class ANN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(520, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    def forward(self, x): return self.net(x)


class ANN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(520, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
    def forward(self, x): return self.net(x)


class ANN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(520, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)


class ANN4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(520, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x): return self.net(x)


class ANN5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(520, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x), y).item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

    elapsed = time.time() - start_time
    return train_losses, val_losses, elapsed

def evaluate_model(model, test_loader):
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            preds = torch.argmax(model(x), dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro")
    }

models = {
    "Arquitectura 1": ANN1(),
    "Arquitectura 2": ANN2(),
    "Arquitectura 3": ANN3(),
    "Arquitectura 4": ANN4(),
    "Arquitectura 5": ANN5()
}

results = {}

for name, model in models.items():
    print(f"\nEntrenando {name}")
    train_l, val_l, t = train_model(model, train_loader, val_loader)
    metrics = evaluate_model(model, test_loader)

    results[name] = {
        "train_loss": train_l,
        "val_loss": val_l,
        "time": t,
        **metrics
    }

    plt.plot(train_l, label="Train")
    plt.plot(val_l, label="Val")
    plt.title(name)
    plt.legend()
    plt.show()


summary = []

for name, res in results.items():
    summary.append({
        "Arquitectura": name,
        "Accuracy": round(res["accuracy"], 3),
        "Precision": round(res["precision"], 3),
        "Recall": round(res["recall"], 3),
        "F1-score": round(res["f1"], 3),
        "Tiempo (s)": round(res["time"], 1)
    })

summary_df = pd.DataFrame(summary)
print(summary_df)

# -------------------------------------------------
# PASO 7: Impacto del número de épocas
# Arquitectura seleccionada: Arquitectura 1
# -------------------------------------------------

selected_model_class = ANN1
epoch_values = [10, 20, 30, 40, 50]
epoch_results = {}

for epochs in epoch_values:
    print(f"\nEntrenando Arquitectura 1 con {epochs} épocas")

    model = selected_model_class()

    train_l, val_l, elapsed_time = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs
    )

    metrics = evaluate_model(model, test_loader)

    epoch_results[epochs] = {
        "train_loss": train_l,
        "val_loss": val_l,
        "time": elapsed_time,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    # Plot de pérdidas
    plt.figure()
    plt.plot(train_l, label="Train Loss")
    plt.plot(val_l, label="Validation Loss")
    plt.title(f"Arquitectura 1 - {epochs} épocas")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


print("\nResumen de resultados - Impacto del número de épocas (Arquitectura 1)\n")

for epochs, res in epoch_results.items():
    print(f"Épocas: {epochs}")
    print(f"  Accuracy : {res['accuracy']:.3f}")
    print(f"  Precision: {res['precision']:.3f}")
    print(f"  Recall   : {res['recall']:.3f}")
    print(f"  F1-score : {res['f1']:.3f}")
    print(f"  Tiempo (s): {res['time']:.2f}")
    print("-" * 40)