# Instale as bibliotecas necessárias
# pip install transformers torch datasets scikit-learn pandas

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import torch

# 1. Carregar os dados do arquivo CSV
data = pd.read_csv("dataset/propostas_preprocessadas.csv")

# Verificar os dados
print(data.head())

# Garantir que não há valores nulos
data = data.dropna()

# Separar os textos e os rótulos
texts = data["Proposta"].tolist()
labels = data["Eixo"].astype("category").cat.codes.tolist()  # Converte os rótulos em índices numéricos

# Criar mapeamento entre rótulos e índices
label_mapping = dict(enumerate(data["Eixo"].astype("category").cat.categories))
print("Mapeamento de rótulos:", label_mapping)

# 2. Dividir os dados em treino e validação
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 3. Tokenização
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Converte os dados para o formato de tensor
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# 4. Carregar o modelo BERT
num_labels = len(label_mapping)  # Número de rótulos únicos
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# 5. Definir métricas de avaliação
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 6. Configurações do treinamento
training_args = TrainingArguments(
    output_dir="./results",  # Diretório para salvar o modelo
    num_train_epochs=3,  # Número de épocas
    per_device_train_batch_size=8,  # Tamanho do lote
    per_device_eval_batch_size=16,
    warmup_steps=500,  # Passos para ajustar a taxa de aprendizado
    weight_decay=0.01,  # Decaimento de peso
    logging_dir="./logs",  # Logs para TensorBoard
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# 7. Criar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 8. Treinamento
trainer.train()

# 9. Avaliação
results = trainer.evaluate()
print("Resultados da avaliação:", results)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 10. Inferência em novas propostas
def predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return [label_mapping[pred.item()] for pred in predictions]

new_proposals = ["Aumentar o investimento em educação básica.", "Criar programas de inclusão social."]
predictions = predict(new_proposals)
print("Classificação das novas propostas:", predictions)

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

