import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
caminho_csv = "dataset/propostas_preprocessadas.csv"
df = pd.read_csv(caminho_csv)

# Verificar colunas
if "Proposta" not in df.columns or "Eixo" not in df.columns:
    raise ValueError("O dataset precisa conter as colunas 'Proposta' e 'Eixo'.")

# Dividir os dados
textos = df["Proposta"].values
labels = df["Eixo"].values

# Codificar os rótulos (Eixos) para valores numéricos
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Dividir os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(textos, encoded_labels, test_size=0.2, random_state=42)

# Representação TF-IDF para os textos
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer.transform(X_val).toarray()

# One-hot encoding para os rótulos
y_train_one_hot = to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_val_one_hot = to_categorical(y_val, num_classes=len(label_encoder.classes_))

# Criar o modelo MLP
model = Sequential([
    Dense(512, input_dim=X_train_tfidf.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Uma saída para cada classe
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    X_train_tfidf, y_train_one_hot,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_tfidf, y_val_one_hot)
)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_val_tfidf, y_val_one_hot)
print(f"\nAcurácia no conjunto de validação: {accuracy:.4f}")

# Fazer previsões
y_pred = model.predict(X_val_tfidf)
y_pred_classes = y_pred.argmax(axis=1)

# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()
