import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

caminho_csv = "dataset/propostas_preprocessadas.csv"
df = pd.read_csv(caminho_csv)

if "Proposta" not in df.columns or "Eixo" not in df.columns:
    raise ValueError("O dataset precisa conter as colunas 'Proposta' e 'Eixo'.")

textos = df["Proposta"].values
labels = df["Eixo"].values

X_train, X_val, y_train, y_val = train_test_split(textos, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

print("Ajustando hiperparâmetros com Grid Search...")
grid_search.fit(X_train_tfidf, y_train)

print("Melhores Hiperparâmetros:", grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_val_tfidf)

print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred))

conf_matrix = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

importances = best_rf.feature_importances_
feature_names = vectorizer.get_feature_names_out()

sorted_indices = importances.argsort()[::-1][:10]
plt.figure(figsize=(10, 6))
plt.barh([feature_names[i] for i in sorted_indices], importances[sorted_indices])
plt.gca().invert_yaxis()
plt.title("Top 10 Palavras mais Importantes")
plt.xlabel("Importância")
plt.show()
