import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("portuguese"))
pontuacao = set(string.punctuation)

caminho_entrada = "propostas_treino.csv"  
caminho_saida = "propostas_treino_processados.csv"

def preprocessar_sentenca(sentenca):
    sentenca = sentenca.lower()
    tokens = word_tokenize(sentenca)
    tokens = [palavra for palavra in tokens if palavra not in stop_words and palavra not in pontuacao]
    return " ".join(tokens)

df = pd.read_csv(caminho_entrada)

if "Proposta" not in df.columns:
    raise ValueError("O dataset precisa conter uma coluna chamada 'Proposta'.")

if "Eixo" not in df.columns:
    print("Coluna 'Label' não encontrada. Criando rótulos automaticamente...")
    df["Label"] = 0  

df["Proposta"] = df["Proposta"].apply(preprocessar_sentenca)

df.to_csv(caminho_saida, index=False)

print(f"Dataset preprocessado salvo em: {caminho_saida}")
