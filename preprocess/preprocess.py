import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string

# Baixar recursos necessários do NLTK
nltk.download("punkt")
nltk.download("stopwords")

# Caminho do arquivo CSV
caminho_csv = "teste/educacao_teste.csv"
caminho_saida = "teste/educacao_teste_sentencas_preprocessado.csv"

# Carregar o CSV
dados = pd.read_csv(caminho_csv)

# Certifique-se de identificar a coluna que contém o texto a ser processado
coluna_texto = "Educacao"  # Substitua pelo nome da coluna correta no seu arquivo

# Definir as stopwords e pontuação
stop_words = set(stopwords.words("portuguese"))
pontuacao = set(string.punctuation)

# Função para pré-processar e dividir em sentenças
def preprocessar_e_tokenizar_sentencas(texto):
    # Dividir o texto em sentenças
    sentencas = sent_tokenize(texto)
    sentencas_processadas = []
    for sentenca in sentencas:
        # Converter para letras minúsculas
        sentenca = sentenca.lower()
        # Tokenizar palavras
        tokens = word_tokenize(sentenca)
        # Remover stopwords e pontuação
        tokens = [palavra for palavra in tokens if palavra not in stop_words and palavra not in pontuacao]
        # Rejuntar tokens em uma sentença processada
        sentencas_processadas.append(" ".join(tokens))
    return sentencas_processadas

# Criar lista para armazenar as sentenças processadas
sentencas_final = []

# Processar cada linha da coluna de texto
for index, linha in dados.iterrows():
    texto = linha[coluna_texto]
    sentencas_processadas = preprocessar_e_tokenizar_sentencas(texto)
    # Adicionar as sentenças processadas no resultado final, substituindo as linhas originais
    for sentenca in sentencas_processadas:
        sentencas_final.append([sentenca])  # Cada sentença será uma nova linha no CSV

# Converter o resultado em um DataFrame
df_sentencas = pd.DataFrame(sentencas_final, columns=["Sentenca"])

# Salvar no CSV
df_sentencas.to_csv(caminho_saida, index=False)

print(f"Arquivo preprocessado e tokenizado por sentenças salvo em: {caminho_saida}")
