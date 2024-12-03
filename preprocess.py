import os
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_pt = set(stopwords.words('portuguese'))

def extrair_texto_pdf(caminho_pdf):
    with pdfplumber.open(caminho_pdf) as pdf:
        texto = ''
        for pagina in pdf.pages:
            texto += pagina.extract_text() + '\n'
    return texto

# Função para limpar pontuação e remover stopwords
def limpar_texto(texto):
    # Remover pontuação e números
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    texto = re.sub(r'\d+', '', texto)     # Remove números
    texto = texto.lower()                 # Converte para letras minúsculas
    # Remover stopwords
    palavras = texto.split()
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stopwords_pt]
    return ' '.join(palavras_filtradas)

# Caminho da pasta com os PDFs
pasta_pdfs = "C:\\Users\\Cayo Cesar\\OneDrive - ufpi.edu.br\\Documentos\\GitHub\\sgm-aws\\PLN-recognition-government-plans\\dataset"

# Lista todos os arquivos PDF na pasta
arquivos_pdf = [os.path.join(pasta_pdfs, f) for f in os.listdir(pasta_pdfs) if f.endswith('.pdf')]

# Processar os PDFs e armazenar todas as propostas em uma lista
propostas = []
for arquivo in arquivos_pdf:
    print(f"Processando: {arquivo}")
    texto_extraido = extrair_texto_pdf(arquivo)
    texto_limpo = limpar_texto(texto_extraido)
    propostas.append(texto_limpo)

# Combinar todas as propostas em um único texto, separando por vírgula
texto_final = ', '.join(propostas)

# Salvar em um arquivo de texto único
caminho_saida = "propostas_processadas.csv"
with open(caminho_saida, 'w', encoding='utf-8') as f:
    f.write(texto_final)

print(f"Texto combinado salvo em: {caminho_saida}")
