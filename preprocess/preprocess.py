import csv

def txt_para_csv(caminho_txt, caminho_csv, cabecalho="Proposta"):
    try:
        # Ler o conteúdo do arquivo .txt
        with open(caminho_txt, "r", encoding="utf-8") as arquivo_txt:
            linhas = arquivo_txt.readlines()
        
        # Processar as linhas do arquivo .txt
        propostas = [linha.strip() for linha in linhas if linha.strip()]  # Remove linhas vazias e espaços extras
        
        # Salvar as propostas no arquivo .csv
        with open(caminho_csv, "w", encoding="utf-8", newline="") as arquivo_csv:
            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow([cabecalho])  # Escrever o cabeçalho
            
            for proposta in propostas:
                escritor_csv.writerow([proposta])
        
        print(f"Arquivo CSV gerado com sucesso: {caminho_csv}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Converter os arquivos de treino para csv
txt_para_csv("treino/saude.txt", "treino/saude.csv", "Saude")
# txt_para_csv("preprocess/educacao.txt", "educacao.csv", "Educacao")
# txt_para_csv("preprocess/seguranca.txt", "seguranca.csv", "Seguranca")
# txt_para_csv("preprocess/meio_ambiente.txt", "meio_ambiente.csv", "Meio Ambiente")
# txt_para_csv("preprocess/cultura.txt", "cultura.csv", "Cultura")

# Converter os arquivos de teste para csv
#txt_para_csv("preprocess/saude_teste.txt", "saude_teste.csv", "Saude")
# txt_para_csv("preprocess/educacao_teste.txt", "educacao_teste.csv", "Educacao")
# txt_para_csv("preprocess/seguranca_teste.txt", "seguranca_teste.csv", "Seguranca")
# txt_para_csv("preprocess/meio_ambiente_teste.txt", "meio_ambiente_teste.csv", "Meio Ambiente")
# txt_para_csv("preprocess/cultura_teste.txt", "cultura_teste.csv", "Cultura")

