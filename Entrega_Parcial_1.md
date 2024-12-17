# Relatório da Entrega Parcial 1

## 1. Tarefa a ser Investigada
A tarefa proposta para este projeto é **classificar propostas de governo do estado do Piaui** em diferentes categorias (áreas temáticas) utilizando **técnicas de Processamento de Linguagem Natural (PLN)** e aprendizado de máquina.

### Objetivos:
- Realizar a classificação das propostas (textos) com modelos de aprendizado supervisionado.
- Avaliar o desempenho de modelos tradicionais, como **MLP (Multilayer Perceptron)** e **Random Forest**, além de modelos baseados em **Transformers**, como **BERT**.

A tarefa tem como finalidade investigar a eficácia de diferentes abordagens na classificação multiclasse das propostas governamentais.

---

## 2. Corpus Utilizado
### Descrição dos Dados
A base de dados utilizada consiste em **propostas de governo preprocessadas**, representadas em um arquivo **CSV**. Cada linha do conjunto de dados possui duas informações principais:
- **Proposta**: Texto da proposta (após etapa de limpeza, remoção de stopwords e tokenização).
- **Eixo**: Classe da proposta, indicando a qual eixo temático pertence (ex: Saúde, Educação, Segurança, etc.).

### Exemplo de Estrutura do Corpus
| Proposta                                                            | Eixo        |
|---------------------------------------------------------------------|-------------|
| melhorar acesso saúde básica hospitais,                             | Saúde       |
| construir novas escolas zonas rurais,                               | Educação    |
| investir em estradas transportes,                                   | Segurança   |
| aumentar quadro médico emergências,                                 | Saúde       |
| reformar bibliotecas ensino fundamental,                            | Educação    |
| desenvolver programa trocas lâmpadas fluorescentes led residências, |Meio Ambiente|

### Estatísticas Quantitativas:
- **Total de Propostas**: 931 Propostas.
- **Número de Classes**: 4 (Saúde, Educação, Segurança, Meio Ambiente).
- **Distribuição das Classes**:
    - Saúde: 218 Propostas.
    - Educação: 257 Propostas.
    - Segurança: 228 Propostas
    - Meio Ambiente: 231 Propostas

Os textos passaram por um preprocessamento inicial, incluindo:
- Remoção de stopwords.
- Tokenização.
- Normalização (letras minúsculas, remoção de pontuação).
---

## 3. Situação Atual do Trabalho
### O que Já Foi Feito:
1. **Preprocessamento do Corpus**:
   - DataMining das Propostas.
   - Remoção de stopwords e pontuação.
   - Tokenização e normalização dos textos.
   - Geração do arquivo CSV preprocessado com colunas `Proposta` e `Eixo`.
2. **Definição dos Modelos**:
   - **Random Forest**: Modelo implementado usando **TF-IDF** como representação vetorial do texto.
   - **MLP**: Rede neural implementada usando embeddings de palavras como entrada.
   - **BERT**: Modelo baseado em Transformers para aprendizado profundo, utilizando embeddings contextuais e classificação.
3. **Validação Inicial**:
   - Modelos foram testados em um subconjunto dos dados para validar a pipeline.
   - **Métricas Iniciais**: Acurácia e relatórios de classificação foram obtidos para Random Forest.

---

## 4. O que Falta Fazer:
1. Adicionar mais propostas ao dataset 
2. Implementar e treinar um modelo baseado em **MLP** e **BERT** para fazer a comparação entre os 3 modelos.
3. Ajustar hiperparâmetros dos modelos (Random Forest) para melhorar o desempenho.
4. Realizar análise comparativa final das métricas de classificação, como **acurácia**, **precision**, **recall** e **F1-score**.
5. Geração de gráficos e visualização de resultados, como:
   - Matriz de confusão.
   - Importância das features nos modelos.
---

