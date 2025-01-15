import matplotlib.pyplot as plt

# Dados extraídos do texto fornecido
modelos = ['BERT', 'Random Forest', 'MLP']
acuracia = [0.8607, 0.76, 0.83]
precisao = [0.8498, 0.78, 0.84]
recall = [0.8373, 0.76, 0.84]
f1_score = [0.8299, 0.76, 0.83]

# Preparar os dados para o gráfico
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
valores = [acuracia, precisao, recall, f1_score]

# Criar o gráfico de linha
plt.figure(figsize=(10, 6))
for i, metrica in enumerate(valores):
    plt.plot(modelos, metrica, marker='o', label=metricas[i])

# Personalizar o gráfico
plt.title('Comparação de Desempenho dos Modelos', fontsize=16)
plt.xlabel('Modelos', fontsize=14)
plt.ylabel('Valores das Métricas', fontsize=14)
plt.ylim(0.7, 0.9)  # Ajustar os limites do eixo Y
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Métricas", fontsize=12, title_fontsize=12)
plt.tight_layout()

# Mostrar o gráfico
plt.show()
