import matplotlib.pyplot as plt

# Dados de desempenho do modelo BERT
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
valores = [0.8607, 0.8498, 0.8373, 0.8299]

# Criar o gráfico de barras
plt.figure(figsize=(5, 6))
plt.bar(metricas, valores, color=['blue'])

# Personalizar o gráfico
plt.title('Desempenho do Modelo BERT', fontsize=16)
plt.xlabel('Métricas', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.ylim(0.7, 0.90)  # Ajustar os limites do eixo Y
plt.grid(True, linestyle='--', alpha=0.6)

# Mostrar o gráfico
plt.tight_layout()
plt.show()