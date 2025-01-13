import matplotlib.pyplot as plt

# Dados de desempenho do modelo MLP
metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
valores = [0.83, 0.82, 0.81, 0.82]

# Criar o gráfico de barras
plt.figure(figsize=(5, 6))
plt.bar(metricas, valores, color=['blue'])

# Personalizar o gráfico
plt.title('Desempenho do Modelo MLP', fontsize=16)
plt.xlabel('Métricas', fontsize=14)
plt.ylabel('Valores', fontsize=14)
plt.ylim(0.8, 0.85)  # Ajustar os limites do eixo Y
plt.grid(True, linestyle='--', alpha=0.6)

# Mostrar o gráfico
plt.tight_layout()
plt.show()