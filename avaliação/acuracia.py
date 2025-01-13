import matplotlib.pyplot as plt

# Dados de acurácia fornecidos
modelos = ['BERT', 'Random Forest', 'MLP']
acuracia = [0.8007, 0.76, 0.83]

# Criar o gráfico de linha
plt.figure(figsize=(8, 5))
plt.plot(modelos, acuracia, marker='o', linestyle='-', color='b', label='Acurácia')

# Personalizar o gráfico
plt.title('Comparação de Acurácia dos Modelos', fontsize=16)
plt.xlabel('Modelos', fontsize=14)
plt.ylabel('Acurácia', fontsize=14)
plt.ylim(0.7, 0.85)  # Ajustar os limites do eixo Y
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Mostrar o gráfico
plt.show()
