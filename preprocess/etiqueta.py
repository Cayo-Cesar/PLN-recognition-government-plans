import pandas as pd

saude_df = pd.read_csv('treino/saude_treino.csv')
seguranca_df = pd.read_csv('treino/seguranca_treino.csv')
educacao_df = pd.read_csv('treino/educacao_treino.csv')

saude_df['Eixo'] = 'Saúde'
seguranca_df['Eixo'] = 'Segurança'
educacao_df['Eixo'] = 'Educação'

saude_df = saude_df[['Proposta', 'Eixo']]
seguranca_df = seguranca_df[['Proposta', 'Eixo']]
educacao_df = educacao_df[['Proposta', 'Eixo']]

df_completo = pd.concat([saude_df, seguranca_df, educacao_df], ignore_index=True)

print(df_completo.head())

df_completo.to_csv('propostas_treino.csv', index=False)

