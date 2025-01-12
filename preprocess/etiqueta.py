import pandas as pd

saude_df = pd.read_csv('dataset/saude.csv')
seguranca_df = pd.read_csv('dataset/seguranca.csv')
educacao_df = pd.read_csv('dataset/educacao.csv')
meio_ambiente_df = pd.read_csv('dataset/meio_ambiente.csv')
infraestrutura_df = pd.read_csv('dataset/infraestrutura.csv')

saude_df['Eixo'] = 'Saúde'
seguranca_df['Eixo'] = 'Segurança'
educacao_df['Eixo'] = 'Educação'
meio_ambiente_df['Eixo'] = 'Meio Ambiente'
infraestrutura_df['Eixo'] = 'Infraestrutura'

saude_df = saude_df[['Proposta', 'Eixo']]
seguranca_df = seguranca_df[['Proposta', 'Eixo']]
educacao_df = educacao_df[['Proposta', 'Eixo']]
meio_ambiente_df = meio_ambiente_df[['Proposta', 'Eixo']]
infraestrutura_df = infraestrutura_df[['Proposta', 'Eixo']]

df_completo = pd.concat([saude_df, seguranca_df, educacao_df, meio_ambiente_df, infraestrutura_df], ignore_index=True)

print(df_completo.head())

df_completo.to_csv('dataset/propostas.csv', index=False)

