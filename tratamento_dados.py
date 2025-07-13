import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Carrega o conjunto de dados
file_path = 'dados_enem_escolas.xlsx'
df = pd.read_excel(file_path)

# Cria uma cópia para modificações
df_processado_v2= df.copy()

# Colunas com valores ausentes
cols_com_na = ['PC_FORMACAO_DOCENTE', 'NU_TAXA_APROVACAO']

# Imputa valores ausentes com a mediana
for col in cols_com_na:
    mediana_val = df_processado_v2[col].median()
    df_processado_v2[col] = df_processado_v2[col].fillna(mediana_val)

print("Valores ausentes após imputação:")
print(df_processado_v2[cols_com_na].isnull().sum())

# Limita NU_TAXA_PERMANENCIA a 100 - tratamento de erro
df_processado_v2['NU_TAXA_PERMANENCIA'] = np.clip(df_processado_v2['NU_TAXA_PERMANENCIA'], a_min=None, a_max=100)

print("\nEstatísticas descritivas para NU_TAXA_PERMANENCIA após o corte:")
print(df_processado_v2['NU_TAXA_PERMANENCIA'].describe())     
  
features_categoricas_numericas = [
    'TP_DEPENDENCIA', 'TP_LOCALIZACAO', 'PORTE_ESCOLA', 	
    'IN_LABORATORIO_CIENCIAS',  		
    'IN_FORMACAO_ALTERNANCIA', 'TP_ATIVIDADE_COMPLEMENTAR',    
]

# Converter Sigla UF para string
df_processado_v2['SG_UF'] = df_processado_v2['SG_UF'].astype(str)

# Converter as outras features categóricas-numéricas para 'object' para que o get_dummies as trate corretamente
for col in features_categoricas_numericas:
    df_processado_v2[col] = df_processado_v2[col].astype(str) 

print("\nTipos de dados de features categóricas-numéricas antes do get_dummies:")
print(df_processado_v2[features_categoricas_numericas].dtypes)

colunas_para_one_hot_encoding = features_categoricas_numericas
df_processado_v2 = pd.get_dummies(df_processado_v2, columns=colunas_para_one_hot_encoding, drop_first=True)

# Estratégia para tratamento dos Outliers: IQR e Corte

def cap_outliers_iqr(df, nome_coluna):
    Q1 = df[nome_coluna].quantile(0.25)
    Q3 = df[nome_coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers_antes = df[(df[nome_coluna] < limite_inferior) | (df[nome_coluna] > limite_superior)].shape[0]
    df[nome_coluna] = np.clip(df[nome_coluna], limite_inferior, limite_superior)
    # outliers_depois = df[(df[nome_coluna] < limite_inferior) | (df[nome_coluna] > limite_superior)].shape[0]
    # print(f"Coluna: {nome_coluna}, Outliers antes: {outliers_antes}, Outliers depois: {outliers_depois}")
    if outliers_antes > 0:
        print(f"Coluna: {nome_coluna}, Outliers antes do corte IQR: {outliers_antes}")
    return df

colunas_para_tratar_outliers = [
    'NU_MEDIA_CN', 'NU_MEDIA_CH', 'NU_MEDIA_LP', 'NU_MEDIA_MT', 'NU_MEDIA_RED', # Variáveis alvo
    'NU_TAXA_PARTICIPACAO', 'PC_FORMACAO_DOCENTE',                              # Features numéricas restantes
    'NU_TAXA_PERMANENCIA', 'QT_COMP_ALUNO', 'QT_FUNCIONARIOS',
    'NU_TAXA_APROVACAO']

print("\n--- Cortando Outliers (IQR) em colunas numéricas e alvo remanescentes ---")
for col in colunas_para_tratar_outliers:
    if col in df_processado_v2.columns and df_processado_v2[col].dtype in ['int64', 'float64']:
        df_processado_v2 = cap_outliers_iqr(df_processado_v2, col)

# 5. Escalonamento de Features (Feature Scaling):  inclue apenas as colunas que são genuinamente numéricas

features_numericas_para_escalonar = colunas_para_tratar_outliers 

features_a_escalonar_realmente = [
    'NU_TAXA_PARTICIPACAO', 'PC_FORMACAO_DOCENTE',
    'NU_TAXA_PERMANENCIA', 'QT_COMP_ALUNO', 'QT_FUNCIONARIOS',
    'NU_TAXA_APROVACAO', 'CO_MUNICIPIO']

features_a_escalonar_realmente_existentes = [col for col in features_a_escalonar_realmente if col in df_processado_v2.columns and df_processado_v2[col].dtype in ['int64', 'float64']]

if features_a_escalonar_realmente_existentes:
    scaler = StandardScaler()
    df_processado_v2[features_a_escalonar_realmente_existentes] = scaler.fit_transform(df_processado_v2[features_a_escalonar_realmente_existentes])
    print("\n--- Features Numéricas Escalonadas ---")
    print(f"Colunas escalonadas: {features_a_escalonar_realmente_existentes}")
    print(df_processado_v2[features_a_escalonar_realmente_existentes].describe()) 
else:
    print("\nNenhuma feature numérica encontrada para escalonamento ou a lista está vazia.")

# Salvar o DataFrame processado
caminho_arquivo_processado_v2 = 'enem_escolas_processado_v2_pt.csv'
df_processado_v2.to_csv(caminho_arquivo_processado_v2, index=False)

print(f"\nDados processados (v2) com tratamento aprimorado de categóricas salvos em '{caminho_arquivo_processado_v2}'")
print("Dimensões finais do DataFrame processado:", df_processado_v2.shape)

