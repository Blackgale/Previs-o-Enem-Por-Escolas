import pandas as pd
import numpy as np
import joblib

#  Aplica as mesmas etapas de pré-processamento usadas no treinamento em um novo conjunto de dados.

def preparar_novos_dados(df_novo, colunas_do_treino, scaler_ajustado, colunas_para_escalonar):
    print("Iniciando pré-processamento dos novos dados...")
    df_processado = df_novo.copy()

    # 1. One-Hot Encoding das mesmas colunas categóricas
    features_categoricas = ['PORTE_ESCOLA', 'TP_LOCALIZACAO', 'TP_DEPENDENCIA']
    # Adiciona colunas faltantes para garantir consistência
    for col in features_categoricas:
        if col not in df_processado.columns:
            df_processado[col] = 0 # ou um valor padrão adequado

    df_processado = pd.get_dummies(df_processado, columns=features_categoricas, drop_first=True)

    # 2. Alinhar colunas: Garante que o novo DataFrame tenha exatamente as mesmas colunas
    #    do modelo treinado, preenchendo com 0 as que faltarem e mantendo a ordem.
    df_processado = df_processado.reindex(columns=colunas_do_treino, fill_value=0)

    # 3. Escalonamento: Usar o scaler JÁ AJUSTADO para transformar os dados
    df_processado[colunas_para_escalonar] = scaler_ajustado.transform(df_processado[colunas_para_escalonar])

    print("Pré-processamento dos novos dados concluído.")
    return df_processado

# 1. Carregar os artefatos salvos do treinamento
try:
    modelo = joblib.load('modelo_final.joblib')
    scaler = joblib.load('scaler.joblib')
    colunas_modelo = joblib.load('lista_de_colunas_modelo.joblib')
    colunas_escalonar = joblib.load('lista_de_colunas_para_escalonar.joblib')
    print("Artefatos de modelo carregados com sucesso.\n")
except FileNotFoundError:
    print("Erro: Arquivos de modelo (.joblib) não encontrados. Execute o script de treinamento primeiro.")
    exit()

# 2. Criar dados de exemplo para 3 novas escolas fictícias ou incluir dos dados_enem_escolas2014
dados_novas_escolas = [
    {
        'CO_ENTIDADE': 999901, 'SG_UF': 'SP', 'PC_FORMACAO_DOCENTE': 95.5,
        'NU_TAXA_APROVACAO': 98.2, 'NU_TAXA_PERMANENCIA': 99.1,
        'QT_SALAS_UTILIZADAS': 25, 'QT_FUNCIONARIOS': 60, 'PORTE_ESCOLA': 3,
        'TP_LOCALIZACAO': 1, 'TP_DEPENDENCIA': 4, 'QT_COMP_ALUNO': 1.5,
        'NU_TAXA_PARTICIPACAO': 85.0
    },
    {
        'CO_ENTIDADE': 999902, 'SG_UF': 'RJ', 'PC_FORMACAO_DOCENTE': 75.0,
        'NU_TAXA_APROVACAO': 88.0, 'NU_TAXA_PERMANENCIA': 92.5,
        'QT_SALAS_UTILIZADAS': 15, 'QT_FUNCIONARIOS': 35, 'PORTE_ESCOLA': 2,
        'TP_LOCALIZACAO': 1, 'TP_DEPENDENCIA': 2, 'QT_COMP_ALUNO': 3.1,
        'NU_TAXA_PARTICIPACAO': 70.0
    },
    {
        'CO_ENTIDADE': 999903, 'SG_UF': 'BA', 'PC_FORMACAO_DOCENTE': 60.1,
        'NU_TAXA_APROVACAO': 81.5, 'NU_TAXA_PERMANENCIA': 85.0,
        'QT_SALAS_UTILIZADAS': 10, 'QT_FUNCIONARIOS': 22, 'PORTE_ESCOLA': 1,
        'TP_LOCALIZACAO': 2, 'TP_DEPENDENCIA': 2, 'QT_COMP_ALUNO': 4.5,
        'NU_TAXA_PARTICIPACAO': 62.3
    }
]

df_novas_escolas = pd.DataFrame(dados_novas_escolas)

# 3. Prepara os novos dados usando a mesma pipeline de pré-processamento
dados_prontos_para_previsao = preparar_novos_dados(
    df_novo=df_novas_escolas,
    colunas_do_treino=colunas_modelo,
    scaler_ajustado=scaler,
    colunas_para_escalonar=colunas_escalonar
)

# 4. Fazer a previsão
previsoes = modelo.predict(dados_prontos_para_previsao)

# 5. Apresentar os resultados de forma clara
TARGETS = ['NU_MEDIA_CN', 'NU_MEDIA_CH', 'NU_MEDIA_LP', 'NU_MEDIA_MT', 'NU_MEDIA_RED']
df_previsoes = pd.DataFrame(previsoes, columns=TARGETS, index=df_novas_escolas['CO_ENTIDADE'])

print("\n===============================================")
print("      PREVISÃO DE NOTAS PARA NOVAS ESCOLAS     ")
print("===============================================")
print(df_previsoes.round(2))