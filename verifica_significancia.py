import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

# Ignorar avisos para uma saída mais limpa
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# PASSO 0: Carregar o DataFrame processado pronto (resultante do tratamento_dados.py)
file_path_processado = 'enem_escolas_processado_v2_pt.csv'
try:
    df_processado_v2 = pd.read_csv(file_path_processado)
    print(f"Carregando dados processados de {file_path_processado}...")
    print(f"Shape do DataFrame carregado: {df_processado_v2.shape}")
except FileNotFoundError:
    print(f"ERRO: Arquivo {file_path_processado} não encontrado. Certifique-se de que ele existe no caminho especificado.")
    exit()

# PASSO 1: Escolha UMA variável alvo para esta análise
# Mude para NU_MEDIA_CN, NU_MEDIA_CH, NU_MEDIA_MT e NU_MEDIA_RED para analisar as outras notas
nome_variavel_alvo_atual = 'NU_MEDIA_LP' 

if nome_variavel_alvo_atual not in df_processado_v2.columns:
    print(f"ERRO: Variável alvo '{nome_variavel_alvo_atual}' não encontrada no DataFrame.")
else:
    print(f"\n--- Iniciando análise de significância para: {nome_variavel_alvo_atual} ---")
    y = df_processado_v2[nome_variavel_alvo_atual].copy()

    # PASSO 2: Preparar X (features)
    nomes_todas_variaveis_alvo = ['NU_MEDIA_CN', 'NU_MEDIA_CH', 'NU_MEDIA_LP', 'NU_MEDIA_MT', 'NU_MEDIA_RED']
    
    # Lista de colunas que NÃO são features
    colunas_identificadoras_nao_features = ['SG_UF','CO_ENTIDADE'] 
    
    features_para_X_candidatas = [
        col for col in df_processado_v2.columns if \
        col not in nomes_todas_variaveis_alvo and \
        col not in colunas_identificadoras_nao_features
    ]
    X = df_processado_v2[features_para_X_candidatas].copy()

    # PASSO 3: Garantir que X e y sejam numéricos e sem NaNs/Infinitos
    print(f"\nShape inicial de X: {X.shape}, Shape inicial de y: {y.shape}")

    colunas_objeto_em_X = X.select_dtypes(include=['object']).columns
    if not colunas_objeto_em_X.empty:
        print(f"AVISO: Removendo colunas 'object' de X: {colunas_objeto_em_X.tolist()}")
        X = X.drop(columns=colunas_objeto_em_X)

    if y.isnull().any():
        print(f"Removendo {y.isnull().sum()} linhas com NaN em y ({nome_variavel_alvo_atual}).")
        linhas_validas_y = y.notnull()
        y = y[linhas_validas_y]
        X = X.loc[linhas_validas_y] 
        
    if X.isnull().any().any():
        print("Preenchendo NaNs restantes em X com a média da coluna...")
        for col_nan_X in X.columns[X.isnull().any()]:
            X.loc[:, col_nan_X] = X.loc[:, col_nan_X].fillna(X[col_nan_X].mean())
            
    if np.isinf(X.select_dtypes(include=np.number)).any().any(): 
        print("Substituindo valores infinitos em X por NaN...")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X.isnull().any().any():
            print("Re-preenchendo NaNs (originados de Inf) em X com a média da coluna...")
            for col_nan_X in X.columns[X.isnull().any()]:
                X.loc[:, col_nan_X] = X.loc[:, col_nan_X].fillna(X[col_nan_X].mean())

    if y.dtype == 'object':
        print(f"AVISO: y ({nome_variavel_alvo_atual}) ainda é do tipo object. Tentando converter para numérico.")
        y = pd.to_numeric(y, errors='coerce')
        if y.isnull().any():
            print(f"Removendo NaNs em y ({nome_variavel_alvo_atual}) após conversão para numérico.")
            linhas_validas_y_apos_conversao = y.notnull()
            y = y[linhas_validas_y_apos_conversao]
            X = X.loc[linhas_validas_y_apos_conversao]

    if X.empty or y.empty or X.shape[0] == 0 or X.shape[0] != y.shape[0]:
        print("ERRO: X ou y estão vazios, sem linhas ou com tamanhos incompatíveis após tratamento de NaNs/Inf. Verifique os dados.")
        print(f"Shape final de X: {X.shape}, Shape final de y: {y.shape}")
    else:
        print("\n--- Estado final de X e y antes da conversão final para float e sm.add_constant ---")
        print(f"Shape de X: {X.shape}")
        print(f"Shape de y: {y.shape}")
                
        print("\nConvertendo explicitamente todas as colunas de X para tipo numérico (float64)...")
        try:
            for col in X.columns:
                if X[col].dtype == 'bool':
                    X.loc[:, col] = X.loc[:, col].astype(int) 
            X = X.astype(float) 
            print("Conversão de X para float64 bem-sucedida.")
            
        except Exception as e_astype:
            print(f"ERRO durante a conversão explícita de X para float64: {e_astype}")
            print("Verifique as colunas que não puderam ser convertidas. Saindo.")
            exit()
        
        # PASSO 4: Adicionar uma constante a X
        X_com_constante_ols = sm.add_constant(X, prepend=True) 
        X_com_constante_ols = X_com_constante_ols.dropna(axis=1, how='all') # Remove colunas que são inteiramente NaN (ex: se X estava vazia)

        if X_com_constante_ols.empty or X_com_constante_ols.shape[1] <=1 and 'const' in X_com_constante_ols.columns : # Se só sobrou a constante ou ficou vazio
            print("ERRO: X_com_constante ficou vazio ou só tem a constante após limpeza. Verifique as features em X.")
        else:
            # PASSO 5: Ajustar o modelo OLS
            print(f"\nAjustando modelo OLS para a variável alvo: {nome_variavel_alvo_atual}...")
            try:
                modelo = sm.OLS(y, X_com_constante_ols)
                resultados_modelo = modelo.fit()

                # PASSO 6: Analisar o Resumo do Modelo
                print("\n--- Resumo do Modelo OLS ---")
                print(resultados_modelo.summary())

                # Código para imprimir interpretação de p-valores e features significativas...)                             
                p_valores = resultados_modelo.pvalues
                coeficientes = resultados_modelo.params
                features_significativas = p_valores[p_valores < 0.05]
                if not features_significativas.empty:
                    print(f"Features estatisticamente significativas (p < 0.05) para {nome_variavel_alvo_atual}:")
                    df_significativas = pd.DataFrame({
                        'Coeficiente': coeficientes[features_significativas.index],
                        'P-valor': features_significativas
                    })
                    print(df_significativas.sort_values(by='P-valor').to_string())
                else:
                    print(f"Nenhuma feature encontrada como estatisticamente significativa (p < 0.05) para {nome_variavel_alvo_atual}.")

                # --- CÁLCULO DO VIF (Variance Inflation Factor) ---
                
                if not X.empty:
                    print(f"\n--- Calculando VIF (Variance Inflation Factor) para as features de X (shape: {X.shape}) ---")
                    
                    # Adiciona uma constante a X para o cálculo do VIF.
                    
                    if 'const' in X.columns:
                        X_vif_input = X.copy()
                    else:
                        X_vif_input = sm.add_constant(X, prepend=True) # Adiciona 'const' no início

                    vif_data = pd.DataFrame()
                    
                    vif_data["feature"] = X.columns 
                    
                    vif_values = []
                    for i, feature_name in enumerate(X.columns):
                        # O índice da feature 'feature_name' em X_vif_input (que tem 'const' no início) é i + 1
                        # A constante está no índice 0.
                        try:
                            vif = variance_inflation_factor(X_vif_input.values, i + 1) # i+1 para pular a 'const'
                            vif_values.append(vif)
                        except Exception as e_vif:
                            print(f"Erro ao calcular VIF para {feature_name}: {e_vif}")
                            vif_values.append(np.nan)

                    vif_data["VIF"] = vif_values
                    
                    print("\nValores de VIF (ordenados do maior para o menor):")
                    print(vif_data.sort_values(by="VIF", ascending=False).to_string())

                    features_com_vif_alto = vif_data[vif_data["VIF"] > 10]
                    if not features_com_vif_alto.empty:
                        print("\nFeatures com VIF > 10 (sugestão de forte multicolinearidade):")
                        print(features_com_vif_alto.sort_values(by="VIF", ascending=False).to_string())
                else:
                    print("Matriz X está vazia. Não é possível calcular VIFs.")

            except Exception as e_ols:
                print(f"ERRO ao ajustar o modelo OLS ou gerar o resumo: {e_ols}")
                if 'X_com_constante_ols' in locals():
                    print("Verifique se X_com_constante_ols contém apenas dados numéricos finitos e não tem problemas de colinearidade perfeita.")
                    print("Shape de X_com_constante_ols:", X_com_constante_ols.shape)
                    print("NaNs em X_com_constante_ols (soma):", X_com_constante_ols.isnull().sum().sum())
                    print("Tipos em X_com_constante_ols:")
                    print(X_com_constante_ols.dtypes.to_string())