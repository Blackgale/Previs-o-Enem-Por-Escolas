# --- Importação de Bibliotecas ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.inspection import permutation_importance

# Ignorar alertas para uma saída mais limpa
warnings.filterwarnings("ignore", category=FutureWarning)

# Funções e classes do Scikit-learn para modelagem e avaliação
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configurações visuais padrão para os gráficos
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# FUNÇÕES AUXILIARES

def carregar_dados(caminho_arquivo):
    """Carrega os dados_enem_escolas.xls - deve estar no mesmo diretorio"""
    print("--- 1. Carregando Dados ---")
    try:
        return pd.read_excel(caminho_arquivo)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_arquivo}' não encontrado.")
        return None


def tratar_outliers_iqr(df: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """
    Identifica e trata outliers em uma coluna usando o método IQR.
    """
    print(f"\nTratando outliers para a coluna: '{coluna}'")
    
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Identificar outliers
    outliers_abaixo = df[df[coluna] < limite_inferior][coluna]
    outliers_acima = df[df[coluna] > limite_superior][coluna]
    
    # Tratar outliers usando np.clip para 'aparar' os valores nos limites
    df[coluna] = np.clip(df[coluna], limite_inferior, limite_superior)
     
    return df

def pre_processar(df):
    """Aplica todo o pré-processamento nos dados."""
    print("\n--- 2. Pré-processamento e Engenharia de Features ---")
    df_processado = df.copy()

    cols_com_na = ['PC_FORMACAO_DOCENTE', 'NU_TAXA_APROVACAO']
    for col in cols_com_na:
        df_processado[col].fillna(df_processado[col].median(), inplace=True)

    df_processado['NU_TAXA_PERMANENCIA'] = np.clip(df_processado['NU_TAXA_PERMANENCIA'], a_min=None, a_max=100)

    features_categoricas = ['PORTE_ESCOLA', 'TP_LOCALIZACAO', 'TP_DEPENDENCIA']
    df_processado = pd.get_dummies(df_processado, columns=features_categoricas, drop_first=True)
    
    print("Pré-processamento concluído.")
    return df_processado

def avaliar_modelo_multi_saida(model_name, y_true, y_pred, targets):
    """Imprime e retorna métricas detalhadas para a tabela final."""
    print(f"\n--- Análise Individual no Teste: {model_name} ---")
    resultados = {}
    for i, target in enumerate(targets):
        r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i]))
        print(f"  > {target}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        resultados[target] = {'R2': r2, 'RMSE': rmse}

    r2_medio = r2_score(y_true, y_pred, multioutput='uniform_average')
    print(f"\n  >> MÉDIA GERAL (R² no Teste): {r2_medio:.4f}")
    return resultados

def make_multi_output_scorer(target_index):
    """Cria uma função de scoring para a análise de importância do MLP."""
    def scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        y_pred_target = y_pred[:, target_index]
        return r2_score(y, y_pred_target)
    return scorer

print("Funções auxiliares definidas.")

caminho_arquivo = 'dados_enem_escolas.xlsx' # certifique-se que o arquivo esteja na mesma pasta
df_raw = carregar_dados(caminho_arquivo)

if df_raw is not None:
    # --- ANÁLISE EXPLORATÓRIA INICIAL ---
    TARGETS = ['NU_MEDIA_CN', 'NU_MEDIA_CH', 'NU_MEDIA_LP', 'NU_MEDIA_MT', 'NU_MEDIA_RED']

    print("\nGerando gráficos de distribuição das notas alvo...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(TARGETS):
        sns.histplot(df_raw[col].dropna(), kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Distribuição da nota de {col}')
    fig.tight_layout()
    if len(TARGETS) < len(axes):
        axes[-1].set_visible(False)
    plt.show()

    print("\nGerando gráficos de correlação entre features e cada nota alvo...")
    corr_matrix = df_raw.select_dtypes(include=np.number).corr()
    for target in TARGETS:
        corr_target = corr_matrix[target].drop(TARGETS).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=corr_target.values, y=corr_target.index)
        plt.title(f'Correlação das Features com a nota de {target}')
        plt.xlabel('Coeficiente de Correlação')
        plt.tight_layout()
        plt.show()

# Chama a função de pré-processamento
df_proc = pre_processar(df_raw)

# Definição de Features (X) e Alvos (y)
FEATURES_REMOVER = ['CO_ENTIDADE', 'SG_UF', 'MEDIA_TOTAL'] + TARGETS
features = [col for col in df_proc.columns if col not in [t for t in FEATURES_REMOVER if t in df_proc.columns]]
X = df_proc[features]
y = df_proc[TARGETS]

# Divisão em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalonamento com correção do FutureWarning
features_numericas_escalonar = X.select_dtypes(include=np.number).columns.tolist()
X_train, X_test = X_train.copy(), X_test.copy()
for col in features_numericas_escalonar:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('float64')
        X_test[col] = X_test[col].astype('float64')
scaler = StandardScaler()
scaler.fit(X_train[features_numericas_escalonar])
X_train.loc[:, features_numericas_escalonar] = scaler.transform(X_train[features_numericas_escalonar])
X_test.loc[:, features_numericas_escalonar] = scaler.transform(X_test[features_numericas_escalonar])

print("\n--- Dados prontos para a modelagem! ---")

# Validação Cruzada para Performance de Baseline

# Configuração dos modelos
modelos_config = {
    'Linear Regression': {'model': LinearRegression(), 'params': {}},
    'Random Forest': {'model': RandomForestRegressor(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [20]}},
    'MLP Regressor': {'model': MLPRegressor(random_state=42, max_iter=1000, early_stopping=True), 'params': {'hidden_layer_sizes': [(50, 50)], 'alpha': [0.001]}}
}

# Validação Cruzada
print("\n--- Validação Cruzada para Performance de Baseline ---")
for name, config in modelos_config.items():
    cv_scores = cross_val_score(config['model'], X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"  > {name} (Padrão) | R² Médio (CV): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Otimização com GridSearchCV, treinamento final e avaliação individual
print("\n--- Otimização, Treinamento e Avaliação Individual ---")
best_estimators = {}
resultados_finais = {}
for name, config in modelos_config.items():
    print(f"\n>>> Processando modelo: {name}...")
    if not config['params']:
        model = config['model']
        model.fit(X_train, y_train)
        best_estimators[name] = model
    else:
        grid_search = GridSearchCV(estimator=config['model'], param_grid=config['params'], cv=3, n_jobs=-1, scoring='r2', verbose=0)
        grid_search.fit(X_train, y_train)
        best_estimators[name] = grid_search.best_estimator_

    # Avaliação individual logo após o treino
    y_pred = best_estimators[name].predict(X_test)
    resultados_finais[name] = avaliar_modelo_multi_saida(name, y_test, y_pred, TARGETS)

# Tabela Comparativa Final
print("\n--- Tabela Comparativa de Performance Final ---")
lista_dfs_resultados = []
for model_name, result_data in resultados_finais.items():
    for target_name, metrics in result_data.items():
        lista_dfs_resultados.append({'Modelo': model_name, 'Área': target_name, 'R2': metrics['R2'], 'RMSE': metrics['RMSE']})
df_resultados = pd.DataFrame(lista_dfs_resultados)
df_comparativo = df_resultados.pivot_table(index='Área', columns='Modelo', values=['R2', 'RMSE'])
df_comparativo = df_comparativo.swaplevel(0, 1, axis=1).sort_index(axis=1)
print(df_comparativo.round(4))

# Análise de Importância das Features
print("\n--- Análise de Importância das Features ---")
for name, model in best_estimators.items():
    if name == 'Linear Regression':
        print("\n--- Tabela de Coeficientes para Regressão Linear ---")
        df_coefs = pd.DataFrame(model.coef_.T, index=X.columns, columns=TARGETS)
        print("Features com maior impacto para cada Nota Média por área:")
        print(df_coefs.reindex(df_coefs.NU_MEDIA_MT.abs().sort_values(ascending=False).index).head(10))

    elif name == 'Random Forest':
        print("\n--- Tabela e Gráfico de Importância para Random Forest ---")
        df_importances_rf = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
        print(df_importances_rf.head(10).round(4))
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df_importances_rf.head(15))
        plt.title('Top 15 Features Mais Importantes (Random Forest)')
        plt.tight_layout()
        plt.show()

    print("\nAnálise concluída com sucesso!")

print("\n--- 8. Salvando os artefatos do modelo para uso futuro ---")

# Salvando o melhor modelo (Random Forest)
modelo_final_para_salvar = best_estimators['Random Forest']

# Usando joblib para salvar os objetos

import joblib
joblib.dump(modelo_final_para_salvar, 'modelo_final.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(features, 'lista_de_colunas_modelo.joblib')
joblib.dump(features_numericas_escalonar, 'lista_de_colunas_para_escalonar.joblib')

print("\nModelo, scaler e listas de colunas foram salvos com sucesso!")
print("Análise concluída.")
