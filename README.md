# Previs-o-Enem-Por-Escolas

Projeto de IA: Previsao de Notas do ENEM por Escola

1. Pre-requisitos

Para executar este projeto, voc  precisar  ter o Python instalado em seu computador (vers o 3.8 ou superior).
Al m disso, as seguintes bibliotecas Python s o necess rias:
* pandas e openpyxl (para manipula  o de dados e leitura de arquivos Excel)
* numpy (para opera  es matem ticas)
* matplotlib e seaborn (para visualiza  o de dados)
* scikit-learn (para os modelos de machine learning e m tricas)
* joblib (para salvar os modelos treinados)

2. Crie uma pasta principal para o projeto e coloque o arquivo 'dados_enem_escolas.xlsx'

3. Instalacao e Configuracao do Ambiente

Criar um ambiente virtual para isolar as depend ncias do projeto.

1. Baixe os Arquivos e descompacte-os. Certifique-se de que os arquivos treinamento_modelos.py, 
   previsao_novas_escolas.py e dados_enem_escolas.xlsx est o na mesma pasta.

2. Abra um terminal ou prompt de comando na pasta do projeto e execute os seguintes comandos:
    Comando para criar o ambiente virtual: 
	 
	python -m venv enem_por_escolas  // sugestao de nome

    Ative o ambiente criado:

		No Windows: .\enem_por_escolas\Scripts\activate
		No macOS ou Linux: source enem_por_escolas/bin/activate

Apos a ativacao, voce vera (enem_por_escolas) no inicio do seu prompt do terminal.

3. Instale as Bibliotecas: Com o ambiente virtual ativado, instale todas as dependencias com o seguinte comando: 
	
	pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl

4. Como Executar o Projeto

Com o ambiente virtual ativado e dentro da pasta do projeto, execute o script principal com o seguinte comando:

	python treinamento_modelos.py

Ao executar o script imprimirá na tela todas as saidas propostas: 
Carregamento dos dados, 
Gráfico com Distribuição das variáveis alvos e Correlações (Fechar os gráficos para continuar),
Pre processamento e Engenharia de features,
Tabelas de Resultados.

Ao final, quatro novos arquivos terão sido criados:

	modelo_final.joblib
	scaler.joblib
	lista_de_colunas_modelo.joblib
	lista_de_colunas_para_escalonar.joblib

5.  Se desejar, após o treinamento ser concluído com sucesso e os arquivos .joblib serem criados, 
execute o segundo script para imprimir a previsão para três novas escolas ou substitua por 
outros exemplos no código:

python previsao_novas_escolas.py

=================================================================================================

PS: os codigos 'tratamento_dados.py' e 'verifica_significancia.py' foram usados para a limpeza
dos dados brutos e seleção das features mais significativas, o que resultou no dataset atual

Para testá-los: 1. execute 'tratamento_dados.py' primeiro e em seguida 'verifica_significancia' 
alterando o nome da variável alvo para obter os valores para cada nota.
