# Desafio-Data-Science-Indicium
Entrega do desafio para o processo seletivo Lighthouse.

Este projeto tem como objetivo desenvolver um modelo preditivo para estimar os preços de aluguel de imóveis temporários na cidade de Nova York. A estratégia envolve uma análise exploratória dos dados (EDA), pré-processamento, modelagem com Random Forest Regressor e avaliação do modelo com a métrica R².

## 1. Requisitos
Antes de executar o projeto, é necessário instalar as dependências listadas no arquivo requirements.txt. Certifique-se de ter o Python instalado.

## 2. Instalação
### 2.1 Clone este repositório:

git clone <https://github.com/davisramos/LH_CD_DAVISPECK.git>

cd <LH_CD_DAVISPECK>


### 2.2 Crie um ambiente virtual:

python -m venv venv

source venv/Scripts/activate  # No Mac use: venv\bin\activate


### 2.3 Instale as dependências:

pip install -r requirements.txt


## 3. Execução

Abra o Jupyter Notebook:

jupyter notebook

Navegue até o arquivo do projeto e execute-o.

## 4. Rodando o modelo

Para rodar o modelo, execute as seguintes linhas em um notebook no mesmo diretório do arquivo modelprice.pkl, sendo as informações contidas em novoap as desejadas para previsão:

import pickle

import pandas as pd

import numpy as np

with open('modelprices.pkl', 'rb') as file:
    mod = pickle.load(file)

novoap = {
   
    'bairro_group': 'Manhattan',
    
    'latitude': 40.75362,
    
    'longitude': -73.98377,
    
    'room_type': 'Entire home/apt'

}

df_novo_apartamento = pd.DataFrame([novoap])

df_novo_apartamento = pd.get_dummies(df_novo_apartamento, columns=['bairro_group', 'room_type'], drop_first=True)

colunas_faltantes = set(mod.feature_names_in_) - set(df_novo_apartamento.columns)

for coluna in colunas_faltantes:

    df_novo_apartamento[coluna] = 0

df_novo_apartamento = df_novo_apartamento[mod.feature_names_in_]

preco_previsto = mod.predict(df_novo_apartamento)

print(f'Sugestão de preço: ${preco_previsto[0]:.2f}')


