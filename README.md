# 📊 Projeto: Análise e Processamento de Dados para Credit Score

## 📌 Sobre o Projeto

Projeto desenvolvido no **Profissão Cientista de Dados**, focado no pré-processamento, análise exploratória e modelagem preditiva para desenvolvimento de um modelo de Credit Score.

O termo "credit score" se refere a uma pontuação numérica que representa a credibilidade de um indivíduo em termos de cumprimento de obrigações financeiras, como pagamento de empréstimos e cartões de crédito. Esta pontuação é calculada com base em diversas informações financeiras e de crédito do indivíduo.

O **objetivo principal** deste projeto é prever o risco de um indivíduo se tornar inadimplente com suas obrigações financeiras. O modelo avalia a probabilidade de não cumprimento de pagamentos, sendo fundamental para instituições financeiras na tomada de decisão sobre concessão de crédito.

---

## 🎯 Objetivos

* Realizar pré-processamento completo dos dados
* Aplicar análises univariadas e bivariadas
* Identificar e tratar outliers e dados faltantes
* Analisar correlações entre variáveis
* Tratar variáveis categóricas
* Realizar balanceamento de classes
* Preparar a base para desenvolvimento de modelo de Machine Learning
* Aplicar e avaliar o algoritmo Naive Bayes para classificação multiclasse
* Aplicar e avaliar o algoritmo de Árvore de Decisão para classificação multiclasse
* Comparar o desempenho dos modelos aplicados

---

## 📁 Estrutura do Projeto

```
├── notebooks/
│   ├── Profissao_Cientista_de_Dados_M17_Projeto.ipynb
│   ├── Profissao_Cientista_de_Dados_M20_Pratique.ipynb
│   └── Profissao_Cientista_de_Dados_M21_Pratique.ipynb
├── data/
│   └── CREDIT_SCORE_PROJETO_PARTE1.csv
├── outputs/
│   ├── X_train_balanced.csv
│   ├── y_train_balanced.csv
│   ├── X_test.csv
│   └── y_test.csv
└── README.md
```

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**
* **Pandas** - Manipulação e análise de dados
* **NumPy** - Operações numéricas
* **Matplotlib** - Visualização de dados
* **Seaborn** - Visualizações estatísticas avançadas
* **Plotly Express** - Gráficos interativos
* **Scikit-learn** - Divisão de dados, pré-processamento, modelagem e métricas de avaliação
* **Imbalanced-learn (SMOTE)** - Balanceamento de classes
* **Jupyter Notebook** - Ambiente de desenvolvimento

---

## 📊 Descrição dos Dados

O dataset contém informações de clientes bancários com as seguintes variáveis:

| Variável | Descrição |
|----------|-----------|
| **Age** | Idade dos clientes |
| **Income** | Salário mensal |
| **Gender** | Gênero (Masculino/Feminino) |
| **Education** | Nível de escolaridade (High School Diploma, Associate's Degree, Bachelor's Degree, Master's Degree, Doctorate) |
| **Marital Status** | Estado civil (Solteiro/Casado) |
| **Number of Children** | Quantidade de filhos |
| **Home Ownership** | Tipo de residência (Própria/Alugada) |
| **Credit Score** | Score de crédito (variável alvo: Low/Medium/High) |

---

## 📈 Análises Realizadas

### **Módulo 17 — Pré-processamento e Análise Exploratória**

#### ✅ Verificação e Ajuste de Tipos de Dados
- Conversão de variáveis categóricas para tipo `string`
- Tratamento da coluna `Income`: remoção de pontos de milhar e conversão de vírgula para ponto decimal
- Conversão de `Age` para tipo inteiro

#### ✅ Tratamento de Dados Faltantes
- **Problema identificado**: 34 registros com valor `NaN` na coluna `Age`
- **Estratégias testadas**:
  1. Imputação pela mediana geral (gerou pico artificial na distribuição)
  2. Imputação condicional por escolaridade (melhorou mas manteve pico)
  3. **Solução adotada**: Imputação condicional dupla por escolaridade e gênero
- **Resultado**: Distribuição preservada sem distorções significativas

#### ✅ Validação de Variáveis Categóricas
- Verificação de valores únicos em todas as colunas categóricas
- **Resultado**: Não foram encontrados erros de digitação ou valores inconsistentes

---

#### 📊 Análise Univariada - Variáveis Numéricas

**Variável Age:**
- Distribuição com alta amplitude
- Sem presença de outliers
- Média próxima a 30 anos

**Variável Income:**
- Alta variabilidade nos valores
- Sem outliers detectados
- Distribuição relativamente uniforme

**Variável Number of Children:**
- Aparente desbalanceamento
- Concentração em valores baixos (0-2 filhos)

#### 📊 Análise Univariada - Variáveis Categóricas

**Gender:**
- ✅ **Balanceado**: Quantidade similar entre masculino e feminino

**Education:**
- ✅ **Balanceado**: Distribuição equilibrada entre os níveis de escolaridade

**Marital Status:**
- ✅ **Balanceado**: Valores próximos entre solteiros e casados

**Home Ownership:**
- ⚠️ **Desbalanceado**: "Owned" possui o dobro de registros em relação a "Rented"

**Credit Score (variável alvo):**
- ⚠️ **Fortemente desbalanceado**: predominância da classe Medium, classe Low com poucos registros

#### 📊 Análise Bivariada

**Principais relações identificadas:**

1. **Age vs Credit Score**: correlação positiva fraca (0.17); scores mais altos associados a idades maiores
2. **Income vs Credit Score**: ⭐ relação mais forte — maior salário, maior score
3. **Gender vs Credit Score**: ausência de homens com Low Score, indicando possível viés
4. **Education vs Credit Score**: níveis mais altos de escolaridade associados a scores melhores
5. **Home Ownership vs Credit Score**: proprietários tendem a ter scores mais altos

---

#### 🔄 Tratamento de Variáveis Categóricas

**Label Encoding (variáveis ordinais):**
- **Education**: Mapeamento ordinal de 1 a 5 (High School → Doctorate)

**One-Hot Encoding (variáveis nominais):**
- Gender, Marital Status, Home Ownership

**Variável Alvo:**
- Credit Score codificado com LabelEncoder (0, 1, 2)

#### ✂️ Divisão e Balanceamento dos Dados

- **Proporção treino/teste**: 75% / 25% (random_state=42)
- **Balanceamento**: SMOTE aplicado apenas nos dados de treino para corrigir o desbalanceamento crítico das classes

---

### **Módulo 20 — Modelagem com Naive Bayes**

#### 🤖 Aplicação do Algoritmo

Nesta etapa foi aplicado o algoritmo **Gaussian Naive Bayes** (`GaussianNB`) às bases de treino e teste preparadas no Módulo 17. O Naive Bayes é um classificador probabilístico que assume independência entre as variáveis e seleciona a classe com maior probabilidade como previsão final — sendo adequado tanto para classificação binária quanto para **múltiplas classes**, como é o caso do Credit Score (Low, Medium, High).

#### 📏 Métricas de Avaliação

As métricas utilizadas foram **acurácia**, **recall** (com `average='macro'` para tratamento multiclasse) e **matriz de confusão** visualizada com Plotly.

**Desempenho na base de treino:**
- O modelo apresentou alto desempenho, com acurácia e recall em proporções similares
- A matriz de confusão revelou apenas três classificações incorretas, com a grande maioria dos registros avaliados corretamente

**Desempenho na base de teste:**
- O modelo atingiu **100% de acurácia e recall** na base de teste, medida que também se deve ao tamanho da base, que era pequena
- A matriz de confusão não apresentou nenhum registro classificado erroneamente
- O resultado é coerente com o excelente desempenho já observado no treino (~98%)

#### 💬 Conclusão da Etapa

O algoritmo Naive Bayes se adequou muito bem ao conjunto de dados, mesmo considerando a correlação moderada identificada entre as variáveis (Age vs Income: 0.63). O pressuposto de independência entre variáveis, central no Naive Bayes, não comprometeu a capacidade preditiva do modelo, que conseguiu prever o score de crédito com alta precisão através da abordagem probabilística.

---

### **Módulo 21 — Modelagem com Árvore de Decisão**

#### 🌳 Aplicação do Algoritmo

Nesta etapa foi aplicado o algoritmo **Decision Tree Classifier** (`DecisionTreeClassifier`) com critério de Gini e `random_state=0`, utilizando as bases de treino e teste preparadas no Módulo 17. A árvore de decisão é um modelo interpretável que particiona os dados de forma hierárquica com base nas features mais relevantes, sendo adequada para classificação multiclasse.

#### 🔍 Processo de Modelagem

O processo seguiu as seguintes etapas:
1. **Treinamento do modelo completo** com todas as features disponíveis
2. **Avaliação na base de treino** — acurácia, relatório de classificação e matriz de confusão
3. **Avaliação na base de teste** — mesmas métricas para comparação com o treino
4. **Visualização da árvore** para análise de profundidade e estrutura de decisão
5. **Análise de importância de features** para identificar as variáveis mais preditivas
6. **Retreinamento com as 2 principais features** e reavaliação dos resultados

#### 📏 Métricas de Avaliação

As métricas utilizadas foram **acurácia**, **relatório de classificação** (precisão, recall e F1-score por classe) e **matriz de confusão** visualizada com Seaborn.

**Desempenho na base de treino:**
- O modelo atingiu alta acurácia, classificando corretamente a grande maioria dos registros
- A matriz de confusão revelou poucos erros de classificação

**Desempenho na base de teste:**
- O modelo atingiu **95% de acurácia** na base de teste
- A matriz de confusão exibiu alguns erros de classificação, especialmente entre classes limítrofes
- A diferença entre treino e teste sugere leve overfitting, esperado em árvores sem poda

#### 🏆 Features Mais Importantes

A análise de importância de features identificou as duas variáveis com maior poder preditivo:
1. **Income** — principal determinante do Credit Score
2. **Home Ownership_Rented** — segunda feature mais relevante para a classificação

O modelo retreinado apenas com essas duas features manteve um desempenho competitivo, confirmando sua alta relevância para a predição do score.

#### ⚖️ Comparação: Árvore de Decisão vs. Naive Bayes

| Métrica | Naive Bayes | Árvore de Decisão |
|---------|-------------|-------------------|
| Acurácia (teste) | 100% | ~95% |
| Erros na matriz de confusão | Nenhum | Alguns |
| Considera relações entre variáveis | Não | Sim |
| Interpretabilidade visual | Baixa | Alta |

Embora o Naive Bayes tenha apresentado 100% de acurácia — resultado raro e provavelmente influenciado pelo tamanho reduzido da base de teste —, a Árvore de Decisão fornece resultados mais próximos da realidade ao capturar as relações entre variáveis. A forte relevância de `Income` e `Home Ownership_Rented` reforça a solidez do modelo de árvore como representação mais fiel do comportamento dos dados.

---

## 🔍 Principais Insights

1. **Income** é a variável mais preditiva do Credit Score (confirmado por ambos os modelos)
2. **Home Ownership** é a segunda feature mais relevante para a classificação
3. **Idade** influencia moderadamente, sobretudo via correlação com salário
4. **Educação** e **posse de imóvel** são indicadores positivos do score
5. **Gênero** apresentou viés (ausência de homens na classe Low Score)
6. A Árvore de Decisão oferece maior interpretabilidade e resultados mais realistas que o Naive Bayes neste dataset

---

## 👩‍💻 Autora

**Bruna S. R. Santos**

* 🔗 LinkedIn: [www.linkedin.com/in/brunasrsantos](https://www.linkedin.com/in/brunasrsantos)
* 📧 Email: brunasrsantos@gmail.com

---

## 📝 Licença

Este projeto está licenciado sob a **MIT License**.
