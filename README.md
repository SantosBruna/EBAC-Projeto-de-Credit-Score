# 📊 Projeto: Análise e Processamento de Dados para Credit Score

## 📌 Sobre o Projeto

Projeto desenvolvido no **Módulo 17 – Profissão Cientista de Dados**, focado no pré-processamento e análise exploratória de dados para desenvolvimento de um modelo de Credit Score.

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

---

## 📁 Estrutura do Projeto

```
├── notebooks/
│   └── Profissao_Cientista_de_Dados_M17_Projeto.ipynb
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
* **Matplotlib** - Visualização de dados
* **Seaborn** - Visualizações estatísticas avançadas
* **Plotly Express** - Gráficos interativos
* **Scikit-learn** - Divisão de dados e pré-processamento
* **Imbalanced-learn (SMOTE)** - Balanceamento de classes
* **NumPy** - Operações numéricas
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

### **Etapa 1: Pré-processamento dos Dados**

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

### **Etapa 2: Análise Exploratória de Dados (EDA)**

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
- Associate's Degree ligeiramente menor que Bachelor's Degree, mas sem desbalanceamento significativo

**Marital Status:**
- ✅ **Balanceado**: Valores próximos entre solteiros e casados

**Home Ownership:**
- ⚠️ **Desbalanceado**: "Owned" possui o dobro de registros em relação a "Rented"

**Credit Score (variável alvo):**
- ⚠️ **Fortemente desbalanceado**: 
  - Predominância da classe Medium
  - Classe Low com poucos registros
  - Classe High com quantidade intermediária
- ⚠️ Atenção à possível multicolinearidade

#### 📊 Análise Bivariada

**Principais relações identificadas:**

1. **Age vs Credit Score:**
   - Média de 41 anos → High Score
   - Média de 31 anos → Medium Score
   - Média de 28 anos → Low Score
   - Correlação positiva fraca (0.17)

2. **Income vs Credit Score:**
   - ⭐ **Relação mais forte identificada**
   - Quanto maior o salário, maior o score
   - Faz sentido: maior capacidade de pagamento

3. **Gender vs Credit Score:**
   - Não foram encontrados homens com Low Score
   - Indica possível desbalanceamento que requer atenção

4. **Education vs Credit Score:**
   - Bachelor's Degree, Master's Degree e Doctorate não possuem os três tipos de score
   - Níveis mais altos de escolaridade associados a scores mais elevados

5. **Age vs Marital Status:**
   - Existe relação entre idade e estado civil
   - Pessoas mais velhas tendem a ser casadas

6. **Home Ownership vs Credit Score:**
   - Clientes com casa própria tendem a ter scores mais altos

**Perguntas adicionais exploradas:**
1. Qual variável tem mais forte associação com o credit score? → **Income**
2. O estado civil influencia no score? → **Correlação indireta via escolaridade**
3. Existe diferença no score entre homens e mulheres? → **Sim, ausência de homens no Low Score**

---

### **Etapa 3: Correlação, Codificação e Preparação dos Dados**

#### 📈 Análise de Correlação (Variáveis Numéricas)

**Antes do tratamento de categóricas:**
- Age vs Income: **0.63** (correlação positiva moderada-forte)
  - Faz sentido: mais idade → mais experiência → maiores salários

**Depois do tratamento de categóricas:**
- Age vs Income: **0.63** (mantida)
- Education vs Number of Children: **0.14**
- Age vs Credit Score: **0.17**

#### 🔄 Tratamento de Variáveis Categóricas

**Label Encoding (variáveis ordinais):**
- **Education**: Mapeamento ordinal de 1 a 5
  - High School Diploma: 1
  - Associate's Degree: 2
  - Bachelor's Degree: 3
  - Master's Degree: 4
  - Doctorate: 5

**One-Hot Encoding (variáveis nominais):**
- Gender → Gender_Female, Gender_Male
- Marital Status → Marital Status_Married, Marital Status_Single
- Home Ownership → Home Ownership_Owned, Home Ownership_Rented

**Variável Alvo:**
- Credit Score codificado com LabelEncoder (0, 1, 2)

#### ✂️ Divisão dos Dados

**Proporção:** 75% treino / 25% teste (random_state=42)

**Dimensões resultantes:**
- X_train: 75% dos dados
- X_test: 25% dos dados
- y_train: 75% dos rótulos
- y_test: 25% dos rótulos

#### ⚖️ Balanceamento de Classes (SMOTE)

**Problema identificado:**
- Classe predominante (Medium) com muito mais registros
- Classes minoritárias (Low e High) sub-representadas
- Risco de viés no modelo

**Solução aplicada:**
- Técnica SMOTE (Synthetic Minority Over-sampling Technique)
- Aplicada APENAS nos dados de treino
- Todas as classes balanceadas para mesma quantidade de amostras

**Resultado:**
- Distribuição uniforme das três classes no conjunto de treino
- Dados de teste mantidos sem alteração (representam distribuição real)

---

## 🔍 Principais Insights

### 💡 Sobre os Dados

1. **Qualidade dos dados**: Boa qualidade geral, com apenas 34 valores faltantes em 1 variável
2. **Ausência de outliers**: Nenhum outlier significativo detectado nas variáveis numéricas
3. **Desbalanceamentos identificados**:
   - Home Ownership (Owned 2x mais que Rented)
   - Credit Score (forte desbalanceamento na variável alvo)

### 💡 Sobre Relações entre Variáveis

1. **Income é a variável mais preditiva**: Forte relação com Credit Score
2. **Idade influencia moderadamente**: Através de correlação com salário (0.63)
3. **Educação importa**: Níveis mais altos associados a melhores scores
4. **Gênero apresenta viés**: Ausência de homens na classe Low Score
5. **Casa própria é indicador positivo**: Associação com scores mais altos

### 💡 Sobre Preparação para Modelagem

1. **Tratamento de categóricas bem-sucedido**: Encoding apropriado para cada tipo
2. **Balanceamento necessário e aplicado**: SMOTE resolveu o desbalanceamento crítico
3. **Dados prontos para ML**: Base limpa, processada e dividida adequadamente

---

## 📌 Conclusão

Este projeto completou com sucesso a **primeira etapa do desenvolvimento de um modelo de Credit Score**, realizando um pré-processamento robusto e uma análise exploratória completa dos dados.

### ✅ Conquistas do Projeto:

1. **Dados limpos e consistentes**: Tratamento eficaz de valores faltantes sem distorcer distribuições
2. **Insights valiosos**: Identificação de variáveis-chave (Income, Education, Age) para predição
3. **Codificação apropriada**: Tratamento correto de variáveis categóricas nominais e ordinais
4. **Balanceamento efetivo**: Aplicação de SMOTE para resolver desbalanceamento crítico
5. **Base preparada para ML**: Dados de treino e teste prontos para modelagem

### 🎯 Próximos Passos:

- Desenvolvimento e treinamento de modelos de classificação
- Avaliação de performance com métricas apropriadas (considerando classes balanceadas)

A base está **sólida e pronta** para a construção de um modelo preditivo de Credit Score confiável e eficaz.

---

## 👩‍💻 Autora

**Bruna S. R. Santos**

* 🔗 LinkedIn: [www.linkedin.com/in/brunasrsantos](https://www.linkedin.com/in/brunasrsantos)
* 📧 Email: brunasrsantos@gmail.com

---

## 📝 Licença

Este projeto está licenciado sob a **MIT License**.
