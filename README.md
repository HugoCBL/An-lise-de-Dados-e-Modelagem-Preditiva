# 🧠 Análise de Dados e Modelagem Preditiva com Machine Learning

Este repositório apresenta um projeto desenvolvido em um notebook Jupyter (Google Colaboratory), abordando três tarefas clássicas de Machine Learning. Utilizamos bibliotecas amplamente reconhecidas como `pandas`, `scikit-learn` e `torch` para aplicar e explorar diferentes algoritmos de **classificação** e **regressão**.

---

## 📁 Estrutura do Projeto

Todo o conteúdo está organizado em um único notebook `.ipynb`, dividido nas seguintes três seções principais:

### 🔹 Problema 1 – Classificador Bayesiano (Titanic)

Problema de **classificação binária** para prever a sobrevivência dos passageiros do Titanic.

- **Análise Exploratória:** Tratamento de valores ausentes (ex: 'Age', 'Cabin'), inspeção de distribuições e variáveis categóricas.
- **Pré-processamento:** Substituições e transformações (mediana, moda, encoding de categorias, ajuste em 'Fare').
- **Treinamento:** Uso do modelo `GaussianNB` com divisão de 70% para treino e 30% para teste.
- **Avaliação:** Métricas como Acurácia, Precisão, Recall, F1-score e Matriz de Confusão.
- **Discussão:** Limitações do Naive Bayes e suposições do modelo.

### 🔹 Problema 2 – Regressão Linear (Preços de Casas na Califórnia)

Problema de **regressão** para estimar preços de imóveis com base em variáveis socioeconômicas e geográficas.

- **Base de Dados:** `fetch_california_housing` (via `scikit-learn`).
- **Treinamento:** Modelo `LinearRegression` com divisão 80/20 (treino/teste).
- **Avaliação:** Erro Médio Absoluto (MAE) e Coeficiente de Determinação (R²).
- **Visualização:** Gráfico de dispersão entre valores reais e predições, com linha ideal para comparação.

### 🔹 Problema 3 – Perceptron (Reconhecimento de Dígitos - MNIST)

Problema de **classificação multiclasse** para identificar dígitos manuscritos usando uma rede neural simples.

- **Base de Dados:** MNIST (via `torchvision`).
- **Pré-processamento:** Normalização e separação dos conjuntos de treino, validação e teste.
- **Modelo:** Perceptron criado com `torch.nn`, utilizando `nn.Flatten()` e uma camada linear com 784 entradas e 10 saídas.
- **Treinamento:** 10 épocas com `CrossEntropyLoss` e otimizador `SGD`.
- **Avaliação:** Cálculo da acurácia e da perda no conjunto de teste. Exibição de predições usando `visualize_predictions()`.

---

## 🛠️ Tecnologias e Bibliotecas

- **Python 3.x**
- **Jupyter Notebook / Google Colab**
- **Pandas** – manipulação de dados
- **NumPy** – operações numéricas e vetoriais
- **Matplotlib** – visualização gráfica
- **Scikit-learn** – pré-processamento, modelos de regressão e classificação
- **PyTorch** – implementação de rede neural (Perceptron)
