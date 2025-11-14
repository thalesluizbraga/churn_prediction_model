# usar regressao logistica para variaveis dependentes classes e regressao linear para variaveis dependentes para numeros
# regressao logistica tambem conhecida como log loss
# sempre calcular o intervalo de confiança para regressao linear e logistica para ver variaveis que sao significantes para o modelo

# metricas para arvore de decisao na classificaçao: gini e entropia


# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model



# %%

# leitura dos dados e primeiro plot 

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df['aprovado'] = df['nota'].apply(lambda x: 1 if x > 5 else 0)

plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('cerveja x aprovacao')
plt.xlabel('cervejas')
plt.ylabel('aprovado')

# %%

# modelos

#regressao logistica 
logloss = linear_model.LogisticRegression(penalty=None,
                                         fit_intercept=True )

logloss.fit(df[['cerveja']], df['aprovado'])

logloss_predict = logloss.predict(df[['cerveja']].drop_duplicates())
logloss_proba = logloss.predict_proba(df[['cerveja']].drop_duplicates()) [:,1]


# decision tree full
arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates()) [:,1]

# decision tree profundidade 2
arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates()) [:,1]

# naive bayes
nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates()) [:,1]

x = df['cerveja'].drop_duplicates().sort_values().values

plt.figure(figsize=(10, 6))

# Observações
plt.plot(df['cerveja'], df['aprovado'], 'o', color='royalblue', label='observação')

# Logistic Regression
plt.plot(x, logloss_predict, color='tomato', linestyle='-', label='logloss_predict')
plt.plot(x, logloss_proba, color='tomato', linestyle='--', label='logloss_proba')

# Árvore Completa
plt.plot(x, arvore_full_predict, color='green', linestyle='-', label='arvore_full_predict')
plt.plot(x, arvore_full_proba, color='green', linestyle='--', label='arvore_full_proba')

# Árvore max_depth=2
plt.plot(x, arvore_d2_predict, color='gray', linestyle='-', label='arvore_d2_predict')
plt.plot(x, arvore_d2_proba, color='gray', linestyle='--', label='arvore_d2_proba')

# Naive Bayes
plt.plot(x, nb_predict, color='blue', linestyle='-', label='nb_predict')
plt.plot(x, nb_proba, color='blue', linestyle='--', label='nb_proba')

# Linha de aprovação
plt.hlines(0.5, xmin=1, xmax=9, linestyle='--', colors='black')

plt.title('Cerveja x Aprovação')
plt.xlabel('Número de cervejas')
plt.ylabel('Probabilidade de aprovação')
plt.grid(True)
plt.legend()
plt.show()
