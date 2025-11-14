# %%
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import linear_model

import matplotlib.pyplot as plt

# %%

# ETL
path = '../data/dados_comunidade_respostas.csv'
df = pd.read_csv(path)

# Convert Sim/Não to 1/0 for ALL object columns
df = df.replace({"Sim": 1, "Não": 0, "Yes": 1, "No": 0})

# Handle categorical variables with dummy encoding
dummy_vars = [
    'Como conheceu o Téo Me Why?',    
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]


#%%

df_dummies = pd.get_dummies(df[dummy_vars], drop_first=True)
var_nao_dummies = []

for col in df.columns:
    if col not in dummy_vars and col != 'Você se considera uma pessoa feliz?':
        var_nao_dummies.append(col) 

df_numeric = df[var_nao_dummies].copy()

for col in df_numeric.select_dtypes(include=['object']).columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

df_analise = pd.concat([df_numeric, df_dummies], axis=1)

df_analise['pessoa_feliz'] = df['Você se considera uma pessoa feliz?'].copy()

#%%

# modelos - decision tree
features = df_analise.columns[1:-1].to_list() # pegar todas as colunas menos a primeira e a ultima que a variavel dependent
X = df_analise[features]
y = df_analise['pessoa_feliz'] 

# decision tree
# definicao do modelo
decision_tree = tree.DecisionTreeClassifier(random_state=42,
                                            min_samples_leaf=5) # min_samples_leaf significa que cada no da arvore tem que ter pelo menos 5 amostras
decision_tree.fit(X,y)
decision_tree_predict = decision_tree.predict(X)

# naive bayes
naive = naive_bayes.GaussianNB()
naive.fit(X,y)
naive_predict = naive.predict(X)

#%% log loss
logloss = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
logloss.fit(X,y)
logloss_predict = logloss.predict(X)


#%%
 
 # apresentacao do resultado do modelo no df
# Juntando o predict no df original paa comparaçao - naive bayes e decision tree tudo no mesmo df
df_predict = df_analise[['pessoa_feliz']]

df_predict['predict_arvore'] = decision_tree_predict
df_predict['proba_arvore'] = decision_tree.predict_proba(X) [:,1]


df_predict['predict_naive'] = naive_predict
df_predict['proba_naive'] = naive.predict_proba(X) [:,1]

df_predict['predict_logloss'] = logloss_predict 
df_predict['proba_logloss'] = logloss.predict_proba(X)[:,1] 

#%% 

# metricas

## confusion matrix


# linhas sao os valores verdadeiros, observados
# colunas sao os valores esta dizendo/calculando   
# A intersecao das colunas 1x1 e 0x0 e o que acertei e o oposto disso é o que errei
pd.crosstab(df_predict['pessoa_feliz'], df_predict['predict_arvore'])


# metricas do modelo - decision tree
accuracy_decision_tree = metrics.accuracy_score(df_predict['pessoa_feliz'], df_predict['predict_arvore'] )
precision_decision_tree = metrics.precision_score(df_predict['pessoa_feliz'], df_predict['predict_arvore'])
recall_decision_tree = metrics.recall_score(df_predict['pessoa_feliz'], df_predict['predict_arvore'])
auc_decision_tree = metrics.roc_curve(df_predict['pessoa_feliz'], df_predict['proba_arvore'])
auc_score_decision_tree = metrics.roc_auc_score(df_predict['pessoa_feliz'], df_predict['proba_arvore'])


# naive bayes
accuracy_naive_bayes = metrics.accuracy_score(df_predict['pessoa_feliz'], df_predict['predict_naive'] )
precision_naive_bayes = metrics.precision_score(df_predict['pessoa_feliz'], df_predict['predict_naive'])
recall_naive_bayes = metrics.recall_score(df_predict['pessoa_feliz'], df_predict['predict_naive'])
auc_naive_bayes = metrics.roc_curve(df_predict['pessoa_feliz'], df_predict['proba_naive'])
auc_score_naive_bayes = metrics.roc_auc_score(df_predict['pessoa_feliz'], df_predict['proba_naive'])

# logloss

accuracy_logloss = metrics.accuracy_score(df_predict['pessoa_feliz'], df_predict['predict_logloss'] )
precision_logloss = metrics.precision_score(df_predict['pessoa_feliz'], df_predict['predict_logloss'])
recall_logloss = metrics.recall_score(df_predict['pessoa_feliz'], df_predict['predict_logloss'])
auc_logloss = metrics.roc_curve(df_predict['pessoa_feliz'], df_predict['proba_logloss'])
auc_score_logloss = metrics.roc_auc_score(df_predict['pessoa_feliz'], df_predict['proba_logloss'])




# acuracia -> quanto o modelo acerta em geral
# precisao -> quanto o modelo acertou de positivos, nao perante o real, mas sim perante o previsto. 
# recall -> quanto o modelo capta na variavel resposta. Quantos positivos, positivos reais ele acertou dos possiveis positivos.
# especificidade -> é o contrario do recall
# curva ROC envolve um threshold de probabilidade que vai dar alavanca pra melhorar ou piorar o recall e melhorar ou piorar a precisao. A escolha do que melhorar e piorar é do negocio. O que gera menos custo? O que da mais ganho? Aumentar precisao ou recall?.Um bom ponto de corte é a porporçao de sucesso sobre o total. Quanto maior a curva ROC, mais ORDENADAS estao as probabilidades de sucesso e erro do modelo.

#%%

# plots

plt.figure(dpi=400)
plt.plot(auc_decision_tree[0], auc_decision_tree[1], 'o-', color='blue')
plt.plot(auc_naive_bayes[0], auc_naive_bayes[1], 'o-', color='red')
plt.plot(auc_logloss[0], auc_logloss[1], 'o-', color='pink')

plt.grid(True)
plt.title('are sob a curva roc')
plt.xlabel('especificidade')
plt.ylabel('recall')
plt.show()


#%%

#serializaçao do modelo escolhido (logloss)

pd.Series({'model':logloss, 'features': features}).to_pickle('modelo_feliz.pkl')