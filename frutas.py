# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# %%

df = pd.read_excel('data/dados_frutas.xlsx')
# o random state 42 é para que todo mundo que rode o modelo, o modelo sempre selecione as variaveis aleatoriamente da mesma forma
arvore = tree.DecisionTreeClassifier(random_state=42)

y = df['Fruta']
X= df.select_dtypes(exclude='object') # Coloquei todas as colunas do df menos a variavel resposta

arvore.fit(X, y) # ESSA LINHA É O AJUSTE DO MODELO

# Cada 1 desse é o positivo na coluna de caracteristicas
# suculenta, arredondada, etc
arvore.predict([[1,1,1,1]]) 

tree.plot_tree(arvore,
                feature_names=X.columns.tolist(), # os nomes das variaveis tem que ser passadas como lista
                class_names=arvore.classes_,
                filled=True)


# A arvore setou as caracteristicas por ordem alfabetica e por isso a banana veio primeiro
# %%

# esse codigo retorna a probabilidade de cada linha ser compativel ao que esta sendo 
# pedido na variavel proba, que no caso é 1=positivo em todas as quatro colunas de caracteristicas
proba = arvore.predict_proba([[1,1,1,1]]) [0]
pd.Series(proba, index=arvore.classes_)
    
# %%
