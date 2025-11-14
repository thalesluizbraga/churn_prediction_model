# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import linear_model

# %%
# REGRESSAO LINEAR

df = pd.read_excel('data/dados_cerveja_nota.xlsx')

X = df[['cerveja']] # tem que passar como um df (duplo colchetes) para formar uma matriz
y = df['nota'] # aqui é só colchete simples mesmo, que é um vetor


# definicaçao do modelo de regressao na variavel reg e o ajuste dele no dataset
# AQUI É O ML
reg = linear_model.LinearRegression()
reg.fit(X,y)

# Coeficientes da regressao rodados
a = reg.intercept_ # o a da funçao de regressao
b = reg.coef_[0] # o b da funcao de regressao
print(a, b) 


# Predicao do modelo
predict = reg.predict(X.drop_duplicates())
plt.plot(X['cerveja'], y, 'o') # Plot dos pontos do dataset
plt.grid(True)
plt.title('Relacao cerveja x nota')
plt.xlabel('cerveja')
plt.ylabel('nota')
plt.plot(X.drop_duplicates()['cerveja'], predict) # Plot da reta de prediçao
plt.legend(['observado', f'y = {a:.3f} + {b:.3f} x']) # ajuste da legenda e casas decimais

#%% 

# ARVORE DE DECISAO - FULL com overfit
arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y) # arvore em todo o dataset que vai overfitar

predict_arvore_full = arvore_full.predict(X.drop_duplicates())
predict = reg.predict(X.drop_duplicates())
plt.plot(X['cerveja'], y, 'o') # Plot dos pontos do dataset
plt.grid(True)
plt.title('Relacao cerveja x nota')
plt.xlabel('cerveja')
plt.ylabel('nota')
plt.plot(X.drop_duplicates()['cerveja'], predict) # modelo de regressao linear
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full) # modelo de arvore de decisao
plt.legend(['observado', f'y = {a:.3f} + {b:.3f} x']) 

#%%

# ARVORE DE DECISAO - sem overfit
arvore = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore.fit(X,y) # arvore em todo o dataset que vai overfitar

predict_arvore = arvore.predict(X.drop_duplicates())
predict = reg.predict(X.drop_duplicates())
plt.plot(X['cerveja'], y, 'o') # Plot dos pontos do dataset
plt.grid(True)
plt.title('Relacao cerveja x nota')
plt.xlabel('cerveja')
plt.ylabel('nota')
plt.plot(X.drop_duplicates()['cerveja'], predict) # modelo de regressao linear
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore) # modelo de arvore de decisao
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.legend(['observado',
            f'y = {a:.3f} + {b:.3f} x',
            'arvore full', 
            'arvore depht = 2']) 

# A ARVORE POR PADRAO NO SKLEARN CHEGA NO NÓ UNICO, O QUE GERA UM OVERFIT, POR ISSO 
# TEM QUE AJUSTAR PELO MAX_DEPTH... as medias do max_depth sao 2^max_depth