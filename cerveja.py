#%%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt


# %%

# leitura de df
df = pd.read_excel('data/dados_cerveja.xlsx')

# %%

# Separacao das features e target + encoding
X = df.drop('classe', axis=1) # fatures que tem que estar TODAS como numero
y = df['classe'] # target

X = X.replace( {
'mud':1 , 'pint':2,
'sim': 1, 'não': 0,
'clara': 0, 'escura':1
}
) # lista para encoding de features que nao estao numericas

# %%

# Modelo rodando
model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)

# %%

plt.figure(dpi=4000) # so pra deixar a imagem mais legivel
tree.plot_tree(model, feature_names=X.columns.to_list(),
                class_names=model.classes_,
                filled=True)

# %%