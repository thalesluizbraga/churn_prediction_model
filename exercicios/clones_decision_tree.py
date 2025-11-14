# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

#%%

df = pd.read_parquet('data/dados_clones.parquet')
df = df.rename(columns={'Status ': 'status'})

# %%

x_features = df.drop(['p2o_master_id', 'status' ,'General Jedi encarregado'], axis=1) 
y_target = df['status']


#%%

x_features['Distância Ombro a ombro'] = x_features['Distância Ombro a ombro'].apply(lambda x: 1 if x == 'Tipo 1' else 
                                      2 if x == 'Tipo 2' else
                                      3 if x == 'Tipo 3' else
                                      4 if x == 'Tipo 4' else
                                      5)

x_features['Tamanho do crânio'] = x_features['Tamanho do crânio'].apply(lambda x: 1 if x == 'Tipo 1' else 
                                      2 if x == 'Tipo 2' else
                                      3 if x == 'Tipo 3' else
                                      4 if x == 'Tipo 4' else
                                      5)

x_features['Tamanho dos pés'] = x_features['Tamanho dos pés'].apply(lambda x: 1 if x == 'Tipo 1' else 
                                      2 if x == 'Tipo 2' else
                                      3 if x == 'Tipo 3' else
                                      4 if x == 'Tipo 4' else
                                      5)

# %%

# Modelo
model = tree.DecisionTreeClassifier()
model.fit(X=x_features, y=y_target)

#%%

# plot da decision tree
plt.figure(dpi=400) # so pra deixar a imagem mais legivel
tree.plot_tree(model, feature_names=x_features.columns.to_list(),
                class_names=model.classes_,
                filled= True,
                max_depth=3)

# %%
