#%%

import pandas as pd


# %%

path= 'data/abt_churn.csv'
model_df = pd.read_pickle('model.pkl')
model = model_df['model']
features = model_df['features']

df= pd.read_csv(path)

# pegando 3 linhas da abt como amostra de predicao
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)


predicao = model.predict_proba(amostra[features])[:,1]
amostra['proba'] = predicao
amostra


# %%
