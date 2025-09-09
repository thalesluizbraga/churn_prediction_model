# %%
import pandas as pd
from sklearn import tree

# %%

# ETL
df = pd.read_csv('data/dados_comunidade_respostas.csv')

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

# modelos
features = df_analise.columns[:-1].to_list() # pegar todas as colunas menos a ultima que a variavel dependent
X = df_analise[features]
y = df_analise['pessoa_feliz'] 

# decision tree
# definicao do modelo
decision_tree = tree.DecisionTreeClassifier(random_state=42,
                                            min_samples_leaf=5) # min_samples_leaf significa que cada no da arvore tem que ter pelo menos 5 amostras
# fit e predict do modelo
decision_tree.fit(X,y)
decision_tree_predict = decision_tree.predict(X)

#%%
 
 # apresentacao do resultado do modelo no df
# Juntando o predict no df original paa comparaçao
df_predict = df_analise[['pessoa_feliz']]
df_predict['predict_arvore'] = decision_tree_predict
(df_predict['pessoa_feliz'] == df_predict['predict_arvore']).mean() # taxa de 'acerto' do modelo.... a famosa ACURACIA do modelo

#%% 

# metricas

## confusion matrix

# linhas sao os valores verdadeiros, observados
# colunas sao os valores esta dizendo/calculando   
# A intersecao das colunas 1x1 e 0x0 e o que acertei e o oposto disso é o que errei
pd.crosstab(df_predict['pessoa_feliz'], df_predict['predict_arvore'])


# acuracia -> quanto o modelo acerta em geral
# precisao -> quanto o modelo acertou de positivos, nao perante o real, mas sim perante o previsto. 
# recall -> quanto o modelo capta na variavel resposta. Quantos positivos, positivos reais ele acertou dos possiveis positivos.
# especificidade -> é o contrario do recall
# curva ROC envolve um threshold de probabilidade que vai dar alavanca pra melhorar ou piorar o recall e melhorar ou piorar a precisao. A escolha do que melhorar e piorar é do negocio. O que gera menos custo? O que da mais ganho? Aumentar precisao ou recall?.Um bom ponto de corte é a porporçao de sucesso sobre o total. Quanto maior a curva ROC, mais ORDENADAS estao as probabilidades de sucesso e erro do modelo.

