#%%

# imports
import streamlit as st
import pandas as pd

#%%

# Construcao do front end
st.markdown('# Descubra a felicidade')

redes = ['LinkedIn', 'Twitch', 'YouTube', 'Instagram', 'Amigos',
       'Twitter / X', 'Outra rede social']

cursos_options = ['0', '2', '1', '3', 'Mais que 3']

redes_sociais = st.selectbox('Como conheceu o Téo Me Why?', options=redes)
curso = st.selectbox('Quantos cursos acompanhou do Téo Me Why?', options=cursos_options)


col1, col2, col3 = st.columns(3)

with col1:
    video_game = st.radio('Curte Video Game?', ['Sim', 'Não'])
    futebol = st.radio('Curte Futebol?', ['Sim', 'Não'])
    idade = st.number_input('Sua idade', 18, 100)
    
    tempo_area = ['Não atuo', 'De 0 a 6 meses', 'De 6 meses a 1 ano', 'De 1 ano a 2 anos',
        'de 2 anos a 4 anos',  'Mais de 4 anos']
    tempo = st.selectbox('Tempo que atua na área de dados', options=tempo_area)

   
with col2:
    livros = st.radio('Curte livros?' , ['Sim', 'Não'])
    tabuleiro = st.radio('Curte jogos de tabuleiro?', ['Sim', 'Não'])
    
    area_formacao_options = ['Exatas', 'Biológicas', 'Humanas']
    formacao = st.selectbox('Área de Formação', options=area_formacao_options)
    
    senioridade = ['Iniciante', 'Júnior', 'Pleno', 'Sênior', 'Especialista', 
                'Coordenação',  'Gerência', 'Diretoria', 'C-Level']
    cadeira_senioridade = st.selectbox('Senioridade', options=senioridade)

with col3:
    f1 = st.radio('Curte jogos de Fórmula 1?', ['Sim', 'Não'])
    mma = st.radio('Curte jogos de MMA?', ['Sim', 'Não'])
    
    estado_options = ['MG', 'SC', 'SP', 'CE', 'PE', 'RJ', 'AM', 'PR', 'BA', 'PA', 'MT',
       'RS', 'DF', 'RN', 'ES', 'PB', 'GO', 'MA']
    estado = st.selectbox('Estado que mora atualmente', options=estado_options)


# Construcao do back end


num_vars = [
'Curte games?',
'Curte futebol?',
'Curte livros?',
'Curte jogos de tabuleiro?',
'Curte jogos de fórmula 1?',
'Curte jogos de MMA?',
'Idade'
]

data = {    'Como conheceu o Téo Me Why?': redes_sociais,
            'Quantos cursos acompanhou do Téo Me Why?': curso,
            'Curte games?': video_game,
            'Curte futebol?': futebol,
            'Curte livros?': livros,
            'Curte jogos de tabuleiro?': tabuleiro,
            'Curte jogos de fórmula 1?': f1,
            'Curte jogos de MMA?': mma ,
            'Idade': idade,
            'Estado que mora atualmente': estado,
            'Área de Formação': formacao,
            'Tempo que atua na área de dados': tempo,
            'Posição da cadeira (senioridade)': cadeira_senioridade  }


num_vars = [
    'Curte games?',
    'Curte futebol?',
    'Curte livros?',
    'Curte jogos de tabuleiro?',
    'Curte jogos de fórmula 1?',
    'Curte jogos de MMA?',
    'Idade'
]

dummy_vars = [
    'Como conheceu o Téo Me Why?',    
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]

# Transformando variaveis numericas em 0 e 1 e as com mais de 2 opcoes (as dummies)
# em dummies
df = pd.DataFrame([data]).replace({"Sim": 1, "Não": 0, "Yes": 1, "No": 0})
df = pd.get_dummies(df[dummy_vars]).astype(int)

# essa lista tem todas as colunas do df_analise do arquivo metricas.py. 
# essas colunas foram as colunas usadas no modelo e o streamlit tem que ter
# elas igualmente
cols_df_template = [
        'Carimbo de data/hora',
        'Curte games?',
        'Curte futebol?',
        'Curte livros?',
        'Curte jogos de tabuleiro?',
        'Curte jogos de fórmula 1?',
        'Curte jogos de MMA?',
        'Idade',
        'Como conheceu o Téo Me Why?_Instagram',
        'Como conheceu o Téo Me Why?_LinkedIn',
        'Como conheceu o Téo Me Why?_Outra rede social',
        'Como conheceu o Téo Me Why?_Twitch',
        'Como conheceu o Téo Me Why?_Twitter / X',
        'Como conheceu o Téo Me Why?_YouTube',
        'Quantos cursos acompanhou do Téo Me Why?_1',
        'Quantos cursos acompanhou do Téo Me Why?_2',
        'Quantos cursos acompanhou do Téo Me Why?_3',
        'Quantos cursos acompanhou do Téo Me Why?_Mais que 3',
        'Estado que mora atualmente_BA',
        'Estado que mora atualmente_CE',
        'Estado que mora atualmente_DF',
        'Estado que mora atualmente_ES',
        'Estado que mora atualmente_GO',
        'Estado que mora atualmente_MA',
        'Estado que mora atualmente_MG', 
        'Estado que mora atualmente_MT',
        'Estado que mora atualmente_PA', 
        'Estado que mora atualmente_PB',
        'Estado que mora atualmente_PE', 
        'Estado que mora atualmente_PR',
        'Estado que mora atualmente_RJ', 
        'Estado que mora atualmente_RN',
        'Estado que mora atualmente_RS', 
        'Estado que mora atualmente_SC',
        'Estado que mora atualmente_SP', 
        'Área de Formação_Exatas',
        'Área de Formação_Humanas',
        'Tempo que atua na área de dados_De 1 ano a 2 anos',
        'Tempo que atua na área de dados_De 6 meses a 1 ano',
        'Tempo que atua na área de dados_Mais de 4 anos',
        'Tempo que atua na área de dados_Não atuo',
        'Tempo que atua na área de dados_de 2 anos a 4 anos',
        'Posição da cadeira (senioridade)_Coordenação',
        'Posição da cadeira (senioridade)_Diretoria',
        'Posição da cadeira (senioridade)_Especialista',
        'Posição da cadeira (senioridade)_Gerência',
        'Posição da cadeira (senioridade)_Iniciante',
        'Posição da cadeira (senioridade)_Júnior',
        'Posição da cadeira (senioridade)_Pleno',
        'Posição da cadeira (senioridade)_Sênior'

]

# join para pegar todas as opçoes possiveis de flag para os botoes que 
# estao no df_template e um fillna para preencher com 0 os dados nao 
# achados ou nao preenchidos
df_template = pd.DataFrame(columns=cols_df_template)
df = pd.concat([df,df_template]).fillna(0)

# import do modelo e proba 
model = pd.read_pickle('modelo_feliz.pkl')
proba = model['model'].predict_proba(df[model['features']]) [:,1]

if proba >= 0.7:
    st.success('Voce é uma pessoa feliz :)')
elif proba < 0.7 and proba > 0.4:
    st.warning('Voce é uma pessoa meio feliz :|')
else:
    st.error('Voce é uma pessoa triste :(')
st.markdown(proba)
