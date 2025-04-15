#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurações iniciais
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans'})

# Carregar os dados
path = "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1/metricas_consolidadas.csv"
df = pd.read_csv(path)

# Definir parâmetros de filtragem
filtros = {
    'NOV_2008': {'dominio': 'd03', 'estacoes': [
        'inmet_florianopolis_limpo',
        'inmet_laguna_limpo',
        'inmet_ararangua_limpo',
        'inmet_itapoa_limpo',
        'inmet_bom_jardim_da_serra_limpo',
        'inmet_urussanga_limpo'
    ]},
    'OUT_2021': {'dominio': 'd02', 'estacoes': [
        'FLORIANOPOLIS',
        'Laguna - Farol de Santa Marta',
        'ARARANGUA',
        'ITAPOA',
        'BOM JARDIM DA SERRA - MORRO DA IGREJA',
        'URUSSANGA'
    ]},
    'NOV_2022': {'dominio': 'd02', 'estacoes': [
        'FLORIANOPOLIS',
        'Laguna - Farol de Santa Marta',
        'ARARANGUA',
        'ITAPOA',
        'BOM JARDIM DA SERRA - MORRO DA IGREJA',
        'URUSSANGA'
    ]}
}

# Processar os dados
dados_plot = []
for evento, params in filtros.items():
    temp = df[(df['Evento'] == evento) & 
             (df['Dominio'] == params['dominio']) &
             (df['Estacao'].isin(params['estacoes']))]
    
    medias = {
        'Evento': evento,
        'MAE': temp['MAE_wrfxobs'].mean(),
        'RMSE': temp['RMSE_wrfxobs'].mean(),
        'Willmott': temp['d_wrfxobs'].mean()
    }
    dados_plot.append(medias)

df_plot = pd.DataFrame(dados_plot)

# Converter nomes dos eventos
df_plot['Evento'] = df_plot['Evento'].replace({
    'NOV_2008': 'Nov/2008',
    'OUT_2021': 'Out/2021',
    'NOV_2022': 'Nov/2022'
})

# Criar gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Definir posições no eixo X
x = np.arange(len(df_plot))
width = 0.25

# Plotar barras
bars1 = ax.bar(x - width, df_plot['MAE'], width, label='MAE (mm)', color='#2ecc71')
bars2 = ax.bar(x, df_plot['RMSE'], width, label='RMSE (mm)', color='#e74c3c')
bars3 = ax.bar(x + width, df_plot['Willmott'], width, label='Willmott (d)', color='#3498db')

# Customizar gráfico
ax.set_title('Evolução das Métricas de Validação entre os Eventos Estudos', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_plot['Evento'])
ax.set_ylabel('Valor da Métrica')
ax.legend(loc='upper right', frameon=True)
ax.set_ylim(0, 12)

# Adicionar valores nas barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Ajustar layout e salvar
plt.tight_layout()
plt.savefig('evolucao_metricas.png', dpi=300, bbox_inches='tight')
plt.show()
