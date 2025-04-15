#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wrfgpm_metrics.py - Versão Refatorada

Principais melhorias:
1. Correção no parser do tempo dos arquivos GPM
2. Tratamento adequado da grade 2D do GPM
3. Melhor tratamento de valores ausentes
4. Ajustes na leitura dos dados do WRF
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from datetime import datetime

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurações
BASE_OUT = "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1"
WRF_DOMAINS = ["d01", "d02", "d03"]

EVENTOS_CONFIG = {
    "NOV_2008": {
        "periodo": ("2008-11-18", "2008-11-25"),
        "gpm_dir": "/home/vinicius/Documentos/tcc/04-SATELITE/NOV_2008",
        "wrf_dir": "/home/vinicius/Documentos/tcc/01-RODADAS/NOV_2008/03.gerados",
        "estacoes": [
            {"nome": "Florianópolis", "lat": -27.602, "lon": -48.620},
            {"nome": "Laguna", "lat": -28.604, "lon": -48.813}
        ]
    }
}

def parse_gpm_time(filename):
    """Extrai o datetime do nome do arquivo GPM"""
    try:
        time_part = os.path.basename(filename).split('.')[4]
        start_time_str = '-'.join(time_part.split('-')[:2])  # Ex: '20081118-S200000'
        return datetime.strptime(start_time_str, "%Y%m%d-S%H%M%S")
    except Exception as e:
        logging.error(f"Erro ao extrair tempo de {filename}: {e}")
        return None

def ler_gpm(lat, lon, gpm_dir, start_dt, end_dt):
    """Lê dados do GPM com tratamento adequado da grade 2D"""
    dados = []
    
    for arquivo in glob.glob(os.path.join(gpm_dir, "*.HDF5")):
        dt = parse_gpm_time(arquivo)
        if not dt or not (start_dt <= dt <= end_dt):
            continue

        try:
            with h5py.File(arquivo, 'r') as hdf:
                # Verificar estrutura dos dados
                lats = hdf['Grid']['lat'][:]
                lons = hdf['Grid']['lon'][:] % 360  # Converter para 0-360
                precip_data = hdf['Grid']['precipitation'][0]  # Assumindo (time, lat, lon)
                
                # Criar grade 2D
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                
                # Encontrar ponto mais próximo
                target_lon = lon % 360
                dist = np.sqrt((lat_grid - lat)**2 + (lon_grid - target_lon)**2)
                y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
                
                # Extrair precipitação
                precip = precip_data[y_idx, x_idx]
                if np.isclose(precip, -9999.9) or np.isnan(precip):
                    continue
                    
                dados.append({"datetime": dt, "precip_gpm": precip})
                
        except Exception as e:
            logging.error(f"Erro ao ler {arquivo}: {str(e)[:50]}...")
    
    return pd.DataFrame(dados).sort_values('datetime')

def ler_wrf(lat, lon, dominio, wrf_dir, start_dt, end_dt):
    """Versão corrigida com decodificação temporal adequada"""
    try:
        arquivos = sorted(glob.glob(os.path.join(wrf_dir, f"wrfout_{dominio}_*")))
        
        # Abrir dados com tratamento especial para tempo
        ds = xr.open_mfdataset(
            arquivos,
            combine='nested',
            concat_dim='Time',
            decode_times=False  # Decodificaremos manualmente
        )
        
        # Decodificar tempo do WRF
        times = ds.Times.astype(str)
        dt_list = [pd.to_datetime(t.replace('_', ' ')) for t in times.values]
        ds = ds.assign_coords(Time=dt_list)
        
        # Processar coordenadas
        lat2d = ds.XLAT.isel(Time=0).values
        lon2d = ds.XLONG.isel(Time=0).values
        dist = np.sqrt((lat2d - lat)**2 + (lon2d - lon)**2)
        y, x = np.unravel_index(np.argmin(dist), dist.shape)
        
        # Extrair precipitação
        rainc = ds.RAINC.isel(south_north=y, west_east=x)
        rainnc = ds.RAINNC.isel(south_north=y, west_east=x)
        precip = (rainc + rainnc).diff(dim='Time')
        
        # Converter para DataFrame com tempo correto
        df = precip.to_dataframe(name='precip_wrf').reset_index()
        df = df.rename(columns={'Time': 'datetime'})
        df = df[(df.datetime >= start_dt) & (df.datetime <= end_dt)]
        
        return df[['datetime', 'precip_wrf']]
    
    except Exception as e:
        logging.error(f"Erro no WRF {dominio}: {str(e)[:100]}")
        return pd.DataFrame()

def calcular_metricas(df):
    """Calcula métricas com tratamento robusto de dados ausentes"""
    validos = df.dropna(subset=['precip_gpm', 'precip_wrf'])
    if len(validos) < 2:
        return {'MAE': np.nan, 'RMSE': np.nan, 'Willmott': np.nan}
    
    obs = validos['precip_gpm'].values
    sim = validos['precip_wrf'].values
    
    mae = np.mean(np.abs(obs - sim))
    rmse = np.sqrt(np.mean((obs - sim)**2))
    
    obs_mean = np.mean(obs)
    num = np.sum((obs - sim)**2)
    den = np.sum((np.abs(obs - obs_mean) + np.abs(sim - obs_mean))**2)
    willmott = 1 - (num / den) if den != 0 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'Willmott': willmott}

def processar_evento(evento, config):
    """Processa um evento com alinhamento temporal"""
    resultados = []
    start_dt = datetime.strptime(config['periodo'][0], "%Y-%m-%d")
    end_dt = datetime.strptime(config['periodo'][1], "%Y-%m-%d")
    
    for estacao in config['estacoes']:
        logging.info(f"Processando: {estacao['nome']}")
        
        # Ler e processar dados
        df_gpm = ler_gpm(estacao['lat'], estacao['lon'], config['gpm_dir'], start_dt, end_dt)
        
        for dominio in WRF_DOMAINS:
            df_wrf = ler_wrf(estacao['lat'], estacao['lon'], dominio, config['wrf_dir'], start_dt, end_dt)
            
            if df_wrf.empty:
                continue
                
            # Alinhamento temporal
            df_merged = pd.merge_asof(
                df_gpm.sort_values('datetime'),
                df_wrf.rename(columns={'Time': 'datetime'}).sort_values('datetime'),
                on='datetime',
                tolerance=pd.Timedelta('30min')
            ).dropna()
            
            if not df_merged.empty:
                metrics = calcular_metricas(df_merged)
                resultados.append({
                    'Evento': evento,
                    'Dominio': dominio,
                    'Estacao': estacao['nome'],
                    **metrics
                })
    
    return resultados

def main():
    """Função principal com tratamento de exceções"""
    try:
        todos_resultados = []
        for evento, config in EVENTOS_CONFIG.items():
            logging.info(f"\n=== PROCESSANDO EVENTO: {evento} ===")
            todos_resultados.extend(processar_evento(evento, config))
        
        df = pd.DataFrame(todos_resultados)
        caminho_saida = os.path.join(BASE_OUT, "metricas_wrfxgpm_v2.csv")
        df.to_csv(caminho_saida, index=False, float_format="%.4f")
        logging.info(f"Arquivo salvo em: {caminho_saida}")
        return 0
    except Exception as e:
        logging.error(f"Erro fatal: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
