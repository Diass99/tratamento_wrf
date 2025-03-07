#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06-estacoes_vs_modelo_prec_wrfgpm.py

Exemplo:
  - Para o evento NOV_2008 (período 2008-11-18 00:00:00 a 2008-11-25 00:00:00)
  - Para cada estação observada:
      * Extrair GPM no ponto (lat, lon)
      * Extrair WRF no ponto (lat, lon) para cada domínio (d01, d02, d03)
  - Gerar um gráfico comparando as 3 curvas: OBS, GPM, WRF (para cada domínio)
  - Salvar a figura em:
      BASE_OUT/NOV_2008/ESTACOES/<station_name>/compare_prec_<station_name>_<domain>.png

Uso:
  python 06-estacoes_vs_modelo_prec_wrfgpm.py
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import h5py
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Intervalo fixo para NOV_2008
START_DT = pd.to_datetime("2008-11-18 00:00:00")
END_DT   = pd.to_datetime("2008-11-25 00:00:00")

# Diretórios de entrada
METADADOS_CSV = "/home/vinicius/Documentos/tcc/03-ESTACOES/metadados_estacoes.csv"
ESTACOES_DIR  = "/home/vinicius/Documentos/tcc/03-ESTACOES"

# Diretórios para as fontes de dados
GPM_DIR = "/home/vinicius/Documentos/tcc/04-SATELITE/NOV_2008"
WRF_DIR = "/home/vinicius/Documentos/tcc/01-RODADAS/NOV_2008/03.gerados"

# Diretório base de saída
BASE_OUT = "/home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1"

###############################################################################
# Função para ler os metadados das estações
###############################################################################
def get_station_list():
    try:
        df = pd.read_csv(METADADOS_CSV, sep=",", header=0, index_col=0)
    except Exception as e:
        logging.error(f"Erro ao ler {METADADOS_CSV}: {e}")
        return []
    
    station_list = []
    for station_name, row in df.iterrows():
        nome_est = row.get("nome", station_name)
        try:
            lat_est = float(str(row["Lat"]).replace(",", "."))
            lon_est = float(str(row["Long"]).replace(",", "."))
            alt_est = float(str(row["Altitude"]).replace(",", ".")) if "Altitude" in row else np.nan
        except Exception as e:
            logging.error(f"Erro ao converter dados para a estação {station_name}: {e}")
            continue
        csv_file = os.path.join(ESTACOES_DIR, f"{nome_est}.csv")
        station_list.append({
            "nome": nome_est,
            "lat": lat_est,
            "lon": lon_est,
            "alt": alt_est,
            "arquivo_csv": csv_file
        })
    return station_list

###############################################################################
# Função para ler dados observados de precipitação da estação
###############################################################################
def read_station_precip(csv_file):
    try:
        df = pd.read_csv(csv_file, sep=",", header=0)
    except Exception as e:
        logging.error(f"Erro ao ler CSV da estação {csv_file}: {e}")
        return None
    if "precipitacao(mm)" not in df.columns or "datetime" not in df.columns:
        logging.error(f"CSV {csv_file} não contém as colunas esperadas ('precipitacao(mm)', 'datetime').")
        return None
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["precipitacao(mm)"] = df["precipitacao(mm)"].astype(str).str.replace(",", ".").astype(float)
    df = df.rename(columns={"precipitacao(mm)": "precip_obs"})
    df = df[["datetime", "precip_obs"]].copy()
    df = df[(df["datetime"] >= START_DT) & (df["datetime"] <= END_DT)]
    logging.info(f"OBS: {csv_file} - {len(df)} registros entre {df['datetime'].min()} e {df['datetime'].max()}")
    return df

###############################################################################
# Função para extrair precipitação do WRF para um domínio específico
###############################################################################
def extract_wrf_precip(lat_est, lon_est, domain):
    import xarray as xr
    pattern = f"wrfout_{domain}_2008"
    wrf_files = sorted([f for f in os.listdir(WRF_DIR) if f.startswith(pattern)])
    if not wrf_files:
        logging.warning(f"WRF: Não achou arquivos {pattern} para NOV_2008.")
        return None
    wrf_paths = [os.path.join(WRF_DIR, f) for f in wrf_files]
    
    try:
        ds = xr.open_mfdataset(wrf_paths, combine="nested", concat_dim="Time")
    except Exception as e:
        logging.error(f"WRF-{domain}: Erro ao abrir arquivos: {e}")
        return None

    if "Times" in ds.variables:
        times_str = [''.join(t.astype(str)) for t in ds["Times"].values]
        try:
            times = pd.to_datetime(times_str, format="%Y-%m-%d_%H:%M:%S", errors="coerce")
        except Exception as e:
            logging.error(f"WRF-{domain}: Erro na conversão de datas: {e}")
            ds.close()
            return None
        ds = ds.assign_coords(time=("Time", times))
        ds = ds.swap_dims({"Time": "time"})
    else:
        ds = ds.rename({"Times": "time"})
    
    ds = ds.sel(time=slice(START_DT, END_DT))
    logging.info(f"WRF-{domain}: Dataset possui {ds.sizes['time']} tempos, primeiros: {ds['time'].values[:3]}")
    if ds.sizes["time"] < 2:
        logging.warning(f"WRF-{domain}: Poucos tempos no intervalo.")
        ds.close()
        return None

    if "XLAT" in ds and "XLONG" in ds:
        lat2d = ds["XLAT"].isel(time=0).squeeze()
        lon2d = ds["XLONG"].isel(time=0).squeeze()
    else:
        logging.warning(f"WRF-{domain}: Não achou XLAT/XLONG.")
        ds.close()
        return None

    latvals = lat2d.values
    lonvals = lon2d.values
    dist2 = (latvals - lat_est)**2 + (lonvals - lon_est)**2
    idx = np.argmin(dist2)
    j, i = np.unravel_index(idx, dist2.shape)
    logging.info(f"WRF-{domain}: Ponto mais próximo para ({lat_est},{lon_est}) é ({j},{i})")

    if "RAINC" not in ds.data_vars or "RAINNC" not in ds.data_vars:
        logging.warning(f"WRF-{domain}: Não achou RAINC/RAINNC.")
        ds.close()
        return None

    rainc = ds["RAINC"].isel(south_north=j, west_east=i)
    rainnc = ds["RAINNC"].isel(south_north=j, west_east=i)
    total_rain = (rainc + rainnc)
    total_np = total_rain.values
    times_np = ds["time"].values
    ds.close()
    
    logging.info(f"WRF-{domain}: Total de registros = {len(total_np)}")
    precip_list = []
    for t in range(1, len(total_np)):
        dt = times_np[t]
        # Calcula a diferença entre os acumulados:
        precip_val = total_np[t] - total_np[t-1]
        # Se a diferença for negativa, define como 0 (pois precipitação não pode ser negativa)
        if precip_val < 0:
            precip_val = 0.0
        precip_list.append((dt, precip_val))
    df = pd.DataFrame(precip_list, columns=["datetime", "precip_wrf"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    logging.info(f"WRF-{domain}: df_wrf tem {len(df)} registros, primeiro tempo: {df['datetime'].iloc[0]}")
    return df

###############################################################################
# Função para extrair precipitação do GPM
###############################################################################
def extract_gpm_precip(lat_est, lon_est, start_dt, end_dt):
    import glob
    gpm_files = sorted(glob.glob(os.path.join(GPM_DIR, "*.HDF5")))
    if not gpm_files:
        logging.warning("GPM: Não achou arquivos GPM HDF5 para NOV_2008.")
        return None
    
    times_list = []
    precip_list = []
    
    for f in gpm_files:
        fname = os.path.basename(f)
        try:
            parts = fname.split(".")
            if len(parts) < 5:
                continue
            date_field = parts[4]  # Ex: "20081118-S000000-E002959"
            subparts = date_field.split("-")
            if len(subparts) < 2:
                continue
            date_part = subparts[0]
            time_part = subparts[1][1:]  # Remove o "S"
            formatted = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}_{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}"
            dt_file = pd.to_datetime(formatted, format="%Y-%m-%d_%H:%M:%S")
        except Exception as e:
            logging.warning(f"GPM: Falha ao parse do arquivo {fname}: {e}")
            continue
        
        if dt_file < start_dt or dt_file > end_dt:
            continue
        
        try:
            hf = h5py.File(f, "r")
            lat_arr = hf["Grid"]["lat"][:]
            lon_arr = hf["Grid"]["lon"][:]
            prec = hf["Grid"]["precipitation"][:]
            prec[prec == -9999.9] = np.nan
            prec2d = prec[0, :, :].T
            Lon2d, Lat2d = np.meshgrid(lon_arr, lat_arr)
            Lon2d_fix = np.where(Lon2d < 0, Lon2d + 360, Lon2d)
            lon_est_adj = lon_est if lon_est >= 0 else lon_est + 360
            dist2 = (Lat2d - lat_est)**2 + (Lon2d_fix - lon_est_adj)**2
            idx = np.argmin(dist2)
            jy, ix = np.unravel_index(idx, dist2.shape)
            gpm_val = prec2d[jy, ix]
            times_list.append(dt_file)
            precip_list.append(gpm_val)
            hf.close()
        except Exception as e:
            logging.warning(f"GPM: Falha ao ler {fname}: {e}")
            continue
    
    if not times_list:
        logging.warning("GPM: Nenhum dado extraído.")
        return None
    df = pd.DataFrame({"datetime": times_list, "precip_gpm": precip_list})
    df = df.sort_values("datetime").reset_index(drop=True)
    logging.info(f"GPM: df_gpm tem {len(df)} registros, primeiro tempo: {df['datetime'].iloc[0]}")
    return df

###############################################################################
# Função para plotar o gráfico comparativo
###############################################################################
def plot_compare_prec(df_obs, df_gpm, df_wrf, sim, station_name, domain):
    outdir = os.path.join(BASE_OUT, sim, "ESTACOES", station_name)
    os.makedirs(outdir, exist_ok=True)
    figname = f"compare_prec_{station_name}_{domain}.png"
    outfile = os.path.join(outdir, figname)
    
    fig, ax = plt.subplots(figsize=(10,5))
    if df_obs is not None and not df_obs.empty:
        ax.plot(df_obs["datetime"], df_obs["precip_obs"], label="Estação", color="black")
    else:
        logging.warning(f"{station_name}: OBS vazio ou None")
    if df_gpm is not None and not df_gpm.empty:
        ax.plot(df_gpm["datetime"], df_gpm["precip_gpm"], label="GPM", color="green")
    else:
        logging.warning(f"{station_name}: GPM vazio ou None")
    if df_wrf is not None and not df_wrf.empty:
        ax.plot(df_wrf["datetime"], df_wrf["precip_wrf"], label="WRF", color="red")
    else:
        logging.warning(f"{station_name}: WRF vazio ou None")
    
    ax.set_xlabel("Data/Hora")
    ax.set_ylabel("Precipitação (mm/h)")
    ax.set_title(f"Comparação de Precipitação - {station_name} - {sim} - {domain}")
    ax.legend(loc="upper right")
    plt.xticks(rotation=25)
    
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Gráfico salvo em: {outfile}")

###############################################################################
# Função principal
###############################################################################
def main():
    # Definindo explicitamente o evento sim
    sim = "NOV_2008"
    station_list = get_station_list()
    if not station_list:
        logging.error("Nenhuma estação encontrada nos metadados.")
        return
    
    logging.info(f"=== Comparando precip para {sim} ===")
    start_dt = START_DT
    end_dt   = END_DT
    
    for st in station_list:
        station_name = st["nome"]
        lat = st["lat"]
        lon = st["lon"]
        csv_file = st["arquivo_csv"]
        logging.info(f"=== Estação {station_name} ===")
        
        # OBS
        df_obs = read_station_precip(csv_file)
        if df_obs is not None:
            logging.info(f"{station_name}: OBS -> {len(df_obs)} registros")
        else:
            logging.warning(f"{station_name}: OBS não disponível.")
        
        # GPM
        df_gpm = extract_gpm_precip(lat, lon, start_dt, end_dt)
        if df_gpm is not None:
            logging.info(f"{station_name}: GPM -> {len(df_gpm)} registros")
        else:
            logging.warning(f"{station_name}: GPM não disponível.")
        
        # Para cada domínio do WRF: d01, d02, d03
        for domain in ["d01", "d02", "d03"]:
            df_wrf = extract_wrf_precip(lat, lon, domain)
            if df_wrf is not None:
                logging.info(f"{station_name}: WRF-{domain} -> {len(df_wrf)} registros")
            else:
                logging.warning(f"{station_name}: WRF-{domain} não disponível.")
            
            plot_compare_prec(df_obs, df_gpm, df_wrf, sim, station_name, domain)
    
    logging.info("=== Fim do script 06-estacoes_vs_modelo_prec_wrfgpm. ===")

if __name__ == "__main__":
    main()

