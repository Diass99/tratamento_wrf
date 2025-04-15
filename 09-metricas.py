#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09-metricas.py

Script que:
 - Lê eventos definidos em EVENTS_CFG (NOV_2008, OUT_2021, NOV_2022),
 - Para cada evento, carrega metadados (nome de estação, lat, lon, arquivo_csv_pad),
 - Lê dados de precipitação observada (CSV padronizado => datetime, precipitacao(mm)),
 - Lê GPM (HDF5) e WRF (NetCDF) para extrair precipitação no ponto lat/lon,
 - Calcula métricas (MAE, RMSE, Willmott d) de WRF vs. OBS e WRF vs. GPM,
 - Salva tudo em metricas_consolidadas.csv no BASE_OUT.

Este script inclui diversos logs em nível DEBUG para rastrear cada passo do processo.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from datetime import datetime
from argparse import ArgumentParser

# -------------------------------------------------------------------
# CONFIGURANDO O LOGGING PARA DEBUG
# -------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Caminho para salvar o CSV final de métricas
BASE_OUT = "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1"

# -------------------------------------------------------------------
# CONFIGURAÇÃO DOS EVENTOS
# -------------------------------------------------------------------
EVENTS_CFG = {
    "NOV_2008": {
        "start_dt": pd.to_datetime("2008-11-18 00:00:00"),
        "end_dt":   pd.to_datetime("2008-11-25 00:00:00"),
        "obs_dir":  "/home/vinicius/Documentos/tcc/03-ESTACOES/nov_2008/padronizados",
        "gpm_dir":  "/home/vinicius/Documentos/tcc/04-SATELITE/NOV_2008",
        "wrf_dir":  "/home/vinicius/Documentos/tcc/01-RODADAS/NOV_2008/03.gerados",
        "wrf_year": "2008",
        "metadata_file": "/home/vinicius/Documentos/tcc/03-ESTACOES/nov_2008/padronizados/metadados_padronizados.csv"
    },
    "OUT_2021": {
        "start_dt": pd.to_datetime("2021-10-10 00:00:00"),
        "end_dt":   pd.to_datetime("2021-10-13 00:00:00"),
        "obs_dir":  "/home/vinicius/Documentos/tcc/03-ESTACOES/out_2021/padronizados",
        "gpm_dir":  "/home/vinicius/Documentos/tcc/04-SATELITE/OUT_2021",
        "wrf_dir":  "/home/vinicius/Documentos/tcc/01-RODADAS/OUT_2021/03.gerados",
        "wrf_year": "2021",
        "metadata_file": "/home/vinicius/Documentos/tcc/03-ESTACOES/out_2021/padronizados/metadados_padronizados.csv"
    },
    "NOV_2022": {
        "start_dt": pd.to_datetime("2022-11-30 00:00:00"),
        "end_dt":   pd.to_datetime("2022-12-03 00:00:00"),
        "obs_dir":  "/home/vinicius/Documentos/tcc/03-ESTACOES/nov_2022/padronizados",
        "gpm_dir":  "/home/vinicius/Documentos/tcc/04-SATELITE/NOV_2022",
        "wrf_dir":  "/home/vinicius/Documentos/tcc/01-RODADAS/NOV_2022/03.gerados",
        "wrf_year": "2022",
        "metadata_file": "/home/vinicius/Documentos/tcc/03-ESTACOES/nov_2022/padronizados/metadados_padronizados.csv"
    }
}

# Lista de domínios WRF que serão processados
WRF_DOMAINS = ["d01", "d02", "d03"]

# -------------------------------------------------------------------
# FUNÇÕES PARA MÉTRICAS
# -------------------------------------------------------------------
def mae(obs, sim):
    """
    Mean Absolute Error
    """
    val = np.mean(np.abs(obs - sim))
    logging.debug(f"   [MAE] => {val:.3f}")
    return val

def rmse(obs, sim):
    """
    Root Mean Squared Error
    """
    val = np.sqrt(np.mean((obs - sim)**2))
    logging.debug(f"   [RMSE] => {val:.3f}")
    return val

def willmott_d(obs, sim):
    """
    Índice de Willmott (d).
    d = 1 - [ sum((obs-sim)^2) / sum((|obs-obs_bar| + |sim-obs_bar|)^2 ) ]
    """
    obs_bar = np.mean(obs)
    num = np.sum((obs - sim)**2)
    den = np.sum((np.abs(obs - obs_bar) + np.abs(sim - obs_bar))**2)
    if den == 0:
        logging.debug("   [Willmott d] => den=0 => retornando NaN")
        return np.nan
    d_val = 1 - (num / den)
    logging.debug(f"   [Willmott d] => {d_val:.3f}")
    return d_val

# -------------------------------------------------------------------
# FUNÇÃO PARA LER DADOS OBS (PADRONIZADAS)
# -------------------------------------------------------------------
def read_obs_padronizadas(obs_dir, csv_name, start_dt, end_dt):
    """
    Espera colunas 'datetime' e 'precipitacao(mm)'.
    Retorna DataFrame => [datetime, precip_obs].
    """
    path = os.path.join(obs_dir, csv_name)
    logging.debug(f"   [OBS] Lendo CSV => {path}")
    if not os.path.isfile(path):
        logging.warning(f"   [OBS] Arquivo nao encontrado => {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"   [OBS] Erro ao ler {path}: {e}")
        return None

    if "datetime" not in df.columns or "precipitacao(mm)" not in df.columns:
        logging.error(f"   [OBS] Falta 'datetime' ou 'precipitacao(mm)' em {path}. Colunas => {df.columns}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["precipitacao(mm)"] = pd.to_numeric(df["precipitacao(mm)"], errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    df = df.rename(columns={"precipitacao(mm)": "precip_obs"})

    # Filtra período
    before_count = len(df)
    df = df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)]
    after_count = len(df)
    logging.debug(f"   [OBS] Registros antes={before_count}, depois={after_count}, período={start_dt}..{end_dt}")
    df.sort_values("datetime", inplace=True)

    return df[["datetime","precip_obs"]]

# -------------------------------------------------------------------
# FUNÇÃO PARA LER GPM
# -------------------------------------------------------------------
def extract_gpm_precip(lat, lon, start_dt, end_dt, gpm_dir):
    """
    Retorna DataFrame => [datetime, precip_gpm] ou None.
    """
    logging.debug(f"   [GPM] lat={lat}, lon={lon}, gpm_dir={gpm_dir}")
    if not os.path.isdir(gpm_dir):
        logging.warning(f"   [GPM] Dir inexistente => {gpm_dir}")
        return None

    hdf5_files = sorted(glob.glob(os.path.join(gpm_dir, "*.HDF5")))
    logging.debug(f"   [GPM] Encontrados {len(hdf5_files)} arquivos .HDF5 em {gpm_dir}")

    tlist, plist = [], []
    for f in hdf5_files:
        fname = os.path.basename(f)
        parts = fname.split(".")
        if len(parts)<5:
            continue
        # Ex.: 3B-HHR...20211010-S000000-E002959...
        date_str = parts[4].split("-")[0]  # "20211010"
        dt_file  = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        if dt_file is None:
            continue
        if dt_file < start_dt.floor("D") or dt_file> end_dt.ceil("D"):
            continue
        # extra hour
        subp = parts[4].split("-")
        if len(subp)<2:
            continue
        time_part = subp[1][1:]  # remove 'S'
        dt_str2 = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}_{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        dt_full = pd.to_datetime(dt_str2, errors="coerce")
        if dt_full is None or dt_full<start_dt or dt_full>end_dt:
            continue

        # Lê o HDF5
        try:
            hf = h5py.File(f, "r")
            lat_arr = hf["Grid"]["lat"][:]
            lon_arr = hf["Grid"]["lon"][:]
            prec    = hf["Grid"]["precipitation"][:]
            prec[prec==-9999.9] = np.nan
            prec2d = prec[0,:,:].T
            Lon2d, Lat2d = np.meshgrid(lon_arr, lat_arr)
            Lon2d_fix = np.where(Lon2d<0, Lon2d+360, Lon2d)
            lon_adj = lon if lon>=0 else lon+360
            dist2 = (Lat2d - lat)**2 + (Lon2d_fix - lon_adj)**2
            idx = np.argmin(dist2)
            jy, ix = np.unravel_index(idx, dist2.shape)
            val = prec2d[jy, ix]
            hf.close()
        except Exception as e:
            logging.warning(f"   [GPM] Falha ao ler {fname}: {e}")
            continue

        tlist.append(dt_full)
        plist.append(val)

    if not tlist:
        logging.debug("   [GPM] Nenhuma amostra no período")
        return None
    df_gpm = pd.DataFrame({"datetime":tlist, "precip_gpm":plist})
    df_gpm.sort_values("datetime", inplace=True)
    logging.debug(f"   [GPM] => {len(df_gpm)} registros no período")
    return df_gpm

# -------------------------------------------------------------------
# FUNÇÃO PARA LER WRF
# -------------------------------------------------------------------
def extract_wrf_precip(lat, lon, domain, wrf_dir, start_dt, end_dt, wrf_year):
    logging.debug(f"   [WRF-{domain}] lat={lat},lon={lon},year={wrf_year}")
    pattern = f"wrfout_{domain}_{wrf_year}"
    wrf_files = sorted(os.path.join(wrf_dir, f) for f in os.listdir(wrf_dir) if f.startswith(pattern))
    logging.debug(f"   [WRF-{domain}] Encontrados {len(wrf_files)} arquivos => {wrf_files}")

    if not wrf_files:
        return None

    try:
        ds = xr.open_mfdataset(
            wrf_files,
            combine="nested",
            concat_dim="Time",
            decode_times=False
        )
    except Exception as e:
        logging.error(f"   [WRF-{domain}] Erro open_mfdataset => {e}")
        return None

    # converte Times
    if "Times" in ds:
        times_bytes = ds["Times"].values
        time_list = []
        for tarr in times_bytes:
            tstr = tarr.tobytes().decode("utf-8")
            dt_ = pd.to_datetime(tstr, format="%Y-%m-%d_%H:%M:%S", errors="coerce")
            time_list.append(dt_)
        ds = ds.assign_coords(time=("Time", np.array(time_list, dtype="datetime64[ns]")))
        ds = ds.swap_dims({"Time":"time"})

    ds = ds.sel(time=slice(start_dt, end_dt))
    nt = ds.sizes.get("time", 0)
    logging.debug(f"   [WRF-{domain}] times filtrados => {nt} => {start_dt}..{end_dt}")
    if nt<2:
        ds.close()
        return None

    if ("XLAT" not in ds) or ("XLONG" not in ds):
        ds.close()
        logging.warning(f"   [WRF-{domain}] sem XLAT/XLONG")
        return None

    lat2d = ds["XLAT"].isel(time=0).values
    lon2d = ds["XLONG"].isel(time=0).values
    dist2 = (lat2d - lat)**2 + (lon2d - lon)**2
    idx = np.argmin(dist2)
    j, i = np.unravel_index(idx, dist2.shape)
    logging.debug(f"   [WRF-{domain}] Ponto mais próximo => (j={j},i={i})")

    if "RAINC" not in ds or "RAINNC" not in ds:
        ds.close()
        logging.warning(f"   [WRF-{domain}] Falta RAINC/RAINNC.")
        return None

    rainc  = ds["RAINC"].isel(south_north=j, west_east=i)
    rainnc = ds["RAINNC"].isel(south_north=j, west_east=i)
    total = rainc + rainnc
    time_vals = ds["time"].values
    ds.close()

    vals = total.values
    if len(vals) < 2:
        logging.debug("   [WRF-{domain}] Menos de 2 valores => sem precip horária.")
        return None

    plist = []
    for k in range(1,len(vals)):
        accum = vals[k] - vals[k-1]
        if accum<0:
            accum=0
        plist.append((time_vals[k], accum))
    df_wrf = pd.DataFrame(plist, columns=["datetime","precip_wrf"])
    logging.debug(f"   [WRF-{domain}] => df_wrf com {len(df_wrf)} regs")
    return df_wrf

# -------------------------------------------------------------------
# CÁLCULO DE MÉTRICAS
# -------------------------------------------------------------------
def calc_metrics_wrfgpm(df_wrf, df_obs, df_gpm):
    """
    Retorna dict => [MAE_wrfxobs, RMSE_wrfxobs, d_wrfxobs,
                     MAE_wrfxgpm, RMSE_wrfxgpm, d_wrfxgpm]
    """
    out = {
        "MAE_wrfxobs": np.nan,
        "RMSE_wrfxobs": np.nan,
        "d_wrfxobs": np.nan,
        "MAE_wrfxgpm": np.nan,
        "RMSE_wrfxgpm": np.nan,
        "d_wrfxgpm": np.nan
    }
    logging.debug("   [calc_metrics_wrfgpm] Entrando ...")

    # (1) WRF vs OBS
    if df_wrf is not None and df_obs is not None:
        logging.debug(f"   -> WRF vs OBS: df_wrf={len(df_wrf)} regs, df_obs={len(df_obs)} regs.")
        dfm = pd.merge(df_wrf, df_obs, on="datetime", how="inner")
        logging.debug(f"   -> Merge wrf-obs => {len(dfm)} regs em comum")
        if not dfm.empty:
            obs = dfm["precip_obs"].values
            wrf = dfm["precip_wrf"].values
            out["MAE_wrfxobs"]  = mae(obs, wrf)
            out["RMSE_wrfxobs"] = rmse(obs, wrf)
            out["d_wrfxobs"]    = willmott_d(obs, wrf)

    # (2) WRF vs GPM
    if df_wrf is not None and df_gpm is not None:
        logging.debug(f"   -> WRF vs GPM: df_wrf={len(df_wrf)} regs, df_gpm={len(df_gpm)} regs.")
        dfm2 = pd.merge(df_wrf, df_gpm, on="datetime", how="inner")
        logging.debug(f"   -> Merge wrf-gpm => {len(dfm2)} regs em comum")
        if not dfm2.empty:
            gpm = dfm2["precip_gpm"].values
            wrf = dfm2["precip_wrf"].values
            out["MAE_wrfxgpm"]  = mae(gpm, wrf)
            out["RMSE_wrfxgpm"] = rmse(gpm, wrf)
            out["d_wrfxgpm"]    = willmott_d(gpm, wrf)

    logging.debug(f"   [calc_metrics_wrfgpm] -> Resultado => {out}")
    return out

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--eventos", nargs="+", default=["NOV_2008","OUT_2021","NOV_2022"],
                        help="Lista de eventos (ex. NOV_2008 OUT_2021 NOV_2022)")
    args = parser.parse_args()

    eventos = args.eventos
    logging.debug(f"[MAIN] Eventos a processar => {eventos}")
    results_list = []

    for evento in eventos:
        if evento not in EVENTS_CFG:
            logging.error(f"[MAIN] Evento {evento} não está em EVENTS_CFG.")
            continue

        cfg = EVENTS_CFG[evento]
        start_dt = cfg["start_dt"]
        end_dt   = cfg["end_dt"]
        obs_dir  = cfg["obs_dir"]
        gpm_dir  = cfg["gpm_dir"]
        wrf_dir  = cfg["wrf_dir"]
        wrf_year = cfg["wrf_year"]
        meta_csv = cfg["metadata_file"]

        logging.info(f"\n=== PROCESSANDO EVENTO {evento} => {start_dt}..{end_dt} ===")
        logging.debug(f"obs_dir={obs_dir}, gpm_dir={gpm_dir}, wrf_dir={wrf_dir}, meta_csv={meta_csv}")

        if not os.path.isfile(meta_csv):
            logging.error(f"[{evento}] Metadados inexistente => {meta_csv}")
            continue

        # Lê metadados
        try:
            df_meta = pd.read_csv(meta_csv)
        except Exception as e:
            logging.error(f"[{evento}] Erro lendo {meta_csv}: {e}")
            continue
        logging.debug(f"[{evento}] Metadados => {len(df_meta)} linhas, colunas => {df_meta.columns.tolist()}")

        # Precisamos de 'nome', 'lat', 'lon', 'arquivo_csv_pad' (ou equivalentes).
        col_map = {}
        for c in df_meta.columns:
            c_low = c.strip().lower()
            if c_low in ("nome","estacao","station"):
                col_map["nome"] = c
            elif c_low in ("lat","latitude"):
                col_map["lat"] = c
            elif c_low in ("lon","long","longitude"):
                col_map["lon"] = c
            elif c_low in ("arquivo_csv_pad","file_csv"):
                col_map["csv_pad"] = c

        missing = [x for x in ["nome","lat","lon","csv_pad"] if x not in col_map]
        if missing:
            logging.error(f"[{evento}] Faltam colunas {missing} em {meta_csv}. Achou => {df_meta.columns.tolist()}")
            continue

        # Loop estações
        for idx, row in df_meta.iterrows():
            st_name = str(row[col_map["nome"]])
            try:
                lat = float(row[col_map["lat"]])
                lon = float(row[col_map["lon"]])
            except:
                logging.warning(f"[{evento}] Falha convertendo lat/lon p/ {st_name}. Pulando.")
                continue
            station_file = str(row[col_map["csv_pad"]])
            logging.debug(f"[{evento}] Estacao={st_name}, lat={lat}, lon={lon}, arquivo={station_file}")

            # Lê OBS
            df_obs = read_obs_padronizadas(obs_dir, station_file, start_dt, end_dt)
            if df_obs is None:
                logging.debug(f"[{evento}] -> df_obs is None => sem obs? Pulando.")
            else:
                logging.debug(f"[{evento}] -> df_obs com {len(df_obs)} linhas")

            # Lê GPM
            df_gpm = extract_gpm_precip(lat, lon, start_dt, end_dt, gpm_dir)
            if df_gpm is None:
                logging.debug(f"[{evento}] -> df_gpm is None => sem GPM ou 0 regs.")
            else:
                logging.debug(f"[{evento}] -> df_gpm com {len(df_gpm)} linhas")

            # Loop domínios
            for dom in WRF_DOMAINS:
                df_wrf = extract_wrf_precip(lat, lon, dom, wrf_dir, start_dt, end_dt, wrf_year)
                if df_wrf is None:
                    logging.debug(f"[{evento}] -> df_wrf do {dom} é None => sem dados?")
                else:
                    logging.debug(f"[{evento}] -> df_wrf {dom} => {len(df_wrf)} linhas")

                metrics = calc_metrics_wrfgpm(df_wrf, df_obs, df_gpm)

                results_list.append({
                    "Evento":   evento,
                    "Dominio":  dom,
                    "Estacao":  st_name,
                    "Lat":      lat,
                    "Lon":      lon,
                    "MAE_wrfxobs":  metrics["MAE_wrfxobs"],
                    "RMSE_wrfxobs": metrics["RMSE_wrfxobs"],
                    "d_wrfxobs":    metrics["d_wrfxobs"],
                    "MAE_wrfxgpm":  metrics["MAE_wrfxgpm"],
                    "RMSE_wrfxgpm": metrics["RMSE_wrfxgpm"],
                    "d_wrfxgpm":    metrics["d_wrfxgpm"]
                })

    # Salva CSV final
    df_res = pd.DataFrame(results_list)
    out_csv = os.path.join(BASE_OUT, "metricas_consolidadas.csv")
    logging.info(f"[MAIN] Salvando métricas em => {out_csv}")
    df_res.to_csv(out_csv, index=False, float_format="%.3f")
    logging.info("[MAIN] Finalizado com sucesso!")

if __name__ == "__main__":
    main()

