#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02-era5_calcula_campos_sinotica.py

Objetivo:
  - Ler o arquivo era5_PL_<evento>.nc (com variáveis: u, v, t, z, w, etc.)
    contendo dados em níveis de pressão.
  - Calcular os campos derivados:
      * Advecção de Temperatura: −(u ∂T/∂x + v ∂T/∂y)  [K/s]
      * Vorticidade Relativa: (∂v/∂x − ∂u/∂y)  [1/s]
      * (Renomear "w" para "omega" e ajustar atributos do geopotencial "z")
  - Salvar um novo NetCDF “era5_PL_<evento>_deriv.nc” contendo as variáveis originais 
    mais os campos derivados, facilitando a posterior visualização.

Uso:
  python 02-era5_calcula_campos_sinotica.py
"""

import os
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import logging

# Configuração do log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Diretório base onde estão os arquivos
BASE_DIR = "/home/vinicius/Documentos/tcc/REANALISE"

# Eventos (simulações)
events = ["NOV_2008", "OUT_2021", "NOV_2022"]

def get_time_and_level_dims(var):
    """
    Retorna os nomes das dimensões de tempo e nível para uma variável.
    Se a dimensão "time" não existir, assume "valid_time".
    Se a dimensão "level" não existir, assume "pressure_level".
    """
    dims = var.dims
    time_dim = "time" if "time" in dims else "valid_time"
    level_dim = "level" if "level" in dims else "pressure_level"
    return time_dim, level_dim

def calculate_advection_of_temperature(ds):
    """
    Calcula a advecção de temperatura: - (u * dT/dx + v * dT/dy)

    Parâmetros:
      ds : xarray.Dataset
           Dataset contendo as variáveis:
           - 't' : temperatura (K)
           - 'u' : componente zonal do vento (m/s)
           - 'v' : componente meridional do vento (m/s)

    Retorna:
      advec_da : xarray.DataArray
                 Campo da advecção da temperatura com unidades [K s^-1].
    """
    ds = ds.metpy.parse_cf()  # Converte metadados e associa unidades

    # Extração das variáveis (ajuste os nomes conforme seu arquivo ERA5)
    try:
        T = ds["t"]
    except KeyError:
        raise KeyError("Variável 't' (temperatura) não encontrada no dataset.")
    try:
        u = ds["u"]
    except KeyError:
        raise KeyError("Variável 'u' (componente zonal) não encontrada no dataset.")
    try:
        v = ds["v"]
    except KeyError:
        raise KeyError("Variável 'v' (componente meridional) não encontrada no dataset.")

    # Determina os nomes das dimensões de tempo e nível
    time_dim, level_dim = get_time_and_level_dims(T)

    # Reorganiza as dimensões: (tempo, nível, latitude, longitude)
    T = T.transpose(time_dim, level_dim, "latitude", "longitude")
    u = u.transpose(time_dim, level_dim, "latitude", "longitude")
    v = v.transpose(time_dim, level_dim, "latitude", "longitude")

    # Obter os arrays de latitude e longitude
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # Calcula os espaçamentos dx e dy (em metros)
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    # Cria um container para o resultado com a mesma forma de T
    advec_data = np.empty(T.shape, dtype=float)

    n_time, n_level = T.shape[0], T.shape[1]
    for t_idx in range(n_time):
        for lev_idx in range(n_level):
            # Seleciona as fatias 2D e converte para quantities (com unidades)
            T2d = T.isel(**{time_dim: t_idx, level_dim: lev_idx}).metpy.quantify()
            u2d = u.isel(**{time_dim: t_idx, level_dim: lev_idx}).metpy.quantify()
            v2d = v.isel(**{time_dim: t_idx, level_dim: lev_idx}).metpy.quantify()

            # Calcula a advecção; x é o último eixo (longitude) e y o penúltimo (latitude)
            adv = mpcalc.advection(T2d, u2d, v2d,
                                   dx=dx, dy=dy,
                                   x_dim=-1, y_dim=-2)
            # Aplica o sinal negativo e extrai os dados numéricos com .values
            advec_data[t_idx, lev_idx, :, :] = -adv.values

    advec_da = xr.DataArray(
        advec_data,
        coords=T.coords,
        dims=T.dims,
        name="advec_temp"
    )
    advec_da.attrs["units"] = "K s^-1"
    advec_da.attrs["long_name"] = "Advecção de Temperatura (K/s)"
    return advec_da

def calculate_vorticity(ds):
    """
    Calcula a vorticidade relativa: (∂v/∂x - ∂u/∂y)

    Parâmetros:
      ds : xarray.Dataset
           Dataset contendo as variáveis:
           - 'u' : componente zonal do vento (m/s)
           - 'v' : componente meridional do vento (m/s)

    Retorna:
      vort_da : xarray.DataArray
                Campo de vorticidade relativa com unidades [s^-1].
    """
    ds = ds.metpy.parse_cf()

    try:
        u = ds["u"]
    except KeyError:
        raise KeyError("Variável 'u' não encontrada no dataset.")
    try:
        v = ds["v"]
    except KeyError:
        raise KeyError("Variável 'v' não encontrada no dataset.")

    time_dim, level_dim = get_time_and_level_dims(u)

    u = u.transpose(time_dim, level_dim, "latitude", "longitude")
    v = v.transpose(time_dim, level_dim, "latitude", "longitude")

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    vort_data = np.empty(u.shape, dtype=float)
    n_time, n_level = u.shape[0], u.shape[1]
    for t_idx in range(n_time):
        for lev_idx in range(n_level):
            u2d = u.isel(**{time_dim: t_idx, level_dim: lev_idx}).metpy.quantify()
            v2d = v.isel(**{time_dim: t_idx, level_dim: lev_idx}).metpy.quantify()
            vort = mpcalc.vorticity(u2d, v2d,
                                    dx=dx, dy=dy,
                                    x_dim=-1, y_dim=-2)
            vort_data[t_idx, lev_idx, :, :] = vort.values

    vort_da = xr.DataArray(
        vort_data,
        coords=u.coords,
        dims=u.dims,
        name="vort"
    )
    vort_da.attrs["units"] = "s^-1"
    vort_da.attrs["long_name"] = "Vorticidade Relativa (1/s)"
    return vort_da

def process_dataset(sim_name):
    """
    Processa o dataset de um determinado evento:
      - Abre o arquivo era5_PL_<sim_name>.nc
      - Calcula advecção de temperatura e vorticidade
      - Renomeia 'w' para 'omega' (se existir) e ajusta atributos de 'z'
      - Salva o novo arquivo era5_PL_<sim_name>_deriv.nc com os campos derivados.
    """
    inpath = os.path.join(BASE_DIR, sim_name, f"era5_PL_{sim_name}.nc")
    outpath = os.path.join(BASE_DIR, sim_name, f"era5_PL_{sim_name}_deriv.nc")

    if not os.path.isfile(inpath):
        logging.warning(f"[{sim_name}] Arquivo não encontrado: {inpath}")
        return

    logging.info(f"[{sim_name}] Abrindo dataset: {inpath}")
    ds = xr.open_dataset(inpath)

    logging.info(f"[{sim_name}] Calculando advecção de temperatura...")
    advec_temp = calculate_advection_of_temperature(ds)

    logging.info(f"[{sim_name}] Calculando vorticidade...")
    vort = calculate_vorticity(ds)

    if "w" in ds.data_vars:
        ds = ds.rename({"w": "omega"})
        ds["omega"].attrs["long_name"] = "Velocidade vertical (omega)"

    if "z" in ds.data_vars:
        ds["z"].attrs["long_name"] = "Geopotencial"
        # Se necessário, converta z (por exemplo, para metros)

    ds_out = ds.copy()
    ds_out["advec_temp"] = advec_temp
    ds_out["vort"] = vort

    # Remover a variável 'metpy_crs' que pode ter sido adicionada automaticamente
    if "metpy_crs" in ds_out.variables:
        ds_out = ds_out.drop_vars("metpy_crs")

    logging.info(f"[{sim_name}] Salvando arquivo derivado em: {outpath}")
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds_out.data_vars}
    ds_out.to_netcdf(outpath, mode="w", format="NETCDF4", encoding=encoding)

    ds_out.close()
    ds.close()
    logging.info(f"[{sim_name}] Processamento concluído. Arquivo salvo: {outpath}")

if __name__ == "__main__":
    for sim in events:
        process_dataset(sim)
    logging.info("=== Fim do script 02-era5_calcula_campos_sinotica. ===")

