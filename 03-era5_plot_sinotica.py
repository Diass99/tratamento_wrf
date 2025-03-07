#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03-era5_plot_sinotica.py

Objetivo:
  - Ler era5_PL_<sim>_deriv.nc (com u, v, z, advec_temp, vort, omega).
  - Gerar mapas sinóticos de:
      1) Advecção de T @925
      2) Omega @925 e @500
      3) Geopotencial (z) @925 e @500
      4) Vorticidade @925, 850, 500
  - Salvar as figuras em: 
      /home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1/<sim>/SINOTICA_ERA5/

Uso:
  python 03-era5_plot_sinotica.py
"""

import os
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pasta onde estão os arquivos
BASE_DIR = "/home/vinicius/Documentos/tcc/REANALISE"

# Pasta de saída (raiz) para gravar as figuras
OUT_BASE = "/home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1"

# Simulações
events = ["NOV_2008", "OUT_2021", "NOV_2022"]

# Níveis que nos interessam
levels_to_plot = [925, 850, 500]

def get_coord_names(ds):
    """
    Retorna uma tupla (time_coord, level_coord) detectando as coordenadas
    de tempo e nível no dataset.
    Preferência: "time" e "level", mas se não existirem, usa "valid_time" e "pressure_level".
    """
    time_coord = "time" if "time" in ds.coords else "valid_time"
    level_coord = "level" if "level" in ds.coords else "pressure_level"
    if time_coord not in ds.coords:
        raise ValueError("Nenhuma coordenada de tempo encontrada no dataset.")
    if level_coord not in ds.coords:
        raise ValueError("Nenhuma coordenada de nível encontrada no dataset.")
    return time_coord, level_coord

def make_map(ax, lat_extent=None, lon_extent=None):
    """
    Configura aspectos do mapa: coastlines, fronteiras, gridlines.
    """
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    ax.add_feature(cfeature.STATES, linestyle=":", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    if lat_extent and lon_extent:
        ax.set_extent([lon_extent[0], lon_extent[1],
                       lat_extent[0], lat_extent[1]], ccrs.PlateCarree())

def plot_field(ax, lons, lats, data2d, cmap, vmin, vmax, cb_label=""):
    """
    Faz pcolormesh do campo 2D (data2d) em ax com PlateCarree, com barra de cor.
    """
    pc = ax.pcolormesh(lons, lats, data2d,
                       transform=ccrs.PlateCarree(),
                       shading="auto",
                       cmap=cmap,
                       vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pc, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label(cb_label, fontsize=10)
    return pc

def plot_contour(ax, lons, lats, data2d, levels, color="black", linewidths=1.0):
    """
    Faz contornos de data2d em ax, com transform PlateCarree.
    """
    cs = ax.contour(lons, lats, data2d,
                    levels=levels,
                    colors=color,
                    linewidths=linewidths,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    return cs

def plot_windbarb(ax, lons, lats, u2d, v2d, skip=10):
    """
    Plota barbs do vento.
    """
    ax.barbs(lons[::skip], lats[::skip], 
             u2d[::skip, ::skip], v2d[::skip, ::skip],
             length=5, transform=ccrs.PlateCarree(),
             linewidth=0.6, barb_increments=dict(half=2, full=4, flag=20))

def plot_geopotential(ax, lons, lats, z2d, levels, cmap="viridis", cb_label="Geopotential Height (m)"):
    """
    Plota o campo de geopotencial como um sombreador (contourf) com sobreposição de contornos.
    Os dados de z devem ser convertidos para metros; normalmente, z é dado em m²/s², 
    então a conversão é: geopot_height (m) = z / g, onde g=9.80665 m/s².
    Se desejar exibir em decâmetros, pode-se dividir ainda por 10.
    """
    # Converte para altura em metros (aqui optamos por metros)
    height = z2d / 9.80665  
    cf = ax.contourf(lons, lats, height,
                     levels=levels,
                     cmap=cmap,
                     transform=ccrs.PlateCarree(),
                     extend="both")
    # Sobrepõe os contornos com linhas finas
    cs = ax.contour(lons, lats, height,
                    levels=levels,
                    colors="black",
                    linewidths=0.8,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")
    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label(cb_label, fontsize=10)
    return cf, cs

def plot_sinotica(ds, time_idx, level_val, sim_name):
    """
    Gera os plots sinóticos:
      1) Advecção de T @925
      2) Vorticidade (@level_val)
      3) Geopotencial (sombreado com contornos) – para 925 e 500 (pode ser feito para qualquer nível)
      4) Omega (@925 ou @500, se disponível)
    """
    # Define diretório de saída para as figuras
    outdir = os.path.join(OUT_BASE, sim_name, "SINOTICA_ERA5")
    os.makedirs(outdir, exist_ok=True)

    # Obtém os nomes das coordenadas de tempo e nível
    time_coord, level_coord = get_coord_names(ds)

    # Obter data/hora para o nome do arquivo
    time_dt = ds[time_coord].isel({time_coord: time_idx}).values
    dt_str = str(np.datetime_as_string(time_dt, unit='h'))

    # Filtra o dataset para o nível desejado
    ds_lev = ds.sel({level_coord: level_val})

    # Extrair coordenadas lat/lon
    lat = ds_lev["latitude"].values
    lon = ds_lev["longitude"].values

    # Exemplo 1: Advecção de T @925 (apenas se level_val==925)
    if level_val == 925 and "advec_temp" in ds_lev.data_vars:
        data2d = ds_lev["advec_temp"].isel({time_coord: time_idx}).values
        fig = plt.figure(figsize=(9,7))
        ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
        make_map(ax)
        pc = plot_field(ax, lon, lat, data2d, cmap="coolwarm",
                        vmin=-0.001, vmax=0.001, cb_label="Advecção de T (K/s)")
        ax.set_title(f"Advec. Temp @925 hPa - {sim_name} - {dt_str}", fontsize=12)
        figname = f"era5_advecT_925_{dt_str}.png"
        plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
        plt.close()

    # Exemplo 2: Vorticidade (@level_val)
    if "vort" in ds_lev.data_vars:
        data2d = ds_lev["vort"].isel({time_coord: time_idx}).values
        fig = plt.figure(figsize=(9,7))
        ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
        make_map(ax)
        pc = plot_field(ax, lon, lat, data2d*1e5, cmap="bwr",
                        vmin=-2, vmax=2, cb_label="Vorticidade (1e-5 s⁻¹)")
        ax.set_title(f"Vorticidade @ {level_val} hPa - {sim_name} - {dt_str}", fontsize=12)
        figname = f"era5_vort_{level_val}_{dt_str}.png"
        plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
        plt.close()

    # Exemplo 3: Geopotencial – agora como campo sombreado com contornos
    if "z" in ds_lev.data_vars:
        z2d = ds_lev["z"].isel({time_coord: time_idx}).values
        # Definindo níveis de contorno; ajuste conforme necessário
        levels = np.arange(0, 6000, 100)
        fig = plt.figure(figsize=(9,7))
        ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
        make_map(ax)
        # Chama a função que plota o geopotencial sombreado com contornos
        cf, cs = plot_geopotential(ax, lon, lat, z2d, levels, cmap="viridis",
                                   cb_label="Altura Geopotencial (m)")
        ax.set_title(f"Altura Geopotencial @ {level_val} hPa - {sim_name} - {dt_str}", fontsize=12)
        figname = f"era5_z_{level_val}_{dt_str}.png"
        plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
        plt.close()

    # Exemplo 4: Omega – se level_val for 925 ou 500
    if "omega" in ds_lev.data_vars and level_val in [925, 500]:
        om2d = ds_lev["omega"].isel({time_coord: time_idx}).values
        fig = plt.figure(figsize=(9,7))
        ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
        make_map(ax)
        pc = plot_field(ax, lon, lat, om2d*100, cmap="RdBu_r",
                        vmin=-10, vmax=10, cb_label="Omega x100 (Pa/s)")
        ax.set_title(f"Omega @ {level_val} hPa - {sim_name} - {dt_str}", fontsize=12)
        figname = f"era5_omega_{level_val}_{dt_str}.png"
        plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
        plt.close()

def main():
    for sim in events:
        infile = os.path.join(BASE_DIR, sim, f"era5_PL_{sim}_deriv.nc")
        if not os.path.isfile(infile):
            logging.warning(f"[{sim}] Arquivo derivado não encontrado: {infile}")
            continue

        ds = xr.open_dataset(infile)
        time_coord, _ = get_coord_names(ds)
        n_times = ds.sizes[time_coord]
        logging.info(f"[{sim}] Dataset com {n_times} tempos – plotando todos os horários.")

        for t_idx in range(n_times):
            for lvl in levels_to_plot:
                _, level_coord = get_coord_names(ds)
                if lvl not in ds[level_coord].values:
                    logging.warning(f"[{sim}] Nível {lvl} hPa não existe no dataset. Pulando.")
                    continue
                plot_sinotica(ds, t_idx, lvl, sim)
        
        ds.close()
        logging.info(f"[{sim}] Plotagem sinótica finalizada.")

if __name__ == "__main__":
    main()
    logging.info("=== Fim do script 03-era5_plot_sinotica. ===")

