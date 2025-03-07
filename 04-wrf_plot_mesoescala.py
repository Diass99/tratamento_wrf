#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04-wrf_plot_mesoescala_NOV2022.py

Objetivo:
  - Ler arquivos WRF (ex.: wrfout_d01_...) apenas para um evento 
    e para cada domínio (ex.: d01, d02, d03).
  - Plotar:
      1) Orografia + Precipitação
      2) Vento (componentes zonal e meridional) interpolados para níveis 925, 850 e 500 hPa
      3) Omega e Umidade Específica (QVAPOR) interpolados para níveis 925, 850 e 500 hPa
  - Salvar as figuras em:
      /home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1/NOV_2022/WRF_MESO/<domínio>/

Uso:
  python 04-wrf_plot_mesoescala_NOV2022.py
"""

import os
import glob
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import logging
import gc
from netCDF4 import Dataset

# Importa funções do wrf-python
from wrf import getvar, interplevel, latlon_coords, get_cartopy, to_np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Diretórios base (ajuste conforme seu ambiente)
BASE_WRF = "/home/vinicius/Documentos/tcc/01-RODADAS"  
OUT_BASE = "/home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1"

# Neste script, processamos apenas um evento
events = ["OUT_2021"]
domains = ["d01", "d02", "d03"]   # Ajuste conforme os domínios disponíveis

# Níveis de pressão para interpolação
levels = [925, 850, 500]

def get_time_string(ncfile, timeidx):
    """
    Retorna uma string representativa do tempo, utilizando a variável 'Times'
    se disponível. Os espaços são substituídos por underlines para uso em nomes de arquivo.
    """
    if "Times" in ncfile.variables:
        time_bytes = ncfile.variables["Times"][timeidx, :].tobytes()
        time_str = time_bytes.decode("utf-8").strip()
        return time_str.replace(" ", "_")
    else:
        return f"t{timeidx:02d}"

def make_map(ax):
    """
    Configura o mapa com coastlines, fronteiras, estados e gridlines.
    Os rótulos são posicionados fora da área do mapa.
    """
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    ax.add_feature(cfeature.STATES, linestyle=":", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.x_inline = False
    gl.y_inline = False

def plot_orography_precip(ncfile, sim_name, domain, outdir, timeidx=0):
    """
    Plota orografia (contornos) + precipitação (sombreado) para um dado tempo (timeidx).
    A extensão do mapa é definida automaticamente a partir dos dados.
    A escala de cor para a precipitação é de 0 a 100 mm.
    """
    hgt    = getvar(ncfile, "HGT", timeidx=timeidx)
    rainc  = getvar(ncfile, "RAINC", timeidx=timeidx)
    rainnc = getvar(ncfile, "RAINNC", timeidx=timeidx)
    precip = rainc + rainnc

    lats, lons = latlon_coords(hgt)
    cart_proj = get_cartopy(hgt)
    time_str = get_time_string(ncfile, timeidx)
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1, projection=cart_proj)
    # Define a extensão do mapa a partir dos dados
    ax.set_extent([np.nanmin(to_np(lons)), np.nanmax(to_np(lons)),
                   np.nanmin(to_np(lats)), np.nanmax(to_np(lats))],
                  ccrs.PlateCarree())
    make_map(ax)
    
    pc = ax.pcolormesh(to_np(lons), to_np(lats), to_np(precip),
                       transform=ccrs.PlateCarree(),
                       shading="auto",
                       cmap="Blues",
                       vmin=0, vmax=100)
    cbar = plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Precipitação (mm)", fontsize=10)
    
    cs = ax.contour(to_np(lons), to_np(lats), to_np(hgt),
                    levels=np.arange(0,3000,200),
                    colors="black",
                    linewidths=0.8,
                    transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f m")
    
    ax.set_title(f"Orografia + Precipitação\n{sim_name} | {domain} | {time_str}", fontsize=12)
    figname = f"wrf_orog_prec_{sim_name}_{domain}_{time_str}.png"
    plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
    plt.close('all')

def plot_wind_component(ncfile, sim_name, domain, outdir, level_hpa, comp="zonal", timeidx=0):
    """
    Plota a componente do vento (zonal ou meridional) interpolada para um nível (level_hpa)
    e para um dado tempo (timeidx).
    """
    pressure = getvar(ncfile, "pressure", timeidx=timeidx)
    u_3d = getvar(ncfile, "ua", timeidx=timeidx, units="m s-1")
    v_3d = getvar(ncfile, "va", timeidx=timeidx, units="m s-1")
    u_lvl = interplevel(u_3d, pressure, level_hpa)
    v_lvl = interplevel(v_3d, pressure, level_hpa)
    
    if comp.lower() == "zonal":
        data2d = u_lvl
        cname = "U (m/s)"
    else:
        data2d = v_lvl
        cname = "V (m/s)"
    
    lats, lons = latlon_coords(data2d)
    cart_proj = get_cartopy(data2d)
    time_str = get_time_string(ncfile, timeidx)
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1, projection=cart_proj)
    make_map(ax)
    pc = ax.pcolormesh(to_np(lons), to_np(lats), to_np(data2d),
                       transform=ccrs.PlateCarree(),
                       shading="auto",
                       cmap="RdBu_r",
                       vmin=-10, vmax=10)
    plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.8, label=cname)
    ax.set_title(f"{comp.capitalize()} Wind @ {level_hpa} hPa\n{sim_name} | {domain} | {time_str}", fontsize=12)
    figname = f"wrf_{comp.lower()}_wind_{level_hpa}_{sim_name}_{domain}_{time_str}.png"
    plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
    plt.close('all')

def plot_omega_qv(ncfile, sim_name, domain, outdir, level_hpa, timeidx=0):
    """
    Plota Omega e Umidade Específica (QVAPOR) interpolados para um nível (level_hpa)
    e para um dado tempo (timeidx).  
    Omega é plotado com pcolormesh (multiplicado por 100 e com transparência) e
    a QVAPOR é apresentada por contornos. Cada variável recebe sua própria legenda.
    """
    pressure = getvar(ncfile, "pressure", timeidx=timeidx)
    try:
        w_3d = getvar(ncfile, "wa", timeidx=timeidx)
        w_lvl = interplevel(w_3d, pressure, level_hpa)
    except Exception as e:
        logging.warning(f"[{sim_name} | {domain}] Falha ao obter 'wa' em {level_hpa} hPa: {e}")
        w_lvl = None
    try:
        qv_3d = getvar(ncfile, "QVAPOR", timeidx=timeidx)  # Umidade específica (kg/kg)
        qv_lvl = interplevel(qv_3d, pressure, level_hpa)
    except Exception as e:
        logging.warning(f"[{sim_name} | {domain}] Falha ao obter 'QVAPOR' em {level_hpa} hPa: {e}")
        qv_lvl = None
    
    if (w_lvl is None) and (qv_lvl is None):
        logging.warning(f"[{sim_name} | {domain}] Nem 'wa' nem 'QVAPOR' disponíveis em {level_hpa} hPa. Pulando plot de Omega+QV!")
        return

    # Usa as coordenadas do que estiver disponível
    if w_lvl is not None:
        lats, lons = latlon_coords(w_lvl)
        cart_proj = get_cartopy(w_lvl)
    elif qv_lvl is not None:
        lats, lons = latlon_coords(qv_lvl)
        cart_proj = get_cartopy(qv_lvl)
    time_str = get_time_string(ncfile, timeidx)
    
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1, projection=cart_proj)
    make_map(ax)
    
    # Plot de Omega (multiplicado por 100 para melhor visualização)
    if w_lvl is not None:
        dataw = to_np(w_lvl) * 100
        pc = ax.pcolormesh(to_np(lons), to_np(lats), dataw,
                           transform=ccrs.PlateCarree(),
                           shading="auto",
                           cmap="RdBu_r",
                           vmin=-10, vmax=10,
                           alpha=0.7)
        cbar = plt.colorbar(pc, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
        cbar.set_label("Omega x100 (Pa/s)", fontsize=10)
    
    # Plot de umidade específica (QVAPOR) como contornos
    if qv_lvl is not None:
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        levels_qv = np.linspace(0, 0.02, 11)  # ajuste conforme os dados (kg/kg)
        cs = ax.contour(to_np(lons), to_np(lats), to_np(qv_lvl),
                        levels=levels_qv,
                        cmap="Greens",
                        linewidths=1.0,
                        transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
        norm_qv = colors.Normalize(vmin=0, vmax=0.02)
        sm = cm.ScalarMappable(norm=norm_qv, cmap="Greens")
        sm.set_array([])
        # Aqui ajustamos o pad para separar mais a legenda de QV da do Omega
        cbar_qv = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, pad=0.02)
        cbar_qv.set_label("Specific Humidity (kg/kg)", fontsize=10)
    
    ax.set_title(f"Omega & QV @ {level_hpa} hPa\n{sim_name} | {domain} | {time_str}", fontsize=12)
    figname = f"wrf_omega_qv_{level_hpa}_{sim_name}_{domain}_{time_str}.png"
    plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
    plt.close('all')


def main():
    # Processa somente o evento NOV_2022
    for sim in events:
        for domain in domains:
            outdir = os.path.join(OUT_BASE, sim, "WRF_MESO", domain)
            os.makedirs(outdir, exist_ok=True)
            wrf_files = sorted(glob.glob(f"{BASE_WRF}/{sim}/03.gerados/wrfout_{domain}_*"))
            if not wrf_files:
                logging.warning(f"[{sim} | {domain}] Não encontrou arquivos wrfout_{domain}_*. Pulando.")
                continue
            for wrf_file in wrf_files:
                try:
                    ncfile = Dataset(wrf_file)
                except Exception as e:
                    logging.error(f"[{sim} | {domain}] Erro ao abrir {wrf_file}: {e}")
                    continue
                # Determina o número de passos de tempo
                if "Times" in ncfile.variables:
                    nt = len(ncfile.variables["Times"])
                elif "Time" in ncfile.dimensions:
                    nt = len(ncfile.dimensions["Time"])
                else:
                    nt = 1
                logging.info(f"[{sim} | {domain}] Processando {wrf_file} com {nt} tempos.")
                for t_idx in range(nt):
                    logging.info(f"[{sim} | {domain}] Processando tempo index {t_idx}.")
                    plot_orography_precip(ncfile, sim, domain, outdir, timeidx=t_idx)
                    for lvl in levels:
                        plot_wind_component(ncfile, sim, domain, outdir, level_hpa=lvl, comp="zonal", timeidx=t_idx)
                        plot_wind_component(ncfile, sim, domain, outdir, level_hpa=lvl, comp="merid", timeidx=t_idx)
                        plot_omega_qv(ncfile, sim, domain, outdir, level_hpa=lvl, timeidx=t_idx)
                    plt.close('all')
                    # Chama o garbage collector para liberar memória
                    gc.collect()
                ncfile.close()
                logging.info(f"[{sim} | {domain}] Concluído processar {wrf_file}.")
            logging.info(f"[{sim} | {domain}] Concluído plots mesoescala para este domínio.")
        logging.info(f"[{sim}] Concluído todos os domínios para {sim}.")
    logging.info("=== Fim do script 04-wrf_plot_mesoescala. ===")

if __name__ == "__main__":
    main()

