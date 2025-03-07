#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zonal_linhac_relevo_iterate.py

Objetivo:
  - Iterar sobre TODOS os eventos, TODOS os domínios e TODOS os tempos disponíveis.
  - Para cada conjunto, usar wrf-python (via netCDF4) para extrair:
       * Vento zonal (ua) e meridional (va)
       * Campo de pressão (pressure) para interpolações
       * Campo de orografia (HGT)
  - Para cada tempo, interpolar para um nível de pressão (ex.: 925 hPa) e gerar um plot que:
       - Plota o campo de vento zonal com pcolormesh e streamlines
       - Sobrepõe o relevo com um contourf em escala de cinza (alpha=0.35)
       - Ajusta os labels de coordenadas para que as longitudes fiquem na parte inferior
  - Salva cada figura na pasta:
       /home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1/<EVENTO>/WRF_MESO/<dominio>/zonal_linhac_relevo/

Uso:
  python zonal_linhac_relevo_iterate.py
"""

import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from netCDF4 import Dataset
from wrf import getvar, interplevel, latlon_coords, get_cartopy, to_np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parâmetros de entrada
BASE_WRF = "/home/vinicius/Documentos/tcc/01-RODADAS"
OUT_BASE = "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1"
EVENTOS  = ["NOV_2008", "OUT_2021", "NOV_2022"]
DOMINIOS = ["d01", "d02", "d03"]
LEVEL_HPA = 925  # Nível de pressão para interpolação

def make_map(ax):
    """Configura o mapa: costas, fronteiras, estados e gridlines.
    Força os labels a ficarem fora (não inline), garantindo que as longitudes apareçam na parte inferior."""
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    ax.add_feature(cfeature.STATES,  linestyle=":", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                      alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.x_inline = False
    gl.y_inline = False

def main():
    for evento in EVENTOS:
        path_evento = os.path.join(BASE_WRF, evento, "03.gerados")
        wrf_files = sorted(glob.glob(f"{path_evento}/wrfout_*"))
        if not wrf_files:
            logging.warning(f"[{evento}] Nenhum arquivo wrfout_* em {path_evento}")
            continue

        for dominio in DOMINIOS:
            dom_files = [f for f in wrf_files if f"wrfout_{dominio}_" in os.path.basename(f)]
            if not dom_files:
                logging.warning(f"[{evento}|{dominio}] Nenhum arquivo wrfout_{dominio}_* encontrado. Pulando.")
                continue

            outdir = os.path.join(OUT_BASE, evento, "WRF_MESO", dominio, "zonal_linhac_relevo")
            os.makedirs(outdir, exist_ok=True)

            # Abre os arquivos WRF via netCDF4 para o wrf-python
            ncfiles = [Dataset(ff) for ff in dom_files]

            try:
                # Extrai as variáveis necessárias de todos os tempos
                u_all = getvar(ncfiles, "ua", timeidx=None, units="m s-1")
                v_all = getvar(ncfiles, "va", timeidx=None, units="m s-1")
                p_all = getvar(ncfiles, "pressure", timeidx=None)
                hgt_all = getvar(ncfiles, "HGT", timeidx=None)
            except Exception as e:
                logging.error(f"[{evento}|{dominio}] Erro ao extrair variáveis: {e}")
                for nc in ncfiles:
                    nc.close()
                continue

            ntimes = u_all.sizes["Time"]
            logging.info(f"[{evento}|{dominio}] n_times = {ntimes}")

            for t in range(ntimes):
                try:
                    # Seleciona o tempo t
                    u_time = u_all.isel(Time=t)
                    v_time = v_all.isel(Time=t)
                    p_time = p_all.isel(Time=t)
                    hgt_time = hgt_all.isel(Time=t)
                except Exception as e_slice:
                    logging.error(f"[{evento}|{dominio}|time={t}] Erro ao selecionar fatia: {e_slice}")
                    continue

                try:
                    # Interpola para o nível desejado
                    u_lvl = interplevel(u_time, p_time, LEVEL_HPA)
                    v_lvl = interplevel(v_time, p_time, LEVEL_HPA)
                except Exception as e_interp:
                    logging.error(f"[{evento}|{dominio}|time={t}] Erro na interpolação para {LEVEL_HPA} hPa: {e_interp}")
                    continue

                # Extrai coordenadas e projeção
                lats, lons = latlon_coords(u_lvl)
                cart_proj = get_cartopy(u_lvl)
                time_str = str(u_lvl.Time.values)

                # Cria a figura
                fig = plt.figure(figsize=(8,6))
                ax = plt.subplot(1,1,1, projection=cart_proj)
                make_map(ax)

                # Plot do campo de vento zonal (U)
                data_u = to_np(u_lvl)
                pcm = ax.pcolormesh(to_np(lons), to_np(lats), data_u,
                                    transform=ccrs.PlateCarree(),
                                    cmap="RdBu_r", shading="auto", vmin=-10, vmax=10)
                cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
                cbar.set_label("Vento Zonal U (m/s)")

                # Plot das streamlines (linhas de corrente)
                data_v = to_np(v_lvl)
                ax.streamplot(to_np(lons), to_np(lats), data_u, data_v,
                              transform=ccrs.PlateCarree(),
                              density=2, color="black", linewidth=0.7, arrowsize=1)

                # Sobrepõe o relevo com um contourf: cmap "Greys" com alpha 0.35
                levels_relevo = np.linspace(np.nanmin(to_np(hgt_time)), np.nanmax(to_np(hgt_time)), 15)
                ax.contourf(to_np(lons), to_np(lats), to_np(hgt_time),
                            levels=levels_relevo, cmap="Greys", alpha=0.35,
                            transform=ccrs.PlateCarree())

                ax.set_title(f"{evento} | {dominio}\nU + Streamlines @ {LEVEL_HPA} hPa\nTempo: {time_str}", fontsize=11)

                # Prepara nome do arquivo removendo espaços e caracteres especiais
                time_tag = time_str.replace(" ", "_").replace(":", "")
                fname = f"zonal_linhac_relevo_{dominio}_{LEVEL_HPA}hPa_{time_tag}.png"
                plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)

            for nc in ncfiles:
                nc.close()
            logging.info(f"[{evento}|{dominio}] Imagens geradas em zonal_linhac_relevo.")

    logging.info("=== Fim do script zonal_linhac_relevo_iterate.py ===")

if __name__ == "__main__":
    main()

