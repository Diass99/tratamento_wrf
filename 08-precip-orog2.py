#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08-precip-orog2_relief.py

Objetivo:
  - Ler arquivos WRF (wrfout) de cada evento/domínio via netCDF4 + wrf-python
  - Plotar orografia + precipitação, usando contornos de 500 m na orografia
    e sobrepor de forma sutil o relevo (contourf cinza com alpha=0.35)
  - Salvar em /home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1/<EVENTO>/WRF_MESO/<dominio>/orog_prec_500/

Uso:
  python 08-precip-orog2_relief.py
"""

import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from netCDF4 import Dataset
from wrf import getvar, latlon_coords, get_cartopy, to_np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Caminhos base (ajuste se necessário)
BASE_WRF = "/home/vinicius/Documentos/tcc/01-RODADAS"
OUT_BASE = "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1"

# Quais eventos e domínios serão processados
EVENTOS  = ["NOV_2008", "OUT_2021", "NOV_2022"]
DOMINIOS = ["d01", "d02", "d03"]

def make_map(ax):
    """Configurações de costas, fronteiras, estados e gridlines."""
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
    ax.add_feature(cfeature.STATES,  linestyle=":", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

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
                logging.warning(f"[{evento}|{dominio}] Sem wrfout_{dominio}_*. Pulando.")
                continue
            
            # Lista de netCDF4.Dataset para wrf-python
            ncfiles = [Dataset(f) for f in dom_files]

            # Extrai variáveis de relevo e precipitação para todos os tempos
            try:
                hgt_3d    = getvar(ncfiles, "HGT",    timeidx=None)  # Orfografia
                rainc_3d  = getvar(ncfiles, "RAINC",  timeidx=None)
                rainnc_3d = getvar(ncfiles, "RAINNC", timeidx=None)
            except Exception as e:
                logging.error(f"[{evento}|{dominio}] Erro getvar: {e}")
                for nc in ncfiles:
                    nc.close()
                continue
            
            # Número de tempos
            ntimes = hgt_3d.sizes["Time"]
            logging.info(f"[{evento}|{dominio}] n_times = {ntimes}")
            
            # Pasta de saída
            outdir = os.path.join(OUT_BASE, evento, "WRF_MESO", dominio, "orog_prec_500")
            os.makedirs(outdir, exist_ok=True)

            # Loop pelos tempos
            for t in range(ntimes):
                try:
                    hgt_2d    = hgt_3d.isel(Time=t)
                    rainc_2d  = rainc_3d.isel(Time=t)
                    rainnc_2d = rainnc_3d.isel(Time=t)
                except Exception as e2:
                    logging.error(f"[{evento}|{dominio}|time={t}] Erro ao selecionar slice: {e2}")
                    continue

                # Precip total acumulada
                precip_2d = rainc_2d + rainnc_2d

                # Extrai lat/lon e projeção
                lats, lons = latlon_coords(hgt_2d)
                cart_proj = get_cartopy(hgt_2d)

                # Rótulo de tempo (geralmente atribuído pelo wrf-python)
                time_str = str(hgt_2d.Time.values)

                # Figura
                fig = plt.figure(figsize=(8,6))
                ax = plt.subplot(1,1,1, projection=cart_proj)
                make_map(ax)

                # Plot da precipitação
                pc = ax.pcolormesh(
                    to_np(lons), to_np(lats), to_np(precip_2d),
                    transform=ccrs.PlateCarree(), shading="auto",
                    cmap="Blues", vmin=0, vmax=100
                )
                cbar = plt.colorbar(pc, ax=ax, orientation="vertical", pad=0.03, shrink=0.8)
                cbar.set_label("Precipitação (mm) [acum.]")

                # Sobrepõe o relevo com um contourf:
                # Usa 15 níveis entre o mínimo e máximo de HGT e alpha=0.35 para menos transparência
                levels_relevo = np.linspace(np.nanmin(to_np(hgt_2d)), np.nanmax(to_np(hgt_2d)), 15)
                cf = ax.contourf(to_np(lons), to_np(lats), to_np(hgt_2d),
                                 levels=levels_relevo, cmap="Greys", alpha=0.35,
                                 transform=ccrs.PlateCarree())

                ax.set_title(f"{evento} - {dominio} - {time_str}\nOrografia (contornos 500m) + Precip", fontsize=11)

                # Nome do arquivo (remove espaços e caracteres especiais)
                time_tag = time_str.replace(" ","_").replace(":","")
                figname = f"orog_prec_500_{time_tag}.png"
                plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
                plt.close(fig)

            for nc in ncfiles:
                nc.close()
            logging.info(f"[{evento}|{dominio}] Concluído orog_prec_500 com relevo.")

    logging.info("=== Fim do script 08-precip-orog2_relief.py ===")

if __name__ == "__main__":
    main()

