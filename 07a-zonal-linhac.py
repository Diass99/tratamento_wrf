#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from netCDF4 import Dataset
from wrf import getvar, interplevel, latlon_coords, get_cartopy, to_np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_WRF   = "/home/vinicius/Documentos/tcc/01-RODADAS"
OUT_BASE   = "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1"
EVENTOS    = ["NOV_2008"]
DOMINIOS   = ["d01","d02","d03"]
LEVELS_HPA = [925, 850, 500]

def make_map(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=":")
    ax.add_feature(cfeature.STATES, linewidth=0.6, linestyle=":")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

def main():
    for evento in EVENTOS:
        path_evento = os.path.join(BASE_WRF, evento, "03.gerados")
        wrf_files   = sorted(glob.glob(f"{path_evento}/wrfout_*"))
        if not wrf_files:
            logging.warning(f"[{evento}] Sem arquivos wrfout_* em {path_evento}")
            continue

        for dominio in DOMINIOS:
            dom_files = [f for f in wrf_files if f"wrfout_{dominio}_" in os.path.basename(f)]
            if not dom_files:
                logging.warning(f"[{evento}|{dominio}] sem wrfout_{dominio}_*. Pulando.")
                continue

            # Monta a pasta de saída
            outdir = os.path.join(OUT_BASE, evento, "WRF_MESO", dominio, "zonal_linhac")
            os.makedirs(outdir, exist_ok=True)

            # Cria lista de netCDF4 para wrf-python
            ncfiles = [Dataset(ff) for ff in dom_files]

            # Lê TUDO de uma vez (todos os tempos concatenados):
            try:
                u_3d = getvar(ncfiles, "ua", timeidx=None, units="m s-1")
                v_3d = getvar(ncfiles, "va", timeidx=None, units="m s-1")
                p_3d = getvar(ncfiles, "pressure", timeidx=None)
            except Exception as e:
                logging.error(f"[{evento}|{dominio}] Erro ao ler ua/va/pressure: {e}")
                continue

            # Dimensão de tempo é “Time”
            ntimes = u_3d.sizes["Time"]
            logging.info(f"[{evento}|{dominio}] n_times = {ntimes}")

            for t in range(ntimes):
                for level_hpa in LEVELS_HPA:
                    # Interpola U e V no nível de pressão
                    u_lvl = interplevel(u_3d.isel(Time=t), p_3d.isel(Time=t), level_hpa)
                    v_lvl = interplevel(v_3d.isel(Time=t), p_3d.isel(Time=t), level_hpa)

                    # Lê coords
                    lats, lons = latlon_coords(u_lvl)
                    cart_proj  = get_cartopy(u_lvl)

                    # Nome do tempo
                    # wrf-python DataArray tem atributo “time” no item
                    time_str = str(u_lvl.Time.values)  # ou u_3d.Time.values[t]

                    # Monta figure
                    fig = plt.figure(figsize=(8,6))
                    ax  = plt.subplot(1,1,1, projection=cart_proj)
                    make_map(ax)

                    data_u = to_np(u_lvl)
                    data_v = to_np(v_lvl)

                    # Color de U
                    pcm = ax.pcolormesh(to_np(lons), to_np(lats), data_u,
                                        transform=ccrs.PlateCarree(),
                                        cmap="RdBu_r", shading="auto",
                                        vmin=-10, vmax=10)
                    cbar = plt.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
                    cbar.set_label("Vento Zonal U (m/s)")

                    # Streamlines
                    ax.streamplot(
                        to_np(lons), to_np(lats), data_u, data_v,
                        transform=ccrs.PlateCarree(),
                        density=2, color="black", linewidth=0.7, arrowsize=1
                    )

                    ax.set_title(f"{evento} {dominio} - T:{time_str}\nU + Linhas de Corrente @ {level_hpa} hPa")
                    figname = f"zonal_linhac_{level_hpa}hPa_t{t:03d}.png"
                    plt.savefig(os.path.join(outdir, figname), dpi=150, bbox_inches="tight")
                    plt.close(fig)

            # Fecha todos os netCDF
            for nc in ncfiles:
                nc.close()
            logging.info(f"[{evento}|{dominio}] Finalizado com sucesso")

    logging.info("=== Fim do script zonal_linhac ===")

if __name__ == "__main__":
    main()

