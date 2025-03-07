#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05-estacoes_map.py

Objetivo:
  - Ler metadados das estações (lat, lon, altitude, nome) de um CSV
  - Plotar um mapa simples com Cartopy, marcando a localização das estações
  - Salvar figura em /home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1/ESTACOES/mapa_estacoes.png

Uso:
  python 05-estacoes_map.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Defina o caminho para o seu CSV
CSV_META = "/home/vinicius/Documentos/tcc/03-ESTACOES/metadados_estacoes.csv"

# Pasta de saída para a figura
OUT_DIR = "/home/vinicius/Documentos/tcc/ANALISES_REFINAMENTO1/ESTACOES"

def make_map(ax, lat_extent=None, lon_extent=None):
    """
    Adiciona costas, fronteiras, estados.
    Se quiser recortar, use lat_extent=[-30, -25], lon_extent=[-52, -45].
    """
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, linestyle=":")
    if lat_extent and lon_extent:
        ax.set_extent([lon_extent[0], lon_extent[1],
                       lat_extent[0], lat_extent[1]], ccrs.PlateCarree())


def main():
    # Lê CSV
    # Formato esperado:
    #   NomeEstacao, Lat, Lon, Altitude
    #   epagri_florianopolis, -27.5814, -48.5072, 10
    # ...
    df = pd.read_csv(CSV_META, delimiter=",", header=0)
    # Ajuste se necessário: se colunas têm outro nome.

    # Cria pasta de saída
    os.makedirs(OUT_DIR, exist_ok=True)

    # Extração de colunas
    # Supondo colunas => "Nome", "Lat", "Long", "Altitude"
    # Mas nas anotações, estava no CSV:
    #   epagri_florianopolis  -27,5814 -48,5072 10
    # Precisamos checar se DF parse float com "," ou "." 
    # Caso tenha problema, define "decimal='.'" ou "decimal=','" no read_csv

    # Se o CSV usa ";" e vírgula nas coords, ajuste:
    # df = pd.read_csv(CSV_META, sep=";", decimal=",")
    # ou algo assim.

    # Verifique as colunas exatas. Supondo:
    #   df.columns => ["nome","Lat","Long","Altitude"]
    df.columns = ["nome","lat","lon","alt"]  # rename se quiser

    # Cria figura
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())

    # Zoom (opcional). Ex: focar SC:
    #   lat_extent=[-29.5, -25], lon_extent=[-53, -48]
    make_map(ax, lat_extent=[-30, -25], lon_extent=[-53, -47])

    # Plota cada estação
    for idx, row in df.iterrows():
        est_name = row["nome"]
        est_lat  = float(str(row["lat"]).replace(",", "."))  # se vier com vírgula
        est_lon  = float(str(row["lon"]).replace(",", "."))

        # Plot
        ax.plot(est_lon, est_lat, marker="o", color="red", markersize=5,
                transform=ccrs.PlateCarree())

        # Rótulo
        ax.text(est_lon+0.05, est_lat+0.05,
                est_name, fontsize=8, transform=ccrs.PlateCarree())

    plt.title("Localização das Estações Meteorológicas", fontsize=10)
    figname = "mapa_estacoes.png"
    plt.savefig(os.path.join(OUT_DIR, figname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa gerado: {os.path.join(OUT_DIR, figname)}")

if __name__ == "__main__":
    main()

