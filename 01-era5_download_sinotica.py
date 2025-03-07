#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01-era5_download_sinotica.py

Objetivo:
  Baixar dados ERA5 em níveis de pressão (pressure-levels) 
  para análise sinótica:
    - 925, 850 e 500 hPa
    - Variáveis: U, V, T, Z, W, (opcional Vorticidade se quiser, "vo")
  em um intervalo de datas específico p/ cada evento:
    "NOV_2008":  2008-11-18 -> 2008-11-25
    "OUT_2021":  2021-10-10 -> 2021-10-13
    "NOV_2022":  2022-11-30 -> 2022-12-03

Salva em: /home/vinicius/Documentos/tcc/REANALISE/<EVENTO>/era5_PL_<EVENTO>.nc

Uso:
  python 01-era5_download_sinotica.py
"""

import os
import cdsapi
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1) Configuração de CLIENTE API (suas credenciais)
# ---------------------------------------------------------
# Se necessário, você pode configurar as credenciais em ~/.cdsapirc
# Exemplo:
# url: https://ads.atmosphere.copernicus.eu/api/v2
# key: 03ac90e2-6ad0-4cce-8c24-1e93f7a1b361:6a4e435d-6403-4137-a974-da4a7143a3d7
# 
# Ou defina manualmente:
# os.environ["CDSAPI_URL"] = "https://ads.atmosphere.copernicus.eu/api/v2"
# os.environ["CDSAPI_KEY"] = "03ac90e2-6ad0-4cce-8c24-1e93f7a1b361:6a4e435d-6403-4137-a974-da4a7143a3d7"

c = cdsapi.Client()

# ---------------------------------------------------------
# 2) Definir cada evento e seu período
# ---------------------------------------------------------
domains = {
    "NOV_2008":  {"start": "2008-11-18", "end": "2008-11-25"},
    "OUT_2021":  {"start": "2021-10-10", "end": "2021-10-13"},
    "NOV_2022":  {"start": "2022-11-30", "end": "2022-12-03"},
}

# ---------------------------------------------------------
# 3) Função para gerar lista de datas (YYYY-MM-DD)
# ---------------------------------------------------------
def generate_date_list(start_date_str, end_date_str):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    date_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%Y-%m-%d"))
        current_dt += timedelta(days=1)
    return date_list

# ---------------------------------------------------------
# 4) Variáveis e níveis de pressão de interesse
# ---------------------------------------------------------
# Ajuste se quiser "vo" (vorticity) ou "q" (specific humidity), etc.
press_levels = ["925", "850", "500"]

# Principais var. p/ sinótica: 
#  - U wind (u_component_of_wind)
#  - V wind (v_component_of_wind)
#  - T temperature
#  - Z geopotential
#  - W vertical velocity (omega) => "vertical_velocity"
#  - vo vorticity (opcional) => "vorticity"
variables = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "geopotential",
    "vertical_velocity",
    # "vorticity",  # se quiser baixar vorticidade pronta
]

# ---------------------------------------------------------
# 5) Área geográfica
# ---------------------------------------------------------
# Se quiser algo amplo p/ sinótica. Formato: [North, West, South, East]
# Observando q latitude diminui do norte p/ sul
# Exemplo p/ ver boa parte da América do Sul:
area = [
    10,  # North lat
    -90, # West lon
    -40, # South lat
    -30, # East lon
]

# ---------------------------------------------------------
# 6) Loop p/ cada simulacao
# ---------------------------------------------------------
def download_era5_pressure_levels(sim_name, start_day, end_day):
    """
    Baixa ERA5 pressure-levels e salva em /home/vinicius/Documentos/tcc/REANALISE/<sim_name>/
    como era5_PL_<sim_name>.nc
    """
    date_list = generate_date_list(start_day, end_day)

    # Montar lista de horas do dia (ex.: 00:00 a 23:00)
    time_list = [f"{hour:02d}:00" for hour in range(24)]  # se quiser hourly

    # Cria pasta de destino
    base_dir = "/home/vinicius/Documentos/tcc/REANALISE"
    domain_dir = os.path.join(base_dir, sim_name)
    os.makedirs(domain_dir, exist_ok=True)

    target_file = os.path.join(domain_dir, f"era5_PL_{sim_name}.nc")
    print(f"\n[ERA5-PL] Baixando p/ {sim_name}. Dados -> {target_file}")

    request_params = {
        "product_type": "reanalysis",
        "format": "netcdf",          # NetCDF
        "pressure_level": press_levels,
        "variable": variables,
        "year":  [],  # vamos preencher com years
        "month": [],  # e months
        "day":   [],  
        "time":  time_list,
        "area":  area,
    }

    # Precisamos separar year, month, day
    # Ex.: "2008-11-18" => year=2008, month=11, day=18
    # MAS a API do CDS também aceita "date" em formato "YYYY-MM-DD"? 
    # reanalysis-era5-pressure-levels aceita arrays year, month, day.
    # Então vamos popular request_params com year, month, day (sem "date").

    # Montar sets
    years  = sorted(list(set([d[:4]   for d in date_list])))
    months = sorted(list(set([d[5:7] for d in date_list])))
    days   = sorted(list(set([d[8:]  for d in date_list])))

    # MAS se formos nesse approach (year,month,day) => Ele vai baixar TUDO de cada dia?
    # A API do ECMWF não filtra "apenas 18 a 25" automaticamente se passarmos day=range(1..31).
    # Precisamos ou usar "date" no request OU baixamos Mês inteiro e filtrar local.
    # A approach + simples (economizar) => usar "date" e "time" no request:
    #   "date": date_list
    # Vamos então usar "date" e remover year/month/day do request:

    request_params2 = {
        "product_type":   "reanalysis",
        "format":         "netcdf",
        "pressure_level": press_levels,
        "variable":       variables,
        "date":           date_list,   # pass array of 'YYYY-MM-DD' strings
        "time":           time_list,
        "area":           area,
    }

    # Dataset = "reanalysis-era5-pressure-levels"
    # Faz a requisição
    dataset_name = "reanalysis-era5-pressure-levels"

    c.retrieve(dataset_name, request_params2, target_file)
    print(f"[OK] Download concluído: {target_file}")


if __name__ == "__main__":
    for sim, per in domains.items():
        st = per["start"]
        ed = per["end"]
        download_era5_pressure_levels(sim, st, ed)
        # repete p/ NOV_2008, OUT_2021, NOV_2022
    print("=== Todos os downloads concluídos ===")

