#!/usr/bin/env python3

import os
from PIL import Image

def create_mosaic_from_pngs(directory, output_path, grid=(2, 3)):
    """
    Procura por arquivos PNG recursivamente nas subpastas de 'directory'
    e cria um mosaico 2x3 usando as primeiras 6 imagens encontradas.
    
    Parâmetros:
      directory: caminho base onde buscar as imagens.
      output_path: caminho completo para salvar o mosaico.
      grid: tupla com (número de linhas, número de colunas).
    """
    png_files = []
    # Percorre recursivamente o diretório
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    if len(png_files) < grid[0] * grid[1]:
        print(f"Em '{directory}' foram encontradas {len(png_files)} imagens. São necessárias pelo menos {grid[0]*grid[1]} imagens.")
        return

    # Ordena os arquivos (opcional)
    png_files.sort()

    # Seleciona as primeiras imagens para o mosaico
    selected_files = png_files[:grid[0] * grid[1]]
    
    # Abre as imagens
    images = [Image.open(f) for f in selected_files]
    
    # Define o tamanho de cada tile usando o tamanho da primeira imagem
    tile_width, tile_height = images[0].size

    # Cria a imagem em branco para o mosaico
    mosaic_width = grid[1] * tile_width
    mosaic_height = grid[0] * tile_height
    mosaic = Image.new('RGB', (mosaic_width, mosaic_height))
    
    # Cola as imagens na posição correta
    for idx, img in enumerate(images):
        # Se o tamanho da imagem for diferente, redimensiona para o tamanho do primeiro tile
        if img.size != (tile_width, tile_height):
            img = img.resize((tile_width, tile_height))
        row = idx // grid[1]
        col = idx % grid[1]
        x = col * tile_width
        y = row * tile_height
        mosaic.paste(img, (x, y))
    
    mosaic.save(output_path)
    print(f"Mosaico salvo em: {output_path}")

# Lista dos diretórios a processar
directories = [
    "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1/NOV_2022/ESTACOES/01_mosaico",
    "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1/OUT_2021/ESTACOES/01_mosaico",
    "/home/vinicius/Documentos/tcc/05-ANALISES_REFINAMENTO1/NOV_2008/ESTACOES/01_mosaico"
]

for dir_path in directories:
    # Define o caminho de saída para o mosaico (neste caso, será salvo dentro do próprio diretório)
    output_file = os.path.join(dir_path, "mosaico_3x2.png")
    create_mosaic_from_pngs(dir_path, output_file, grid=(3, 2))

