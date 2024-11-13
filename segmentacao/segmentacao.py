import os
import json
from PIL import Image, ImageDraw

# Diretórios
base_dir = "C:/git/image-segmentation/Dataset/nematode-detection-labels"
subdirs = ["Train", "Test", "Val"]
output_base_dir = "C:/git/image-segmentation/Dataset/nematode-detection-labels/Segmentadas"

def carregar_imagem(caminho_imagem):
    """Carrega a imagem original."""
    return Image.open(caminho_imagem)

def carregar_dados_json(caminho_json):
    """Carrega os dados do arquivo JSON."""
    with open(caminho_json, "r") as arquivo_json:
        return json.load(arquivo_json)

def criar_imagem_segmentada(imagem_original, dados_json, output_path, label, index, filename):
    """Cria uma nova imagem segmentada a partir dos dados JSON."""
    imagem_segmentada = Image.new("RGB", imagem_original.size)
    draw = ImageDraw.Draw(imagem_segmentada)

    for index, shape in enumerate(dados_json["shapes"]):
        pontos = shape["points"]
        label = shape["label"]
        pontos = [(int(x), int(y)) for x, y in pontos]
        draw.polygon(pontos, fill=(255, 255, 255)) 
        nome_imagem = filename.split(".")[0]
        imagem_segmentada.save(os.path.join(output_path, f"{nome_imagem}_{label}_{index}.png"))

def processar_pasta(subdir):
    input_dir = os.path.join(base_dir, subdir)
    output_dir = os.path.join(output_base_dir, subdir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            caminho_imagem = os.path.join(input_dir, filename)
            caminho_json = os.path.join(input_dir, filename.replace(".png", ".json"))

            if os.path.exists(caminho_json):
                imagem_original = carregar_imagem(caminho_imagem)
                dados_json = carregar_dados_json(caminho_json)
                criar_imagem_segmentada(imagem_original, dados_json, output_dir, filename.replace(".png", ""), 0, filename)

def main():
    for subdir in subdirs:
        print(f"Processando pasta {subdir}...")
        # se a pasta não existir, cria
        if not os.path.exists(os.path.join(output_base_dir, subdir)):
            os.makedirs(os.path.join(output_base_dir, subdir))
        # processa a pasta
        processar_pasta(subdir)

if __name__ == "__main__":
    main()