import os
import json
from PIL import Image, ImageDraw

# Diretórios
base_dir = "C:/git/image-segmentation/Dataset/mamoeiro"
subdirs = ["Train", "Test", "Val"]
output_base_dir = "C:/git/image-segmentation/Dataset/mamoeiro/Segmentadas"
extensao_imagem = ".JPG"
extensao_JSON = ".json"

def carregar_imagem(caminho_imagem):
    """Carrega a imagem original."""
    return Image.open(caminho_imagem)

def carregar_dados_json(caminho_json):
    """Carrega os dados do arquivo JSON."""
    with open(caminho_json, "r") as arquivo_json:
        return json.load(arquivo_json)

def criar_imagem_segmentada(imagem_original, dados_json, output_path, label, index, filename):
    """Cria uma nova imagem segmentada a partir dos dados JSON e salva a anotação em um novo arquivo JSON."""
    imagem_segmentada = Image.new("RGB", imagem_original.size)
    draw = ImageDraw.Draw(imagem_segmentada)
    novas_anotacoes = []

    for index, shape in enumerate(dados_json["shapes"]):
        pontos = shape["points"]
        label = shape["label"]
        pontos = [(int(x), int(y)) for x, y in pontos]
        draw.polygon(pontos, fill=(255, 255, 255)) 
        nome_imagem = filename.split(".")[0]
        imagem_segmentada.save(os.path.join(output_path, f"{nome_imagem}_{label}_{index}{extensao_imagem}"))
        
        # Adiciona a nova anotação
        novas_anotacoes.append({
            "label": label,
            "points": pontos,
            "group_id": shape.get("group_id"),
            "shape_type": shape.get("shape_type", "polygon"),
            "flags": shape.get("flags", {})
        })

    # Salva as novas anotações em um novo arquivo JSON
    novo_json_path = os.path.join(output_path, f"{nome_imagem}_{label}_{index}{extensao_JSON}")
    with open(novo_json_path, "w") as novo_json_file:
        json.dump({"shapes": novas_anotacoes}, novo_json_file, indent=4)

def processar_pasta(subdir):
    input_dir = os.path.join(base_dir, subdir)
    output_dir = os.path.join(output_base_dir, subdir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(extensao_imagem):
            caminho_imagem = os.path.join(input_dir, filename)
            caminho_json = os.path.join(input_dir, filename.replace(extensao_imagem, extensao_JSON))

            if os.path.exists(caminho_json):
                imagem_original = carregar_imagem(caminho_imagem)
                dados_json = carregar_dados_json(caminho_json)
                criar_imagem_segmentada(imagem_original, dados_json, output_dir, filename.replace(extensao_imagem, ""), 0, filename)

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