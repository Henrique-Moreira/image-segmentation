import os
import json
import base64
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Diretórios
input_base_dir = "C:/git/image-segmentation/dataset/base"
subfolders = ["Train", "Test", "Val"]
output_base_dir = "C:/git/image-segmentation/dataset/base_augmented"
extensao_imagem = ".JPG"
extensao_JSON = ".json"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def zoom_image(image, zoom_factor):
    height, width = image.shape[:2]
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
    image_zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    if zoom_factor > 1:
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        return image_zoomed[start_y:start_y+height, start_x:start_x+width]
    else:
        border_x = (width - new_width) // 2
        border_y = (height - new_height) // 2
        return cv2.copyMakeBorder(image_zoomed, border_y, border_y, border_x, border_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

def mirror_image(image):
    return cv2.flip(image, 1)

def mirror_points(points, image_width):
    return [[image_width - x, y] for [x, y] in points]

def rotate_points(points, angle, image_shape):
    (h, w) = image_shape[:2]
    center = (w // 2, h // 2)
    angle_rad = np.deg2rad(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotated_points = []
    for (x, y) in points:
        x_new = cos_angle * (x - center[0]) - sin_angle * (y - center[1]) + center[0]
        y_new = sin_angle * (x - center[0]) + cos_angle * (y - center[1]) + center[1]
        rotated_points.append([x_new, y_new])
    return rotated_points

def zoom_points(points, zoom_factor, image_shape):
    (h, w) = image_shape[:2]
    new_points = []
    for (x, y) in points:
        x_new = x * zoom_factor
        y_new = y * zoom_factor
        if zoom_factor > 1:
            x_new -= (w * (zoom_factor - 1)) / 2
            y_new -= (h * (zoom_factor - 1)) / 2
        else:
            x_new += (w * (1 - zoom_factor)) / 2
            y_new += (h * (1 - zoom_factor)) / 2
        new_points.append([x_new, y_new])
    return new_points

def augment_data(input_base_dir, output_base_dir):
    create_dir(output_base_dir)
    
    for subfolder in subfolders:
        input_dir = os.path.join(input_base_dir, subfolder)
        output_dir = os.path.join(output_base_dir, subfolder)
        create_dir(output_dir)

        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)

                    base_name, ext = os.path.splitext(file)
                    img_shape = img.shape

                    json_path = os.path.join(root, base_name + '.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as json_file:
                            annotations = json.load(json_file)

                    mirrored_img = mirror_image(img)
                    mirrored_json = annotations.copy()
                    for shape in mirrored_json['shapes']:
                        shape['points'] = mirror_points(shape['points'], img_shape[1])
                    cv2.imwrite(os.path.join(output_dir, base_name + '_mirror' + ext), mirrored_img)
                    
                    with open(os.path.join(output_dir, base_name + '_mirror.json'), 'w') as json_file:
                        json.dump(mirrored_json, json_file, indent=4)

                    rotated_img = rotate_image(img, 90)
                    rotated_json = annotations.copy()
                    for shape in rotated_json['shapes']:
                        shape['points'] = rotate_points(shape['points'], 90, img_shape)
                    cv2.imwrite(os.path.join(output_dir, base_name + '_rotate' + ext), rotated_img)
                    with open(os.path.join(output_dir, base_name + '_rotate.json'), 'w') as json_file:
                        json.dump(rotated_json, json_file, indent=4)

                    zoomed_img = zoom_image(img, 1.2)
                    zoomed_json = annotations.copy()
                    for shape in zoomed_json['shapes']:
                        shape['points'] = zoom_points(shape['points'], 1.2, img_shape)
                    cv2.imwrite(os.path.join(output_dir, base_name + '_zoom' + ext), zoomed_img)
                    with open(os.path.join(output_dir, base_name + '_zoom.json'), 'w') as json_file:
                        json.dump(zoomed_json, json_file, indent=4)

def carregar_imagem(caminho_imagem):
    """Carrega a imagem original."""
    return Image.open(caminho_imagem)

def carregar_dados_json(caminho_json):
    """Carrega os dados do arquivo JSON."""
    with open(caminho_json, "r") as arquivo_json:
        return json.load(arquivo_json)

def salvar_imagem_segmentada(imagem_segmentada, output_path, nome_imagem, label, index):
    """Salva a imagem segmentada e retorna o caminho da imagem salva."""
    caminho_imagem = os.path.join(output_path, f"{nome_imagem}_{label}_{index}{extensao_imagem}")
    imagem_segmentada.save(caminho_imagem)
    print(f"Imagem salva em {caminho_imagem}")
    return caminho_imagem

def salvar_anotacoes_json(output_path, nome_imagem, label, index, novas_anotacoes, imagem_original, image_data):
    """Salva as novas anotações em um novo arquivo JSON."""
    novo_json_path = os.path.join(output_path, f"{nome_imagem}_{label}_{index}{extensao_JSON}")
    novo_json_conteudo = {
        "version": "4.6.0",
        "flags": {},
        "shapes": novas_anotacoes,
        "imagePath": f"{nome_imagem}_{label}_{index}{extensao_imagem}",
        "imageData": image_data,
        "imageHeight": imagem_original.height,
        "imageWidth": imagem_original.width
    }
    with open(novo_json_path, "w") as novo_json_file:
        json.dump(novo_json_conteudo, novo_json_file, indent=4)

def criar_imagem_segmentada(imagem_original, dados_json, output_path, label, index, filename):
    """Cria uma nova imagem segmentada a partir dos dados JSON e salva a anotação em um novo arquivo JSON."""
    imagem_segmentada = Image.new("RGB", imagem_original.size)
    draw = ImageDraw.Draw(imagem_segmentada)
    novas_anotacoes = []

    for shape in dados_json["shapes"]:
        pontos = shape["points"]
        label = shape["label"]
        pontos = [(int(x), int(y)) for x, y in pontos]
        draw.polygon(pontos, fill=(255, 255, 255)) 
        novas_anotacoes.append({
            "label": label,
            "points": pontos,
            "group_id": shape.get("group_id"),
            "shape_type": shape.get("shape_type", "polygon"),
            "flags": shape.get("flags", {})
        })

    nome_imagem = filename.split(".")[0]
    caminho_imagem_segmentada = salvar_imagem_segmentada(imagem_segmentada, output_path, nome_imagem, label, index)

    # Codifica a imagem em base64
    with open(caminho_imagem_segmentada, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

    salvar_anotacoes_json(output_path, nome_imagem, label, index, novas_anotacoes, imagem_original, image_data)

def aplicar_data_augmentation(imagem, dados_json, output_path, filename):
    """Aplica técnicas de data augmentation e salva as novas imagens e anotações."""
    augmentations = [
        ("original", imagem),
        ("rotacao_90", imagem.rotate(90)),
        ("rotacao_180", imagem.rotate(180)),
        ("rotacao_270", imagem.rotate(270)),
        ("espelhamento_horizontal", ImageOps.mirror(imagem)),
        ("espelhamento_vertical", ImageOps.flip(imagem))
    ]
    
    for index, (label, imagem_aug) in enumerate(augmentations):
        criar_imagem_segmentada(imagem_aug, dados_json, output_path, label, index, filename)

def processar_pasta(subdir):
    input_dir = os.path.join(input_base_dir, subdir)
    output_dir = os.path.join(output_base_dir, subdir)
    create_dir(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(extensao_imagem):
            caminho_imagem = os.path.join(input_dir, filename)
            caminho_json = os.path.join(input_dir, filename.replace(extensao_imagem, extensao_JSON))

            if os.path.exists(caminho_json):
                imagem_original = carregar_imagem(caminho_imagem)
                dados_json = carregar_dados_json(caminho_json)
                aplicar_data_augmentation(imagem_original, dados_json, output_dir, filename)

def main():
    augment_data(input_base_dir, output_base_dir)
'''
    for subdir in subfolders:
        print(f"Processando pasta {subdir}...")
        if not os.path.exists(os.path.join(output_base_dir, subdir)):
            os.makedirs(os.path.join(output_base_dir, subdir))
        processar_pasta(subdir)
'''
if __name__ == "__main__":
    main()