# # Importe os módulos necessários
import glob
import json
import os
import os.path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import logging

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datetime import datetime

# Configuração do logger
log_dir = r'C:\git\image-segmentation\results\linknet-dataset-segmentadas'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
filenamelog = 'linknet-dataset-base-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.log'
logging.basicConfig(filename=osp.join(log_dir, filenamelog), level=logging.INFO, format='%(asctime)s - %(message)s')

# Train
matplotlib.use('agg')

# # Declaração de Variáveis
# CUDA:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Dispositivo: {device}")

# Caminho do diretório Dataset:
directory = r'C:\git\image-segmentation\dataset'
logging.info(f'Diretório do Projeto {directory}.')

if not os.path.exists(directory):
    os.makedirs(directory)
img_folder_val = directory + r'\\base_segmentadas\\Val'
img_folder_train = directory + r'\\base_segmentadas\\Train'
img_folder_test = directory + r'\\base_segmentadas\\Test'
save_dir = directory + r'\\result_linknet_base_segmentadas\\'
if not os.path.exists(img_folder_val):
    os.makedirs(img_folder_val)
if not os.path.exists(img_folder_train):
    os.makedirs(img_folder_train)
if not os.path.exists(img_folder_test):
    os.makedirs(img_folder_test)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
## Imagens Segmentadas
img_folder_train_segmentadas = directory + r'\\segmentadas_linknet\\train\\'
img_folder_val_segmentadas = directory + r'\\segmentadas_linknet\\val\\'
img_folder_test_segmentadas = directory + r'\\segmentadas_linknet\\test\\'
if not os.path.exists(img_folder_train_segmentadas):
    os.makedirs(img_folder_train_segmentadas)
if not os.path.exists(img_folder_val_segmentadas):
    os.makedirs(img_folder_val_segmentadas)
if not os.path.exists(img_folder_test_segmentadas):
    os.makedirs(img_folder_test_segmentadas)

# Local onde o Modelo será salvo
model_file_name = save_dir + 'model_linknet_segmentadas.pth'

# Configurações do treinamento
#Width x Height 
#resolution = (640, 480)
# resolution_input = (800, 448)
#resolution = (1600, 896)
#resolution = (2400, 1344)
#resolution = (3200, 1792)
#resolution = (4000, 2240)

# Inicialize labels e color_label com as dimensões corretas
labels = np.zeros((448, 800))  # Ajuste as dimensões para corresponder à saída do modelo
color_label = np.zeros((448, 800, 3))  # Ajuste as dimensões para corresponder à saída do modelo

resolution_input = (800, 448)  # Tamanho de entrada
logging.info(f'Resolução de Entrada: {resolution_input}.')
assert resolution_input[0] % 32 == 0 and resolution_input[1] % 32 == 0, "A resolução de entrada deve ser divisível por 32."
dummy_input = torch.randn(1, 3, 448, 800).to(device) 

patience = 30
logging.info(f'Patience: {patience}.')
plot_val = True
plot_train = True
max_epochs = 300
logging.info(f'Número Máximo de Épocas: {max_epochs}.')

# Mapeamento de classes e cores
class_to_color = {'Doenca': (255, 0, 0), 'Solo': (0, 0, 255), 'Saudavel': (0, 255, 255), 'Folhas': (0, 255, 0)}
logging.info(f'Mapeamento de Classes e Cores: {class_to_color}.')
class_to_id = {'Doenca': 0, 'Solo': 1, 'Saudavel': 2, 'Folhas': 3}
logging.info(f'Mapeamento de Classes e IDs: {class_to_id}.')
num_classes = len(class_to_id)
class_weights = [1.0, 2.0, 3.0, 4.0]
logging.info(f'Pesos de Classes: {class_weights}.')
class_weights = torch.tensor(class_weights).to(device)  

id_to_class = {v: k for k, v in class_to_id.items()}
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# # Clase para Segmentação de Dataset
# 
#     Este arquivo provavelmente contém funções utilitárias para manipular e preparar os dados para o treinamento do modelo. As utilidades de dados aqui podem incluir:
#         - Carregamento de dados de diferentes fontes (arquivos, bancos de dados, APIs).
#         - Limpeza e pré-processamento de dados (por exemplo, lidar com valores ausentes, normalização, conversão de tipos de dados).
#         - Aumento de dados para aumentar o tamanho do conjunto de dados de treinamento.
#         - Divisão dos dados em conjuntos de treinamento, validação e teste.


class SegmentationDataset(Dataset):
    """Segmentation dataset loader."""

    def __init__(self, json_folder, img_folder, is_train, class_to_id, resolution_input = (640, 480), augmentation = False, transform=None):
    #def __init__(self, json_folder, img_folder, is_train, class_to_id, resolution_input = (1280, 960), augmentation = False, transform=None):
        """
        Args:
            json_folder (str): Path to folder that contains the annotations.
            img_folder (str): Path to all images.
            is_train (bool): Is this a training dataset ?
            augmentation (bool): Do dataset augmentation (crete artificial variance) ?
        """

        self.gt_file_list = glob.glob(osp.join(json_folder, '*.json'))

        self.total_samples = len(self.gt_file_list)
        self.img_folder = img_folder
        self.is_train = is_train
        self.transform = transform
        self.augmentation = augmentation
        self.resolution = resolution_input
        self.class_to_id = class_to_id
        
        
        # Mean and std are needed because we start from a pre trained net
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        gt_file = self.gt_file_list[idx]
        img_number_str = gt_file.split('.')[0].split('/')[-1]
        
        # Abre Json
        gt_json = json.load(open(gt_file, 'r'))
        
        # Abre imagem
        img_np = cv2.imread(osp.join(self.img_folder, img_number_str + '.JPG'), cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
        original_shape = img_np.shape
        
        # Redimensiona a imagem
        img_np = cv2.resize(img_np, (self.resolution[0], self.resolution[1]))[..., ::-1]
        img_np = np.ascontiguousarray(img_np)
        
        # Cria imagem zerada para os rótulos
        label_np = np.zeros((img_np.shape[0], img_np.shape[1]))
        label_np[...] = -1
        
        # Para todos os polígonos
        for shape in gt_json['shapes']:
            # Transforma os pontos do polígono em array
            points_np = np.array(shape['points'], dtype=np.float64)
            
            # Ajusta os pontos porque eu mudo a resolução
            points_np[:, 0] *= self.resolution[0] / original_shape[1]
            points_np[:, 1] *= self.resolution[1] / original_shape[0]
            
            # As coordenadas dos pontos que formam o polígono têm que ser inteiros
            points_np = np.round(points_np).astype(np.int64)
            
            # Coloca os pontos no formato certo para o OpenCV
            points_np = points_np.reshape((-1, 1, 2))
            
            # Pinta o polígono usando o OpenCV com o valor referente ao rótulo
            label_np = cv2.fillPoly(label_np, [points_np], self.class_to_id[shape['label']])
        
        # Transforma o GT em inteiro
        label_np = label_np.astype(np.int32)
        
        # Aumento de dados (opcional)
        if self.is_train and self.augmentation:
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np)
                label_np = np.fliplr(label_np)
                img_np = np.ascontiguousarray(img_np)
                label_np = np.ascontiguousarray(label_np)
                
        # Normalização da imagem
        img_pt = img_np.astype(np.float32) / 255.0
        for i in range(3):
            img_pt[..., i] -= self.mean[i]
            img_pt[..., i] /= self.std[i]
        
        # Transposição para formato de tensor
        img_pt = img_pt.transpose(2, 0, 1)
        img_pt = torch.from_numpy(img_pt)
        label_pt = torch.from_numpy(label_np).long()
        
        return {'image': img_pt, 'gt': label_pt, 'image_original': img_np}


# # LINKNETVGG


import torch.nn as nn
import torch.nn.functional as F

class LinkNet(nn.Module):
    def __init__(self, num_classes):
        super(LinkNet, self).__init__()
        
        # Primeira camada convolucional (entrada com 3 canais, saída com 64 canais)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        
        # Segunda camada convolucional (saída com 128 canais)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Camada para ajustar de volta para 64 canais (conforme esperado nas camadas seguintes)
        self.adjust_channels = nn.Conv2d(128, 64, kernel_size=1)
        
        # Camada final de classificação
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Passa pela primeira convolução
        x = F.relu(self.conv1(x))
        
        # Passa pela segunda convolução (128 canais)
        x = F.relu(self.conv2(x))
        
        # Ajustar os canais de volta para 64
        x = F.relu(self.adjust_channels(x))
        
        # Saída final de classificação
        x = self.final_conv(x)
        return x

    # Ajustar a função de perda para ignorar o valor -1 (áreas sem rótulo)
    def eval_net_with_loss(self, image, gt, class_weights, device):
        # Forward pass: obter a saída do modelo
        output = self(image)
        
        # Redimensionar o alvo para corresponder à saída
        gt_resized = F.interpolate(gt.unsqueeze(1).float(), size=output.shape[2:], mode='nearest').squeeze(1).long()
        
        # Verificações de integridade, se as dimensões de saída e do alvo coincidem
        assert output.shape[2:] == gt_resized.shape[1:], \
            f"Dimension mismatch: Output size {output.shape[2:]} and target size {gt_resized.shape[1:]}"

        # logging.info(f"Tamanho da entrada: {image.shape}")
        # logging.info(f"Tamanho da saída: {output.shape}")
        # logging.info(f"Tamanho do alvo redimensionado: {gt_resized.shape}")

        # Ajustar para que a função de perda ignore regiões com -1
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        
        # Calcular a perda
        loss = loss_fn(output, gt_resized)
        
        return output, loss
    
    def get_params_by_kind(model):
        """
        Retorna os pesos e vieses das camadas do modelo separadamente.
        """
        base_vgg_weight = []
        base_vgg_bias = []
        core_weight = []
        core_bias = []
        
        for name, param in model.named_parameters():
            # Verifica se o parâmetro faz parte de uma camada convolucional
            if 'conv' in name:
                if 'weight' in name:
                    base_vgg_weight.append(param)
                elif 'bias' in name:
                    base_vgg_bias.append(param)
            else:
                if 'weight' in name:
                    core_weight.append(param)
                elif 'bias' in name:
                    core_bias.append(param)
        
        return base_vgg_weight, base_vgg_bias, core_weight, core_bias


# # Realiza o Treinamento da Rede


# Inicializar listas para armazenar a perda e a precisão
train_losses = []
train_accuracies = []
val_accuracies = []

# Inicia o treinamento
train_dataset = SegmentationDataset(img_folder_train, img_folder_train, True, class_to_id, resolution_input, True)
logging.info(f"Número de amostras no dataset de treinamento: {len(train_dataset)}")
logging.info(f"Arquivos no dataset de treinamento: {os.listdir(img_folder_train)}")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

val_dataset = SegmentationDataset(img_folder_val, img_folder_val, False, class_to_id, resolution_input)
logging.info(f"Número de amostras no dataset de validação: {len(val_dataset)}")
logging.info(f"Arquivos no dataset de validação: {os.listdir(img_folder_val)}")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)


if plot_train:

    for i_batch, sample_batched in enumerate(train_loader):
    
        image_np = np.squeeze(sample_batched['image_original'].cpu().numpy())
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())
            
        color_label = np.zeros((resolution_input[1], resolution_input[0], 3))
        
        for key, val in id_to_class.items():
            color_label[gt == key] = class_to_color[val]
            
        plt.figure()
        plt.imshow((image_np/255) * 0.5 + (color_label/255) * 0.5)
        plt.savefig(img_folder_train_segmentadas + "IMG_" + str(i_batch) + "_max_epochs_" + str(max_epochs) + ".png")
        plt.close('all')
        
        plt.figure()
        plt.imshow(color_label.astype(np.uint8))
        plt.savefig(img_folder_train_segmentadas + "GT_" + str(i_batch) + "_max_epochs_" + str(max_epochs) +  ".png")
        plt.close('all')            

model = LinkNet(num_classes).to(device)

# Passar a entrada pelo modelo
dummy_output = model(dummy_input)

# Verificar o tamanho da saída
logging.info(f"Input shape: {dummy_input.shape}, Output shape: {dummy_output.shape}")


# Essa função itera sobre os parâmetros do modelo e os separa em pesos e vieses (bias), distinguindo entre camadas convolucionais (base do VGG) e outras partes do modelo (núcleo/core).
# Filtragem por Nome: O nome de cada parâmetro é inspecionado para identificar se ele é um peso ou viés de uma camada convolucional (usando conv no nome).
base_vgg_weight, base_vgg_bias, core_weight, core_bias = LinkNet.get_params_by_kind(model)

# Learning rate para a parte principal do modelo
core_lr = 0.02  

# Otimizador SGD com diferentes taxas de aprendizado para vieses e pesos
optimizer = torch.optim.SGD([
    {'params': base_vgg_bias, 'lr': 0.00001}, 
    {'params': base_vgg_weight, 'lr': 0.00001},
    {'params': core_bias, 'lr': core_lr},
    {'params': core_weight, 'lr': core_lr}
])

# Scheduler para ajustar a taxa de aprendizado ao longo do tempo
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

# Treinamento e validação
best_val_acc = 0
best_epoch = 0
n_correct = 0
n_false = 0
val_accuracies = []

# Start training...
for epoch in range(max_epochs):
    
    logging.info(f'Epoch %d starting...' % (epoch+1))
    
    lr_scheduler.step()
    model.train()
    mean_loss = 0
    
    n_correct = 0
    n_false = 0    
        
    # Dentro do loop de treinamento
    for i_batch, sample_batched in enumerate(train_loader):
        image = sample_batched['image'].to(device)
        gt = sample_batched['gt'].to(device)
    
        optimizer.zero_grad()
        output, total_loss = model.eval_net_with_loss(image, gt, class_weights, device)
        total_loss.backward()
        optimizer.step()
    
        mean_loss += total_loss.cpu().detach().numpy()
    
        # Measure accuracy
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())
        
        label_out = torch.nn.functional.softmax(output, dim=1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)
        
        labels = np.argmax(label_out, axis=0)
        valid_mask = gt != -1
    
        # Redimensionar gt e valid_mask para ter as mesmas dimensões que a saída do modelo
        output_shape = labels.shape
        valid_mask_resized = cv2.resize(valid_mask.astype(np.uint8), (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        gt_resized = cv2.resize(gt, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        curr_correct = np.sum(gt_resized[valid_mask_resized] == labels[valid_mask_resized])
        curr_false = np.sum(valid_mask_resized) - curr_correct
        
        n_correct += curr_correct
        n_false += curr_false
    
    mean_loss /= len(train_loader)
    train_acc = n_correct / (n_correct + n_false)
    
    logging.info(f'Train loss: %f, train acc: %f' % (mean_loss, train_acc))
    # Armazenar a perda e a precisão de treinamento
    train_losses.append(mean_loss)
    train_accuracies.append(train_acc)    
    
    n_correct = 0
    n_false = 0

    for i_batch, sample_batched in enumerate(val_loader):
        image = sample_batched['image'].to(device)
        image_np = np.squeeze(sample_batched['image_original'].cpu().numpy())
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())

        label_out = model(image)
        label_out = torch.nn.functional.softmax(label_out, dim=1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)

        labels = np.argmax(label_out, axis=0)

        ## Se plot_val for verdadeiro e epoch for divisivel por 100, salva as imagens segmentadas
        if plot_val and epoch % 10 == 0:
            labels = np.zeros((resolution_input[1], resolution_input[0]))
            color_label = np.zeros((resolution_input[1], resolution_input[0], 3))                 

            for key, val in id_to_class.items():
                color_label[labels == key] = class_to_color[val]
                
            plt.figure()
            plt.imshow((image_np / 255) * 0.5 + (color_label / 255) * 0.5)
            plt.savefig(img_folder_val_segmentadas + "IMG_" + str(i_batch) + "_epoch_" + str(epoch) + ".png")
            logging.info(f'Imagem salva: {img_folder_val_segmentadas}IMG_' + str(i_batch) + '_epoch_' + str(epoch) + '.png')
            plt.close()
            
            plt.figure()
            plt.imshow(color_label.astype(np.uint8))
            plt.savefig(img_folder_val_segmentadas + "GT_" + str(i_batch) + "_epoch_" + str(epoch) + ".png")
            logging.info(f'Imagem salva: {img_folder_val_segmentadas}GT_' + str(i_batch) + '_epoch_' + str(epoch) + '.png')
            plt.close()

        valid_mask = gt != -1
        # Redimensionar gt e valid_mask para ter as mesmas dimensões que a saída do modelo
        output_shape = labels.shape
        valid_mask_resized = cv2.resize(valid_mask.astype(np.uint8), (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        gt_resized = cv2.resize(gt, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)

        curr_correct = np.sum(gt_resized[valid_mask_resized] == labels[valid_mask_resized])
        curr_false = np.sum(valid_mask_resized) - curr_correct

        n_correct += curr_correct
        n_false += curr_false

    total_acc = n_correct / (n_correct + n_false)
    val_accuracies.append(total_acc)

    if best_val_acc < total_acc:
        best_val_acc = total_acc
        if epoch > 7:
            torch.save(model.state_dict(), model_file_name)
            logging.info('Nova melhor conta de validação. Salvo... %f', epoch)
        best_epoch = epoch

    if (epoch - best_epoch) > patience:
        logging.info(f"Terminando o treinamento, melhor conta de validação {best_val_acc:.6f}")
        break

    logging.info(f'Validação Acc: %f -- Melhor Avaliação Acc: %f -- epoch %d.' % (total_acc, best_val_acc, best_epoch))


# # Plotar os gráficos de perda e precisão
#     Inicialização das listas: train_losses, train_accuracies e val_accuracies são listas para armazenar a perda e a precisão de treinamento e validação em cada época.
#     
#     Armazenamento dos valores: Durante o loop de treinamento, a perda e a precisão são calculadas e armazenadas nas listas correspondentes.
#     Plotagem dos gráficos: Após o loop de treinamento, os gráficos de perda e precisão são plotados usando matplotlib.
# 
# Este código deve ser adicionado ao final do seu loop de treinamento no notebook para visualizar os resultados do treinamento ao longo das épocas.


import matplotlib.pyplot as plt

# Supondo que train_losses, train_accuracies e val_accuracies já estejam definidos

plt.figure(figsize=(12, 5))

# Primeiro gráfico: Perda de Treinamento
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Perda de Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Perda de Treinamento ao longo das Épocas')
plt.legend()

# Segundo gráfico: Precisão de Treinamento e Validação
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Precisão de Treinamento')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Precisão de Validação')
plt.xlabel('Época')
plt.ylabel('Precisão')
plt.title('Precisão de Treinamento e Validação ao longo das Épocas')
plt.legend()

plt.tight_layout()
plt.savefig(save_dir + 'result_model_segmentadas_linknet_loss_accuracy.png')
logging.info(f"Gráficos salvos: {save_dir + 'result_model_segmentadas_linknet_loss_accuracy.png'}")
plt.close()


# # Inferência de dados
model = LinkNet(num_classes)
model.load_state_dict(torch.load(model_file_name))
model.eval()
model.to(device)
logging.info(f"Modelo carregado e pronto para uso.")

img_list = glob.glob(osp.join(img_folder_val, '*.JPG'))
 
for img_path in img_list:
    img_np = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
    img_np = cv2.resize(img_np, (resolution_input[0], resolution_input[1]))[..., ::-1]
    img_np = np.ascontiguousarray(img_np)
    
    img_pt = np.copy(img_np).astype(np.float32) / 255.0
    for i in range(3):
        img_pt[..., i] -= mean[i]
        img_pt[..., i] /= std[i]
        
    img_pt = img_pt.transpose(2,0,1)
        
    img_pt = torch.from_numpy(img_pt[None, ...]).to(device)
    
    label_out = model(img_pt)
    label_out = torch.nn.functional.softmax(label_out, dim = 1)
    label_out = label_out.cpu().detach().numpy()
    label_out = np.squeeze(label_out)
    
    labels = np.zeros((resolution_input[1], resolution_input[0]))
    color_label = np.zeros((resolution_input[1], resolution_input[0], 3))     

    for key, val in id_to_class.items():
        color_label[labels == key] = class_to_color[val]
        
    final_image = osp.basename(img_path)
    final_image = osp.splitext(final_image)[0]
    final_image = osp.join(save_dir, final_image)
    
    plt.figure()
    plt.imshow((img_np / 255) * 0.5 + (color_label / 255) * 0.5)
    plt.savefig(final_image + "RESULT_INFERENCIA_IMG_" + ".png")
    logging.info(f"Imagem salva: {final_image + 'RESULT_INFERENCIA_IMG_' + '.png'}")
    plt.close('all')

    plt.figure()
    plt.imshow(color_label.astype(np.uint8))
    plt.savefig(final_image + "RESULT_INFERENCIA_GT_" + ".png")
    logging.info(f"Imagem salva: {final_image + 'RESULT_INFERENCIA_GT_' + '.png'}")
    plt.close('all')    
