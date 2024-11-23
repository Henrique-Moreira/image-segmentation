import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import torch
import os.path as osp
import glob
import torchvision
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configuração do logger
log_dir = r'C:\git\image-segmentation\results\psp-dataset-base'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
filenamelog = 'psp-dataset-base-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.log'
logging.basicConfig(filename=osp.join(log_dir, filenamelog), level=logging.INFO, format='%(asctime)s - %(message)s')

class PSPDec(torch.nn.Module):
	def __init__(self, in_dim, reduction_dim, setting):
		super(PSPDec, self).__init__()
		self.features = []
		for s in setting:
			self.features.append(torch.nn.Sequential(
				torch.nn.AdaptiveAvgPool2d(s),
				torch.nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
				#torch.nn.BatchNorm2d(reduction_dim, momentum=.95),
				#torch.nn.InstanceNorm2d(reduction_dim, momentum=.95),
				torch.nn.ReLU(inplace=True)
			))
		self.features = torch.nn.ModuleList(self.features)


	def forward(self, x):
		x_size = x.size()
		out = [x]
		for f in self.features:
			out.append(torch.nn.functional.upsample(f(x), x_size[2:], mode='bilinear'))
		out = torch.cat(out, 1)
		return out

class PSPNet(torch.nn.Module):

	def __init__(self, num_classes):
		super(PSPNet, self).__init__()

		resnet = torchvision.models.resnet101(pretrained=True)
		#print('resnet', resnet)

		self.layer0 = torch.nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
			resnet.maxpool
		)
		
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4
		

		for n, m in self.layer3.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)
		for n, m in self.layer4.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)


		self.ppm = PSPDec(2048, 512, (1, 2, 3, 6))

		self.final = torch.nn.Sequential(
			torch.nn.Conv2d(4096, 512, 3, padding=1, bias=False),
			torch.nn.BatchNorm2d(512, momentum=.95),
			torch.nn.ReLU(inplace=True),
			torch.nn.Dropout(.1),
			torch.nn.Conv2d(512, num_classes, 1),
		)


	def forward(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		
		x = self.ppm(x)
		x = self.final(x)
		return torch.nn.functional.upsample(x, (480, 640), mode='bilinear')


	@staticmethod
	def eval_net_with_loss(model, inp, gt, class_weights, device):

		weights = torch.from_numpy(np.array(class_weights, dtype=np.float32)).to(device)
		out = model(inp)

		softmax = torch.nn.functional.log_softmax(out, dim = 1)
		loss = torch.nn.functional.nll_loss(softmax, gt, ignore_index=-1, weight=weights)

		return (out, loss)

plot_val = False
plot_train = True

# Configuração do dispositivo CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda_available = torch.cuda.is_available()
logging.info('CUDA disponível: %s', cuda_available)

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convertendo para GB
    vram_available = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convertendo para GB
    logging.info('Nome da GPU: %s', gpu_name)
    logging.info('VRAM Total: %.2f GB', vram_total)

# Caminho do diretório Dataset
directory = r'C:\git\image-segmentation\dataset'
logging.info('Diretório do Projeto %s', directory)
if not os.path.exists(directory):
    os.makedirs(directory)
img_folder_val = directory + r'\\base\\Val'
img_folder_train = directory + r'\\base\\Train'
img_folder_test = directory + r'\\base\\Test'
save_dir = directory + r'\\result_PSPNET_base\\'
if not os.path.exists(img_folder_val):
    os.makedirs(img_folder_val)
if not os.path.exists(img_folder_train):
    os.makedirs(img_folder_train)
if not os.path.exists(img_folder_test):
    os.makedirs(img_folder_test)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
## Imagens Segmentadas
img_folder_train_segmentadas = directory + r'\\segmentadas_pspnet\\train\\'
img_folder_val_segmentadas = directory + r'\\segmentadas_pspnet\\val\\'
img_folder_test_segmentadas = directory + r'\\segmentadas_pspnet\\test\\'
if not os.path.exists(img_folder_train_segmentadas):
    os.makedirs(img_folder_train_segmentadas)
if not os.path.exists(img_folder_val_segmentadas):
    os.makedirs(img_folder_val_segmentadas)
if not os.path.exists(img_folder_test_segmentadas):
    os.makedirs(img_folder_test_segmentadas)
    
# Local onde o Modelo será salvo
model_file_name = save_dir + 'model_PSPNET_segmentadas.pth'
logging.info('Modelo será salvo em: %s', model_file_name)

# Configurações do treinamento
resolution_input = (640, 480)  # Tamanho de entrada
logging.info('Resolução de entrada: %s', resolution_input)
patience = 30
logging.info('Patience: %s', patience)
max_epochs = 300
logging.info('Número máximo de épocas: %s', max_epochs)

# Mapeamento de classes e cores
class_to_color = {'Doenca': (255, 0, 0), 'Solo': (0, 0, 255), 'Saudavel': (0, 255, 255), 'Folhas': (0, 255, 0)}
logging.info('Mapeamento de classes para cores: %s', class_to_color)
class_to_id = {'Doenca': 0, 'Solo': 1, 'Saudavel': 2, 'Folhas': 3}
logging.info('Mapeamento de classes para IDs: %s', class_to_id)
num_classes = len(class_to_id)
class_weights = [1, 2, 3, 4]
logging.info('Pesos de classes: %s', class_weights)
id_to_class = {v: k for k, v in class_to_id.items()}


class SegmentationDataset(Dataset):
    """Segmentation dataset loader."""

    def __init__(self, json_folder, img_folder, is_train, class_to_id, resolution_input=(640, 480), augmentation=False, transform=None):
        self.gt_file_list = glob.glob(osp.join(json_folder, '*.json'))
        self.total_samples = len(self.gt_file_list)
        self.img_folder = img_folder
        self.is_train = is_train
        self.transform = transform
        self.augmentation = augmentation
        self.resolution = resolution_input
        self.class_to_id = class_to_id
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        gt_file = self.gt_file_list[idx]
        img_number_str = osp.splitext(osp.basename(gt_file))[0]
        
        # Verificação de existência de arquivos
        if not osp.exists(gt_file):
            raise FileNotFoundError(f"Arquivo JSON não encontrado: {gt_file}")
        
        # Extrair o nome da imagem considerando que o nome da classe está no meio do nome do arquivo
        img_path = osp.join(self.img_folder, img_number_str + '.JPG')
        
        if not osp.exists(img_path):
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        
        gt_json = json.load(open(gt_file, 'r'))
        img_np = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
        
        if img_np is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        
        original_shape = img_np.shape
        img_np = cv2.resize(img_np, (self.resolution[0], self.resolution[1]))[..., ::-1]
        img_np = np.ascontiguousarray(img_np)
        label_np = np.zeros((img_np.shape[0], img_np.shape[1]))
        label_np[...] = -1

        for shape in gt_json['shapes']:
            points_np = np.array(shape['points'], dtype=np.float64)
            points_np[:, 0] *= self.resolution[0] / original_shape[1]
            points_np[:, 1] *= self.resolution[1] / original_shape[0]
            points_np = np.round(points_np).astype(np.int64)
            points_np = points_np.reshape((-1, 1, 2))
            label = shape['label']
            if label not in self.class_to_id:
                raise KeyError(f"Label '{label}' não encontrado em class_to_id")
            label_np = cv2.fillPoly(label_np, [points_np], self.class_to_id[label])

        label_np = label_np.astype(np.int32)

        if self.is_train and self.augmentation:
            if np.random.rand() > 0.5:
                img_np = np.fliplr(img_np)
                label_np = np.fliplr(label_np)
                img_np = np.ascontiguousarray(img_np)
                label_np = np.ascontiguousarray(label_np)

        img_pt = img_np.astype(np.float32) / 255.0
        for i in range(3):
            img_pt[..., i] -= self.mean[i]
            img_pt[..., i] /= self.std[i]

        img_pt = img_pt.transpose(2, 0, 1)
        img_pt = torch.from_numpy(img_pt)
        label_pt = torch.from_numpy(label_np).long()

        sample = {'image': img_pt, 'gt': label_pt, 'image_original': img_np}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Inicializar listas para armazenar a perda e a precisão
train_losses = []
train_accuracies = []
val_accuracies = []

# Inicia o treinamento
train_dataset = SegmentationDataset(img_folder_train, img_folder_train, True, class_to_id, resolution_input, True, None)
logging.info('Número de amostras no dataset de treinamento: %s', len(train_dataset))
logging.info('Arquivos no dataset de treinamento: %s', os.listdir(img_folder_train))

val_dataset = SegmentationDataset(img_folder_val, img_folder_val, False, class_to_id, resolution_input, False, None)
logging.info('Número de amostras no dataset de validação: %s', len(val_dataset))
logging.info('Arquivos no dataset de validação: %s', os.listdir(img_folder_val))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

## Se plot_val for verdadeiro e epoch for divisivel por 100, salva as imagens segmentadas
if plot_val:
    for i_batch, sample_batched in enumerate(train_loader):
        image_np = np.squeeze(sample_batched['image_original'].cpu().numpy())
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())
        
        color_label = np.zeros((resolution_input[1], resolution_input[0], 3))
        
        for key, val in id_to_class.items():
            color_label[gt == key] = class_to_color.get(val, [0, 0, 0])  # Provide a default color if key is missing
        
        plt.figure()
        plt.imshow((image_np / 255) * 0.5 + (color_label / 255) * 0.5)
        plt.savefig(img_folder_train_segmentadas + "IMG_" + str(i_batch) + "_max_epochs_" + str(max_epochs) + ".png")
        logging.info('Imagem salva em: %s', img_folder_train_segmentadas + "IMG_" + str(i_batch) + "_max_epochs_" + str(max_epochs) + ".png")
        plt.close('all')
        
        plt.figure()
        plt.imshow(color_label.astype(np.uint8))
        plt.savefig(img_folder_train_segmentadas + "GT_" + str(i_batch) + "_max_epochs_" + str(max_epochs) +  ".png")
        logging.info('Imagem salva em: %s', img_folder_train_segmentadas + "GT_" + str(i_batch) + "_max_epochs_" + str(max_epochs) +  ".png")
        plt.close('all')   

model = PSPNet(num_classes).to(device)

optimizer = torch.optim.SGD(model.parameters(), 1e-2, .9, 1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.2)

best_val_acc = -1
best_epoch = 0

# Start training...
for epoch in range(max_epochs):
    
    print('Epoch %d starting...' % (epoch+1))
    logging.info('Epoch %d starting...', epoch+1)
    
    lr_scheduler.step()
    
    model.train()
    
    mean_loss = 0.0
    
    n_correct = 0
    n_false = 0
    
    for i_batch, sample_batched in enumerate(train_loader):
    
    
        image = sample_batched['image'].to(device)
        gt = sample_batched['gt'].to(device)
    
        optimizer.zero_grad()
        output, total_loss = model.eval_net_with_loss(model, image, gt, class_weights, device)
        total_loss.backward()
        optimizer.step()
        
        mean_loss += total_loss.cpu().detach().numpy()
        
        # Measure accuracy
        
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())
        
        label_out = torch.nn.functional.softmax(output, dim = 1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)
        
        labels = np.argmax(label_out, axis=0)
        valid_mask = gt != -1
        curr_correct = np.sum(gt[valid_mask] == labels[valid_mask])
        curr_false = np.sum(valid_mask) - curr_correct
        n_correct += curr_correct
        n_false += curr_false
        
    mean_loss /= len(train_loader)
    train_acc = n_correct / (n_correct + n_false)
        
    print('Train loss: %f, train acc: %f' % (mean_loss, train_acc))
    logging.info('Train loss: %f, train acc: %f', mean_loss, train_acc)
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
        label_out = torch.nn.functional.softmax(label_out, dim = 1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)
        
        labels = np.argmax(label_out, axis=0)
        
        # Salvar imagens segmentadas
        if plot_val and epoch % 100 == 0:
            
            color_label = np.zeros((resolution_input[1], resolution_input[0], 3))
            
            for key, val in id_to_class.items():
                color_label[labels == key] = class_to_color[val]
                
            plt.figure()
            plt.imshow((image_np/255) * 0.5 + (color_label/255) * 0.5)
            plt.savefig(img_folder_val_segmentadas + "IMG_" + str(i_batch) + "_epoch_" + str(epoch) + ".png")
            logging.info('Imagem salva em: %s', img_folder_val_segmentadas + "IMG_" + str(i_batch) + "_epoch_" + str(epoch) + ".png")
            plt.close('all')
            
            plt.figure()
            plt.imshow(color_label.astype(np.uint8))
            plt.savefig(img_folder_val_segmentadas + "GT_" + str(i_batch) + "_epoch_" + str(epoch) +  ".png")
            logging.info('Imagem salva em: %s', img_folder_val_segmentadas + "GT_" + str(i_batch) + "_epoch_" + str(epoch) +  ".png")
            plt.close('all')
        
        valid_mask = gt != -1
        curr_correct = np.sum(gt[valid_mask] == labels[valid_mask])
        curr_false = np.sum(valid_mask) - curr_correct
        n_correct += curr_correct
        n_false += curr_false    
        
    total_acc = n_correct / (n_correct + n_false)
    val_accuracies.append(total_acc)
    
    if best_val_acc < total_acc:
        best_val_acc = total_acc
        if epoch > 7:
            torch.save(model.state_dict(), model_file_name)
            print('Nova melhor conta de validação. Salvo... %f', epoch)
            logging.info('Nova melhor conta de validação. Salvo... %f', epoch)
        best_epoch = epoch

    if (epoch - best_epoch) > patience:
        print(f"Terminando o treinamento, melhor conta de validação {best_val_acc:.6f}")
        logging.info("Terminando o treinamento, melhor conta de validação %f", best_val_acc)
        break
    
    print('Validação Acc: %f -- Melhor Avaliação Acc: %f -- epoch %d.' % (total_acc, best_val_acc, best_epoch))
    logging.info('Validação Acc: %f -- Melhor Avaliação Acc: %f -- epoch %d.', total_acc, best_val_acc, best_epoch)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Perda de Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Perda de Treinamento ao longo das Épocas')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Precisão de Treinamento')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Precisão de Validação')
plt.xlabel('Época')
plt.ylabel('Precisão')
plt.title('Precisão de Treinamento e Validação ao longo das Épocas')
plt.legend()

plt.tight_layout()
plt.savefig(save_dir + 'result_model_segmentadas_pspnet_loss_accuracy.png')
logging.info('Imagem salva em: %s', save_dir + 'result_model_segmentadas_unet_loss_accuracy.png')
plt.close('all')

## Configurações do treinamento
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

model = PSPNet(num_classes)
model.load_state_dict(torch.load(model_file_name, weights_only=True))
model.eval()
print("Modelo carregado e pronto para uso.")
logging.info('Modelo carregado e pronto para uso.')
model.to(device)

img_list = glob.glob(osp.join(img_folder_val, '*.JPG'))
logging.info('Imagens de teste: %s', len(img_list))

for img_path in img_list:
    img_np = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
    img_np = cv2.resize(img_np, (resolution_input[0], resolution_input[1]))[..., ::-1]
    img_np = np.ascontiguousarray(img_np)
    
    img_pt = np.copy(img_np).astype(np.float32) / 255.0
    for i in range(3):
        img_pt[..., i] -= mean[i]
        img_pt[..., i] /= std[i]

    img_pt = img_pt.transpose(2, 0, 1)
    img_pt = torch.from_numpy(img_pt[None, ...]).to(device)

    label_out = model(img_pt)
    label_out = torch.nn.functional.softmax(label_out, dim=1)
    label_out = label_out.cpu().detach().numpy()
    label_out = np.squeeze(label_out)

    labels = np.argmax(label_out, axis=0)

    color_label = np.zeros((resolution_input[1], resolution_input[0], 3))

    for key, val in id_to_class.items():
        color_label[labels == key] = class_to_color[val]
        
    final_image = osp.basename(img_path)
    final_image = osp.splitext(final_image)[0]
    final_image = osp.join(img_folder_test_segmentadas, final_image)
    
    plt.figure()
    plt.imshow((img_np / 255) * 0.5 + (color_label / 255) * 0.5)
    plt.savefig(img_folder_test_segmentadas + "RESULT_INFERENCIA_IMG_" + ".png")
    logging.info('Imagem salva em: %s', img_folder_test_segmentadas + "RESULT_INFERENCIA_IMG_" + ".png")
    plt.close('all')

    plt.figure()
    plt.imshow(color_label.astype(np.uint8))
    plt.savefig(img_folder_test_segmentadas + "RESULT_INFERENCIA_GT_" + ".png")
    logging.info('Imagem salva em: %s', img_folder_test_segmentadas + "RESULT_INFERENCIA_GT_" + ".png")
    plt.close('all')