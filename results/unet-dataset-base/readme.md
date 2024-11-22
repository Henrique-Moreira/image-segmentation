# U-Net com VGG16

Este projeto implementa uma rede neural U-Net utilizando a arquitetura VGG16 para segmentação de imagens. O código é escrito em Python e utiliza a biblioteca PyTorch para a construção e treinamento do modelo.

## Descrição do Arquivo `u-net-VGG16.py`

O arquivo `u-net-VGG16.py` contém o código para configurar, treinar e avaliar o modelo U-Net com VGG16. Abaixo estão os principais componentes do arquivo:

1. **Configuração do Dispositivo CUDA**: Verifica se uma GPU está disponível e configura o dispositivo para usar CUDA se disponível.
2. **Configuração dos Diretórios**: Define os caminhos para os diretórios de treinamento, validação, teste e onde os resultados e modelos serão salvos.
3. **Configurações do Treinamento**: Define parâmetros como resolução de entrada, paciência, número máximo de épocas, pesos das classes e número de classes.
4. **Treinamento do Modelo**: Inclui o loop de treinamento que calcula a perda e a acurácia para cada época, salva o modelo se a acurácia de validação melhorar e interrompe o treinamento se a acurácia não melhorar após um número definido de épocas (paciência).

## Resultados Obtidos

Durante o treinamento, o modelo imprime a perda e a acurácia de treinamento e validação para cada época. Abaixo estão alguns dos resultados obtidos durante a execução do treinamento:

## Resulttados após execução.
Epoch 266 starting...
Train loss: 0.313497, train acc: 0.845227
Validação Acc: 0.826479 -- Melhor Avaliação Acc: 0.826481 -- epoch 236.
Epoch 267 starting...
Train loss: 0.313483, train acc: 0.844028
Validação Acc: 0.826481 -- Melhor Avaliação Acc: 0.826481 -- epoch 236.
Epoch 268 starting...
Train loss: 0.315446, train acc: 0.844245
Terminando o treinamento, melhor conta de validação 0.826481