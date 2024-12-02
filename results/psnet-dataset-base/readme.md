# Rede PSPNET

Este projeto implementa uma rede neural PSPNET, para segmentação de imagens. O código é escrito em Python e utiliza a biblioteca PyTorch para a construção e treinamento do modelo.

## Descrição do Arquivo `psp-net.py`

O arquivo `psp-net.py` contém o código para configurar, treinar e avaliar o modelo U-Net com VGG16. Abaixo estão os principais componentes do arquivo:

1. **Configuração do Dispositivo CUDA**: Verifica se uma GPU está disponível e configura o dispositivo para usar CUDA se disponível.
2. **Configuração dos Diretórios**: Define os caminhos para os diretórios de treinamento, validação, teste e onde os resultados e modelos serão salvos.
3. **Configurações do Treinamento**: Define parâmetros como resolução de entrada, paciência, número máximo de épocas, pesos das classes e número de classes.
4. **Treinamento do Modelo**: Inclui o loop de treinamento que calcula a perda e a acurácia para cada época, salva o modelo se a acurácia de validação melhorar e interrompe o treinamento se a acurácia não melhorar após um número definido de épocas (paciência).

## Resultados Obtidos

Durante o treinamento, o modelo imprime a perda e a acurácia de treinamento e validação para cada época. Abaixo estão alguns dos resultados obtidos durante a execução do treinamento:

## Resulttados após execução.
Epoch 39 starting...
Train loss: 0.003896, train acc: 0.999004
Validação Acc: 0.917224 -- Melhor Avaliação Acc: 0.923547 -- epoch 11.
Epoch 40 starting...
Train loss: 0.004067, train acc: 0.998913
Validação Acc: 0.917240 -- Melhor Avaliação Acc: 0.923547 -- epoch 11.
Epoch 41 starting...
Train loss: 0.004041, train acc: 0.998918
Validação Acc: 0.917312 -- Melhor Avaliação Acc: 0.923547 -- epoch 11.
Epoch 42 starting...
Train loss: 0.003925, train acc: 0.998983
Validação Acc: 0.917139 -- Melhor Avaliação Acc: 0.923547 -- epoch 11.
Epoch 43 starting...
Train loss: 0.003882, train acc: 0.999005
Terminando o treinamento, melhor conta de validação 0.923547
Modelo carregado e pronto 