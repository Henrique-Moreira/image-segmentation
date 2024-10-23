# Documentação - Segmentação de Imagens com PSPNet

## Introdução
Este projeto implementa a rede **PSPNet (Pyramid Scene Parsing Network)**, uma arquitetura de redes neurais profundas para segmentação de imagens. O objetivo é realizar a segmentação semântica precisa de cada pixel, capturando informações contextuais em múltiplas escalas. A PSPNet é particularmente eficiente em imagens complexas onde objetos de diferentes tamanhos estão presentes.

## Estrutura do Código
1. **Importação de Bibliotecas**:
   - Bibliotecas como `tensorflow`, `keras`, `cv2`, `numpy` e `matplotlib` são importadas para manipulação de imagens, construção de redes neurais e visualização dos resultados.

2. **Carregamento do Dataset**:
   - As imagens de entrada são carregadas do dataset. Neste exemplo, utilizamos o dataset de segmentação semântica pré-processado.
   - Cada imagem de entrada tem um correspondente "ground truth" (verdade de solo) que contém a segmentação real para a imagem.

3. **Arquitetura PSPNet**:
   - O modelo PSPNet é carregado com pesos pré-treinados. A rede utiliza o **pooling piramidal** para capturar informações contextuais de diferentes escalas, o que melhora a precisão da segmentação.
   - Após o pooling, as diferentes características são combinadas para produzir uma máscara de segmentação precisa.

4. **Treinamento e Inferência**:
   - O modelo pode ser treinado em novos datasets ou usado diretamente para inferência em imagens de teste.
   - Após a inferência, as imagens segmentadas são comparadas com o ground truth para avaliar a precisão.

5. **Visualização dos Resultados**:
   - Para facilitar a análise, as previsões do modelo são visualizadas graficamente ao lado das imagens originais e das máscaras de segmentação reais.

## Pré-requisitos
- Python 3.x
- TensorFlow/Keras
- OpenCV
- Matplotlib

## Execução do Código
1. Clone o repositório com o código.
2. Instale as dependências listadas no arquivo `requirements.txt`.
3. Execute o notebook Jupyter fornecido para treinar o modelo ou fazer inferências em novas imagens.

## Resultados
As imagens segmentadas serão salvas e exibidas com sobreposição de cores para destacar as áreas corretamente identificadas pelo modelo.

