# Detecção e Análise Facial com OpenCV

## Visão Geral
Este projeto utiliza a biblioteca OpenCV para realizar detecção e análise facial em vídeos ou imagens estáticas. Ele utiliza uma rede neural convolucional para detectar rostos e landmarks faciais e realiza cálculos de Aspect Ratio (AR) para identificar estados faciais como "Olhos Fechados" ou "Boca Aberta".

## Funcionalidades Principais

1. **Detecção de Faces:**
   - Utiliza um modelo de rede neural (DNN) treinado para identificar rostos em um frame.
   - A detecção é baseada em um modelo Caffe com configurações e pesos pré-treinados.

2. **Identificação de Landmarks Faciais:**
   - Após a detecção do rosto, landmarks faciais são identificados usando o modelo LBF (Local Binary Features).

3. **Cálculo de Aspect Ratio (AR):**
   - **EAR (Eye Aspect Ratio):** Mede a abertura dos olhos para determinar se estão abertos ou fechados.
   - **MAR (Mouth Aspect Ratio):** Mede a abertura da boca para determinar se está aberta ou fechada.

4. **Visualização:**
   - Desenha retângulos ao redor dos rostos detectados.
   - Marcações dos landmarks faciais.
   - Exibe textos indicando o estado dos olhos e da boca.

## Estrutura do Código

### 1. **Classe `FrameProcesser`**
Responsável por processar cada frame da imagem ou vídeo.

- **Função `processFrame`:**
  - Reduz a resolução do frame para otimizar o processamento.
  - Converte o frame para escala de cinza e equaliza o histograma para melhorar a detecção.
  - Detecta rostos usando o modelo DNN.
  - Ajusta os landmarks faciais e calcula EAR e MAR.
  - Exibe o frame com as anotações visuais.

### 2. **Classe `CalculateDNN`**
Gerencia o modelo de rede neural para detecção facial.

- **Funções Principais:**
  - `getInstance`: Retorna a instância Singleton da rede neural.
  - `detectFace`: Detecta rostos no frame usando o modelo DNN.

### 3. **Classe `CalculateAR`**
Realiza os cálculos de Aspect Ratio para olhos e boca.

- **Funções Principais:**
  - `calculateAspectRatio`: Cálculo genérico do Aspect Ratio.
  - `eyes`: Cálculo do EAR.
  - `mouth`: Cálculo do MAR.

### 4. **Funções Auxiliares**
- **`drawPolyline` e `drawLandmarks`:** Desenham os landmarks faciais no frame.

### 5. **Função `main`**
Carrega o vídeo ou imagem, inicializa os componentes necessários e processa cada frame.

- **Fluxo Principal:**
  - Carrega o arquivo de entrada (vídeo ou imagem).
  - Inicializa o modelo DNN e o modelo de facemark.
  - Processa cada frame, chamando `processFrame`.
  - Exibe os resultados e aguarda uma tecla para sair.

## Como Executar

### Pré-requisitos
- OpenCV instalado com suporte para DNN e facemark.
- Arquivos de modelo para detecção facial (`deploy.prototxt` e `res10_300x300_ssd_iter_140000_fp16.caffemodel`).
- Modelo LBF para landmarks (`lbfmodel.yaml`).

### Comandos
1. Compile o código:
   ```bash
   g++ -o facial_analysis main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_face
   ```

2. Execute o programa:
   ```bash
   ./facial_analysis
   ```

## Personalização

- **Parâmetros Threshold:** Os valores de EAR e MAR podem ser ajustados na classe `FrameProcesser` para alterar os critérios de detecção de olhos fechados e boca aberta.
- **Modelos Customizados:** Pode-se substituir os modelos de DNN e LBF por outros, conforme necessário.

## Possíveis Melhorias

1. **Melhor Gerenciamento de Erros:** Adicionar tratamento de exceções para falhas no carregamento de arquivos ou na execução de funções.
2. **Interface Gráfica Mais Rica:** Incluir gráficos ou outros elementos visuais para uma melhor representação dos resultados.
3. **Otimização de Performance:** Implementar técnicas de paralelização ou processamento em lote para melhorar o desempenho.
4. **Configuração Externa:** Permitir configuração dos parâmetros através de arquivos ou argumentos de linha de comando.

## Conclusão
Este projeto demonstra o uso de OpenCV para detecção e análise facial em tempo real. Com algumas melhorias e personalizações, ele pode ser expandido para aplicações em monitoramento de fadiga, segurança, ou interações baseadas em expressões faciais.

