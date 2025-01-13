# Validacao Facial com OpenCV e DNN

Este projeto implementa a detecção de rostos, além da análise de possiçoes utilizando redes neurais profundas (DNN) e a biblioteca OpenCV.
A aplicação usa um modelo pré-treinado para detecção facial e outro para detectar pontos de referência faciais.
O código também calcula a relação da abertura dos olhos (EAR) e da boca (MAR), proporcionando informações sobre o estado do rosto para validacoes como as ultilizadas para reconhecimento facial".

## Funcionalidades

- **Detecção de Rostos**: Utiliza um modelo DNN pré-treinado para detectar rostos em imagens ou vídeos.
- **Pontos de Referência Faciais**: Detecta e mapeia os pontos de referência faciais para detectar espresoes faciais.
- **Cálculo de EAR (Eye Aspect Ratio)**: Avalia a abertura dos olhos para determinar se a pessoa está com os olhos fechados ou abertos.
- **Cálculo de MAR (Mouth Aspect Ratio)**: Avalia a abertura da boca para determinar se está aberta ou fechada.
- **Exibição de Resultados**: Exibe em tempo real no vídeo a detecção do rosto e indicadores de status para a validacoes das possicoes dos olhos e da boca.

## Requisitos

- OpenCV 4.x
- Modelo Caffe para detecção facial
- Modelo LBF para pontos de referência faciais
- Vídeo ou imagem de entrada

## Estrutura do Código

### 1. **DNNNetSingleton**
   - A classe `DNNNetSingleton` é um singleton para garantir que a rede neural (Net) seja carregada uma única vez. Ela utiliza os arquivos de configuração e pesos para inicializar a rede.

### 2. **Funções de Cálculo**
   - **`calculateAspectRatio`**: Funçao generica ultilizada para a implementacao dos calculos de *Aspect Ratio* que podem ser implementadas posteriormente.
   - **`eyes`**: Calcula o **Eye Aspect Ratio (EAR)** para medir a abertura dos olhos com base nos pontos de referência faciais.
   - **`mouth`**: Calcula o **Mouth Aspect Ratio (MAR)** para medir a abertura da boca com base nos pontos de referência faciais.

### 3. **Detecção de Rostos com DNN**
   - **`detectFace`**: Utiliza o modelo DNN carregado para detectar rostos na imagem ou vídeo. A detecção é feita em duas etapas: pré-processamento da imagem e execução da rede neural para identificar as regiões que correspondem a rostos.

### 4. **Processamento de Frames**
   - **`processFrame`**: A função principal para processar cada frame de vídeo. Realiza:
     - Redimensionamento do frame para diminuir a resolução.
     - Conversão para escala de cinza e equalização do histograma.
     - Detecção de rostos.
     - Detecção e exibição de pontos de referência faciais.
     - Cálculo e exibição do EAR e MAR.
     - Exibição de mensagens indicativas do estado dos olhos e boca (abertos ou fechados).

### 5. **Função Principal (main)**
   - A função principal (`main`) abre o vídeo ou a imagem fornecida, inicializa a rede DNN e o modelo de pontos de referência faciais, e processa os frames em tempo real. Ela também exibe a imagem com os resultados da análise.

## Como Usar

1. **Instalar as dependências**: Certifique-se de que você tenha o OpenCV instalado no seu ambiente. Você pode instalar usando o comando:

   ```bash
   pip install opencv-python opencv-contrib-python
   ```

2. **Modelos Necessários**:
   - Baixe os arquivos de configuração e pesos para a detecção facial:
     - `deploy.prototxt`: Arquivo de configuração da rede.
     - `res10_300x300_ssd_iter_140000_fp16.caffemodel`: Arquivo de pesos do modelo.
   - Baixe o modelo LBF para a detecção de pontos de referência faciais:
     - `lbfmodel.yaml`

3. **Configuração de Arquivos**:
   - Defina os caminhos corretos para os arquivos de configuração e modelos no código, caso os arquivos estejam em diretórios diferentes.

4. **Executar o Programa**:
   - Você pode usar um arquivo de vídeo ou uma imagem como entrada. O código é configurado para aceitar um vídeo pelo caminho `./New folder/video_calibracao.mp4`, mas você pode alterar para outro arquivo de sua escolha.
   - Para rodar o programa, execute o código:

   ```bash
   g++ -o face_analysis face_analysis.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_objdetect -lopencv_face
   ./face_analysis
   ```

## Detalhamento das Funções e Métodos

### `DNNNetSingleton::getInstance`
Retorna uma referência para a rede DNN. Se a rede ainda não estiver carregada, ela será carregada da primeira vez que o método for chamado.

### `calculateEAR`
Calcula a **Eye Aspect Ratio** (EAR), que é usada para determinar se os olhos estão abertos ou fechados. A fórmula baseia-se na distância vertical entre os pontos do olho e na distância horizontal.

### `calculateMAR`
Calcula a **Mouth Aspect Ratio** (MAR), que é usada para determinar se a boca está aberta ou fechada. A fórmula baseia-se nas distâncias verticais e horizontais entre os pontos da boca.

### `detectFaceOpenCVDNN`
Utiliza o modelo DNN para detectar rostos na imagem ou vídeo. A função retorna um vetor de retângulos delimitadores para os rostos encontrados.

### `processFrame`
Processa cada frame de vídeo:
- Detecta rostos.
- Detecta pontos de referência faciais.
- Calcula o EAR e MAR.
- Exibe informações sobre o estado dos olhos e boca.

### `main`
A função principal que executa o código, processando vídeo ou imagem e exibindo os resultados em tempo real.

## Conclusão

Este projeto oferece uma solução simples para análise facial em tempo real usando OpenCV e DNN, com o cálculo da abertura dos olhos e da boca. Ele pode ser facilmente modificado para incluir outras análises faciais ou aprimorar o desempenho.
