#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);  

const std::string caffeConfigFile = "./DNN_Detection/deploy.prototxt";
const std::string caffeWeightFile = "./DNN_Detection/res10_300x300_ssd_iter_140000_fp16.caffemodel";

std::vector<cv::Rect> detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN)
{
    std::vector<cv::Rect> faceRects;
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
    net.setInput(inputBlob, "data");
    Mat detection = net.forward("detection_out");

    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            faceRects.push_back(Rect(Point(x1, y1), Point(x2, y2)));
        }
    }

    return faceRects;
}

void processFrame(Mat &frame, Net &net, CascadeClassifier &eyeCascade, CascadeClassifier &mouthCascade)
{
    cv::resize(frame, frame, cv::Size(), 0.5, 0.5); // Reduz a resolução para 50%

    // Converter o quadro para escala de cinza
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    // Detectar rostos com DNN
    std::vector<cv::Rect> faces = detectFaceOpenCVDNN(net, frame);

    bool validDetection = false;

    for (const auto &face : faces)
    {
        cv::Mat roiGray = gray(face);
        cv::Mat roiSrc = frame(face);

        cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "Rosto", cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);

        // Detectar olhos
        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(roiGray, eyes, 1.1, 5, 0, cv::Size(30, 30));
        for (const auto &eye : eyes)
        {
            cv::Rect eyeRect = eye;
            eyeRect.x += face.x;
            eyeRect.y += face.y;
            cv::rectangle(frame, eyeRect, cv::Scalar(0, 255, 0), 2);
        }

        // Detectar boca
        std::vector<cv::Rect> mouths;
        mouthCascade.detectMultiScale(roiGray, mouths, 1.1, 5, 0, cv::Size(40, 40));
        if (!mouths.empty())
        {
            const auto &mouth = mouths[0];
            cv::Rect mouthRect = mouth;
            mouthRect.x += face.x;
            mouthRect.y += face.y;
            cv::rectangle(frame, mouthRect, cv::Scalar(0, 0, 255), 2);
        }

        validDetection = !eyes.empty() && !mouths.empty();
    }

    // Exibir mensagem "Inválido" caso a validação falhe
    if (!validDetection)
    {
        cv::putText(frame, "Invalido", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
    else
    {
        cv::putText(frame, "Valido", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }

    // Exibir o quadro processado
    cv::imshow("Resultado", frame);
}

int main()
{
    std::string inputPath = "./New folder/imagem_teste.jpg";

    // Carregar o modelo DNN
    Net net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);

    // Carregar classificadores para olhos e boca
    cv::CascadeClassifier eyeCascade, mouthCascade;
    if (!eyeCascade.load("./New folder/haarcascade_eye_tree_eyeglasses (1).xml"))
    {
        std::cerr << "Erro ao carregar o classificador de olhos!" << std::endl;
        return -1;
    }
    if (!mouthCascade.load("./haarcascade_smile (1).xml"))
    {
        std::cerr << "Erro ao carregar o classificador de boca!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::VideoCapture cap(inputPath);

    if (!cap.isOpened())
    {
        // Tenta carregar como imagem se o vídeo falhar
        frame = cv::imread(inputPath);
        if (frame.empty())
        {
            std::cerr << "Erro ao abrir o vídeo ou imagem!" << std::endl;
            return -1;
        }
        processFrame(frame, net, eyeCascade, mouthCascade);
        cv::waitKey(0); // Espera indefinidamente para imagens
    }
    else
    {
        while (true)
        {
            cap >> frame; // Ler um quadro do vídeo
            if (frame.empty())
                break; // Fim do vídeo

            processFrame(frame, net, eyeCascade, mouthCascade);

            // Sair ao pressionar a tecla 'q'
            if (cv::waitKey(1) == 'q')
                break;
        }
    }

    // Liberar recursos
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
