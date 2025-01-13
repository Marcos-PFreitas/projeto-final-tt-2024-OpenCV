#ifndef CALCULATEAR_HPP
#define CALCULATEAR_HPP

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

// Aqui não precisa ser classe
class CalculateAR
{
public:
    // Função genérica para calcular Aspect Ratio
    double calculateAspectRatio(const std::vector<Point2f> &landmarks,
                                const std::vector<int> &indices, double normalizer) const
    {
        double vertical1 = norm(landmarks[indices[1]] - landmarks[indices[5]]);
        double vertical2 = norm(landmarks[indices[2]] - landmarks[indices[4]]);
        double horizontal = norm(landmarks[indices[0]] - landmarks[indices[3]]);

        return (vertical1 + vertical2) / (normalizer * horizontal);
    }
    // Função para cálculo de abertura do olho (EAR)
    double eyes(const std::vector<Point2f> &landmarks, int startIndex) const
    {
        std::vector<int> eyeIndices = {startIndex, startIndex + 1, startIndex + 2,
                                       startIndex + 3, startIndex + 4, startIndex + 5};
        return calculateAspectRatio(landmarks, eyeIndices, 1.5);
    }

    // Função para cálculo de abertura da boca (MAR)
    double mouth(const std::vector<Point2f> &landmarks) const
    {
        std::vector<int> mouthIndices = {48, 50, 52, 54, 61, 67};
        return calculateAspectRatio(landmarks, mouthIndices, 0.8);
    }
};

#endif // CALCULATEAR_HPP