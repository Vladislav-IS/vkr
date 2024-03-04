#ifndef DATALOADER_H
#define DATALOADER_H

#include <QObject>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <stdio.h>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QDateTime>
#include <QMouseEvent>
#include <QGuiApplication>
#include <QTimer>

class DataLoader: public QObject
{
    Q_OBJECT
public:
    DataLoader();
    ~DataLoader();
    Q_INVOKABLE void getFrameAndCoords(double mouseX, double mouseY);
    Q_INVOKABLE void deleteLast();

private:
    cv::VideoCapture video_capture;
    cv::Mat frame;
    QString dir_path = "d://Data_collection_1280x720";
    QString csv_file_name = dir_path + "//coords.csv";
    QStringList jpg_file_names = QStringList();
    QTimer* timer;
    int lastIndex = 0;

public slots:
    void getFrameOnTimer();

signals:
    void errorOccured();
};

#endif // DATALOADER_H
