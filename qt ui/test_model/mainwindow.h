#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QLabel>
#include <QPushButton>
#include <QApplication>
#include <QTimer>
#include <QObject>
#include <QVBoxLayout>
#include <QQmlEngine>
#include <QQmlComponent>
#include <QHBoxLayout>
#include <QPainter>
#include <QKeyEvent>
#include <QThread>
#include <QProcess>
#include <QDateTime>
#include "camerahandler.h"
#include "messagewidget.h"

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    QThread* thread;
    QProcess *proc;
    QLabel *label;
    bool isStarted = false;
    CameraHandler *handler;
    cv::Mat frame;
    int currentSegment = -3;
    int currentX = -3;
    int currentY = -3;
    int currentTime = -3;
    double gazeVelocity = -3;
    MessageWidget *msg;

public slots:
    void setCalibrationImage(QPixmap);
    void calibrationDone();
    void predictGaze();
    void forceClose();

protected:
    void paintEvent(QPaintEvent* event);
    void keyPressEvent(QKeyEvent* event);
};

#endif // MAINWINDOW_H
