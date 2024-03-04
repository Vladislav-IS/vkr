#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent) : QWidget(parent)
{
    QFile gaze_coords("gaze_coords.txt");
    qDebug() << gaze_coords.open(QIODevice::Append | QIODevice::Text);
    QString str = "Начало нового сеанса: "+QDate::currentDate().toString("dd.MM.yyyy")+"\nвремя,координата X (см),координата Y (см),изменение расстояния (см)\n";
    qDebug() << gaze_coords.write(str.toStdString().c_str());
    gaze_coords.close();
    thread = new QThread;
    handler = new CameraHandler;
    handler->moveToThread(thread);
    connect(thread, SIGNAL(started()), handler, SLOT(started()));
    connect(thread, SIGNAL(finished()), handler, SLOT(finished()));
    connect(handler, &CameraHandler::sendPixmap, this, &MainWindow::setCalibrationImage);
    setWindowTitle("Калибровка положения");
    setFixedSize(1280/2,410);
    label = new QLabel(this);
    label->setAlignment(Qt::AlignCenter);
    QVBoxLayout *layout = new QVBoxLayout;
    QPushButton *button = new QPushButton(this);
    button->setText("OK");
    button->setFixedSize(100,30);
    QLabel *txt = new QLabel("Калибровка веб-камеры. Лицо должно располагаться в рамке.");
    txt->setAlignment(Qt::AlignCenter);
    QHBoxLayout *hl = new QHBoxLayout;
    hl->addStretch();
    hl->addWidget(button);
    hl->addStretch();
    layout->addWidget(txt);
    layout->addWidget(label);
    layout->addLayout(hl);
    setLayout(layout);
    thread->start();
    show();
    connect(button, &QPushButton::clicked, this, &MainWindow::calibrationDone);
}

void MainWindow::setCalibrationImage(QPixmap pixmap)
{
    label->setPixmap(pixmap);
}

void MainWindow::predictGaze()
{
    if (msg->isVisible()) msg->close();
    QString reply = QString(proc->readAll());
    qDebug() << reply;
    QString segment = reply.split(",").at(0);
    QString x = reply.split(",").at(1);
    QString y = reply.split(",").at(2).split(")").at(0);
    qDebug() << "SXY" << segment << x << y;
    if (segment.at(0) != "(") return;
    if (currentSegment >= 0)
        gazeVelocity = sqrt(pow(round(x.toDouble()*1600)-currentX,2) + pow(round(y.toDouble()*900)-currentY,2))*0.26/10;
    else
        gazeVelocity = 0;
    currentSegment = segment.remove(0,1).toInt();
    currentX = round(x.toDouble()*1600);
    currentY = round(y.toDouble()*900);//y.remove(")").toInt();
    qint64 time = QDateTime::currentMSecsSinceEpoch();
    qint64 secs = time/1000;
    qint64 msecs = time%1000;
    qDebug() << time << secs << msecs;
    QString str = QDateTime::fromSecsSinceEpoch(secs).time().toString("hh:mm:ss")+" и "+QString::number(msecs)+" мсек,"+
            QString::number(currentX*0.26/10)+","+QString::number(currentY*0.26/10)+","+QString::number(gazeVelocity)+"\n";
    if (currentSegment >= 0)
    {
        //QFile gaze_coords("gaze_coords.txt");
        //gaze_coords.open(QIODevice::Append | QIODevice::Text);
        //gaze_coords.write(str.toStdString().c_str());
        //gaze_coords.close();
    }
    update();
}

void MainWindow::calibrationDone()
{
    msg = new MessageWidget;
    msg->setFixedSize(300,50);
    msg->setWindowFlags(Qt::CustomizeWindowHint);
    connect(msg, &MessageWidget::timeToClose, this, &MainWindow::forceClose);
    disconnect(handler, &CameraHandler::sendPixmap, this, &MainWindow::setCalibrationImage);
    qDebug() << "NOOOOOOOOOOOOOOOOOOO";
    isStarted = true;
    qDeleteAll(findChildren<QWidget *>(QString(), Qt::FindDirectChildrenOnly));
    proc = new QProcess;
    proc->start("python predict.py");
    proc->waitForStarted();
    connect(proc, &QProcess::readyReadStandardOutput, this, &MainWindow::predictGaze);
    setWindowState(Qt::WindowFullScreen);
    showFullScreen();
    msg->move(width()/2-msg->width()/2, height()/2);
    msg->show();
    handler->releaseCap();
    thread->quit();
    thread->wait();
}

MainWindow::~MainWindow()
{
    thread->quit();
    thread->wait();
    if (proc)
    {
        QProcess::execute("taskkill /IM python.exe /F");
        qDebug() << "kill";
        proc->kill();
        delete proc;
    }
    if (handler) handler->~CameraHandler();
}

void MainWindow::paintEvent(QPaintEvent *event)
{
    if (isStarted)
    {
        QPainter painter(this);
        QPen pen;
        pen.setColor("black");
        pen.setWidth(3);
        painter.setPen(pen);
        QPolygon triangle;
        triangle << QPoint(width()/2-100,height()/2+100) << QPoint(width()/2+100,height()/2+100) << QPoint(width()/2,height()/2-100);
        QPainterPath path;
        path.addPolygon(triangle);
        painter.fillPath(path, Qt::blue);
        painter.drawPolygon(triangle);
        QRect square1;
        square1.setCoords(triangle.at(0).x()-300, height()/2-100, triangle.at(0).x()-100, height()/2+100);
        painter.fillRect(square1, Qt::green);
        painter.drawRect(square1);
        QRect square2;
        square2.setCoords(triangle.at(1).x()+100, height()/2-100, triangle.at(1).x()+300, height()/2+100);
        painter.fillRect(square2, Qt::red);
        painter.drawRect(square2);
        if (currentSegment >= 0)
        {
            int x = currentSegment%4;
            int y = currentSegment/4;
            QRect rect;
            rect.setCoords(x*400, y*300, x*400+400, y*300+300);
            painter.setPen(QPen(QBrush(),0));
            //painter.setBrush(QBrush(QColor(128, 128, 255, 128)));
            //painter.drawEllipse(QPoint(currentX, currentY), 146, 146);
            painter.fillRect(rect, QBrush(QColor(128, 128, 255, 128)));
            painter.drawRect(rect);
            if (gazeVelocity > 3.8)
            {
                QPolygon activity;
                activity << QPoint(0,0) << QPoint(0,height()) << QPoint(width(),height()) << QPoint(width(),0);
                pen.setColor("green");
                pen.setWidth(40);
                painter.setPen(pen);
                painter.setBrush(QBrush());
                painter.drawPolygon(activity);
            }
        }
        else if (currentSegment == -1 || currentSegment == -4)
        {
            QPolygon screen_frame;
            screen_frame << QPoint(0,0) << QPoint(0,height()) << QPoint(width(),height()) << QPoint(width(),0);
            pen.setColor("red");
            pen.setWidth(40);
            painter.setPen(pen);
            painter.drawPolygon(screen_frame);
        }
    }
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    if (isStarted && event->key() == Qt::Key::Key_Escape)
    {
        QFile gaze_coords("gaze_coords.txt");
        QString str = "Окончание сеанса\n";
        gaze_coords.open(QIODevice::Append | QIODevice::Text);
        gaze_coords.write(str.toStdString().c_str());
        gaze_coords.close();
        if (proc)
        {
            qDebug() << "kill";
            proc->kill();
            proc->startDetached("taskkill /IM python.exe /F");
            proc->waitForFinished();
            proc->kill();
            delete proc;
        }
        if (handler) handler->~CameraHandler();
        close();
    }
}

void MainWindow::forceClose()
{
    msg->close();
    if (proc)
    {
        qDebug() << "kill";
        proc->kill();
        proc->startDetached("taskkill /IM python.exe /F");
        proc->waitForFinished();
        proc->kill();
        delete proc;
    }
    if (handler) handler->~CameraHandler();
    close();
}
