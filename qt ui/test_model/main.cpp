#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <mainwindow.h>
#include <QApplication>
#include <QGuiApplication>

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QApplication app(argc, argv);

    MainWindow *mw = new MainWindow;

    return app.exec();
}


