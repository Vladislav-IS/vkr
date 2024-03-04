QT += quick widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
        dataloader.cpp \
        main.cpp

RESOURCES += qml.qrc \
    qrc.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    dataloader.h

INCLUDEPATH += D:\opencv\opencv\build\include

LIBS += D:\opencv\opencv-build\bin\libopencv_core454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_highgui454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_imgcodecs454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_imgproc454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_features2d454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_calib3d454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_objdetect454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_videoio454.dll
LIBS += D:\opencv\opencv-build\bin\libopencv_dnn454.dll

DISTFILES +=
