import QtQuick 2.6
import QtQuick.Window 2.2
import QtQuick.Controls 2.0

ApplicationWindow {
    id: app_win
    width: 640
    height: 480
    visible: true
    title: qsTr("Test model")
    visibility: "FullScreen"

    Image {
        id: img
        width: parent.width
        height: parent.height
        source: "fon.jpg"
        x: 0
        y: 0

        Rectangle {
            id: track_rec
            x: 0
            y: 0
            width: 400
            height: 300
            visible: true
            color: "green"
            opacity: 0.5
        }

        MouseArea{anchors.fill: parent; onClicked:{img.grabToImage(function(result) {
            result.saveToFile("./something.png")}); /*app_win.close();*/}}
    }

}
