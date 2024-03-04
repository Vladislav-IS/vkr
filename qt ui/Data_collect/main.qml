import QtQuick 2.6
import QtQuick.Window 2.2
import QtQuick.Controls 1.1
import QtQuick.Layouts 1.0
import QtQuick.Dialogs 1.0
import DataLoader 1.0

ApplicationWindow {
    id: app_win
    width: 640
    height: 480
    visible: true
    visibility: "FullScreen"
    property int count: 0

    DataLoader {
        id: data_loader
    }

    Item {
        focus: true
        Keys.onEscapePressed:  {app_win.close();}
        Keys.onDeletePressed: {
            if (app_win.count==0) return;
            track_point.oldX.pop();
            track_point.oldY.pop();
            app_win.count--;
            data_loader.deleteLast();
            if (track_point.oldX[app_win.count] != -1)
            {
                track_point.x = track_point.oldX[app_win.count];
                track_point.y = track_point.oldY[app_win.count];
                if (!track_point.visible) track_point.visible = true;
            }
            else
                track_point.visible = false;
        }
        Keys.onReturnPressed: {
            data_loader.getFrameAndCoords(-1,-1);
            track_point.oldX.push(-1);
            track_point.oldY.push(-1);
            track_point.visible = false;
            app_win.count++;
        }
    }

    MouseArea {
        id: ma_app_win
        anchors.fill: parent
        onClicked: {
            data_loader.getFrameAndCoords(mouseX / app_win.width, mouseY / app_win.height);
            if (!track_point.visible) track_point.visible = true;
            console.log(mouseX, mouseY);
            track_point.x = mouseX - track_point.width / 2;
            track_point.y = mouseY - track_point.height / 2;
                track_point.oldX.push(track_point.x);
                track_point.oldY.push(track_point.y);
            app_win.count++;
        }
    }

    Image {
        id: img
        source: "fon.jpg"
        width: app_win.width
        height: app_win.height

        Text {
            id: countText
            text: "Сделано записей в текущем сеансе: "+Number(app_win.count)
            x: 30
            y: 100
            color: "white"
            font.pixelSize: 24
        }

        Text {
            id: annotationText
            visible: true
            text: "ЛКМ - записать фото и координаты взгляда на экране,\nDelete - удалить последнюю запись в текущем сеансе,\nEnter - записать фото и координаты взгляда вне экрана,\nEscape - закрыть программу"
            x: 30
            y: 650
            color: "white"
            font.pixelSize: 24
            lineHeight: 1
        }
    }

    Rectangle {
        id: track_point
        color: "red"
        width: 10
        height: 10
        x: 0
        y: 0
        property var oldX: [-1]
        property var oldY: [-1]
        visible: false
    }
}
