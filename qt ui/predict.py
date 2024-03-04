import torch
import numpy as np
import argparse
import cv2
import mediapipe
import torchvision


class HybridNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_eye = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.first_eye_fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*9*9, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.second_eye = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())
        self.second_eye_fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=64*9*9, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.coords = torch.nn.Sequential(
            torch.nn.Linear(in_features=4*2, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU())
        self.union = torch.nn.Sequential(
            torch.nn.Linear(in_features=128*3, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=2),
            torch.nn.ReLU())

    def forward(self, imgs, coords):
        x_1 = self.first_eye(imgs[0].unsqueeze(0))
        x_2 = self.second_eye(imgs[1].unsqueeze(0))
        x_1 = self.first_eye_fc(x_1.reshape(x_1.shape[0], -1))
        x_2 = self.second_eye_fc(x_2.reshape(x_2.shape[0], -1))
        if coords.size(0) > 1:
            coords = torch.cat(tuple(coords), dim=-1)
        x_3 = self.coords(coords)
        return self.union(torch.cat((x_1, x_2, x_3), dim=-1))


class BaselineNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_eye = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU())
        self.second_eye = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU())
        self.coords = torch.nn.Sequential(
            torch.nn.Linear(in_features=4*2, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU())
        self.union = torch.nn.Sequential(
            torch.nn.Linear(in_features=16+4096, out_features=8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4, out_features=2),
            torch.nn.ReLU())

    def forward(self, imgs, coords):
        x_1 = self.first_eye(imgs[0].unsqueeze(0))
        x_2 = self.second_eye(imgs[1].unsqueeze(0))
        x_1 = x_1.reshape(x_1.shape[0], -1)
        x_2 = x_2.reshape(x_2.shape[0], -1)
        if coords.size(0) > 1:
            coords = torch.cat(tuple(coords),dim=-1)
        x_3 = self.coords(coords)
        return self.union(torch.cat((x_1, x_2, x_3), dim=-1)).squeeze()


class OneEyeNetAsRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first_eye = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU())
        self.coords = torch.nn.Sequential(
            torch.nn.Linear(in_features=6, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=16),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU())
        self.union = torch.nn.Sequential(
            torch.nn.Linear(in_features=16+2048, out_features=8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=4),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4, out_features=2),
            torch.nn.ReLU())

    def forward(self, imgs, coords):
        x_1 = self.first_eye(imgs[1].unsqueeze(0))
        x_1 = x_1.reshape(x_1.shape[0], -1)
        if coords.size(0) > 1:
            coords = torch.cat(tuple(coords), dim=-1)
        x_2 = self.coords(coords)
        return self.union(torch.cat((x_1, x_2), dim=-1))


RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
R_H_LEFT = 33
R_H_RIGHT = 133
R_H_PUPIL = 468
L_H_LEFT = 362
L_H_RIGHT = 263
L_H_PUPIL = 473
LEFT_EYE_3_POINTS = [473, 263, 362]
RIGHT_EYE_3_POINTS = [468, 133, 33]
LEFT_EYE_6_POINTS = [385, 380, 387, 373, 362, 263]
RIGHT_EYE_6_POINTS = [160, 144, 158, 153, 33, 133]



transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


def get_ear(mesh, idxs):
    return (distance(mesh[idxs[0]], mesh[idxs[1]]) + distance(mesh[idxs[2]], mesh[idxs[3]]))/ 2 / distance(mesh[idxs[4]], mesh[idxs[5]])


def get_is_darker(gray_frame, mesh, idxs):
    pupils_darker = []
    pupils_darker.append(gray_frame[mesh[idxs[0]][1],mesh[idxs[0]][0]]<= gray_frame[mesh[idxs[1]][1], mesh[idxs[1]][0]])
    pupils_darker.append(gray_frame[mesh[idxs[0]][1],mesh[idxs[0]][0]]<= gray_frame[mesh[idxs[2]][1], mesh[idxs[2]][0]])
    return sum(pupils_darker)


def setupModels():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    mp_face_mesh = mediapipe.solutions.face_mesh
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = OneEyeNetAsRegression().to(device)
    model.load_state_dict(torch.load('./best_one_eye.pt', map_location=device))
    model.eval()
    return model, mp_face_mesh, cap


def predictGazeInLoop(model, mesh, cap):
    ret, frame = cap.read()
    mp_face_mesh = mediapipe.solutions.face_mesh
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mesh.FaceMesh(
              max_num_faces=1,
              refine_landmarks=True,
              min_detection_confidence=0.65,
              min_tracking_confidence=0.65) as face_mesh:
        res = face_mesh.process(frame)
        if not res.multi_face_landmarks:
            return -1,-1,-1
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            h, w = frame.shape[:2]
            norm_meshpoints = np.array([np.array([p.x, p.y]) for p in res.multi_face_landmarks[0].landmark])
            meshpoints = np.array([np.array([p.x * w, p.y * h]).astype(int) for p in res.multi_face_landmarks[0].landmark])
            is_darker = get_is_darker(gray_frame, meshpoints, LEFT_EYE_3_POINTS) + get_is_darker(gray_frame, meshpoints,
                                                                                                 RIGHT_EYE_3_POINTS)
            ear = get_ear(meshpoints, LEFT_EYE_6_POINTS) + get_ear(meshpoints, RIGHT_EYE_6_POINTS)
            #if is_darker < 3:
            #    return -2,-2,-2
            if ear / 2 < 0.2:
                return -4,-4,-4
            else:
                left_eye = frame[
                           int(meshpoints[L_H_PUPIL][1]-abs(meshpoints[L_H_LEFT][0]-meshpoints[L_H_RIGHT][0])/2):
                           int(meshpoints[L_H_PUPIL][1]+abs(meshpoints[L_H_LEFT][0]-meshpoints[L_H_RIGHT][0])/2),
                           meshpoints[L_H_LEFT][0]:meshpoints[L_H_RIGHT][0]]
                right_eye = frame[
                           int(meshpoints[R_H_PUPIL][1]-abs(meshpoints[R_H_RIGHT][0]-meshpoints[R_H_LEFT][0])/2):
                           int(meshpoints[R_H_PUPIL][1]+abs(meshpoints[R_H_LEFT][0]-meshpoints[R_H_RIGHT][0])/2),
                           meshpoints[R_H_LEFT][0]:meshpoints[R_H_RIGHT][0]]
                cv2.imwrite('re.jpg',right_eye)
                cv2.imwrite('le.jpg', left_eye)
                left_eye = transform(cv2.resize(cv2.flip(np.array(cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB) / 255, dtype='float32'),1), (128,128), interpolation=cv2.INTER_AREA))
                right_eye = transform(cv2.resize(np.array(cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB) / 255, dtype='float32'), (128,128), interpolation=cv2.INTER_AREA))
                coords = torch.cat([torch.FloatTensor(norm_meshpoints[R_H_PUPIL]),
                                   torch.FloatTensor(norm_meshpoints[R_H_LEFT]),
                                   torch.FloatTensor(norm_meshpoints[R_H_RIGHT])], dim=-1)
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                left_eye = left_eye.to(device)
                right_eye = right_eye.to(device)
                coords = coords.to(device).unsqueeze(0)
                with torch.no_grad():
                    outputs = model([left_eye, right_eye], coords).squeeze()
                segment = int(torch.nn.functional.relu(outputs[1])*900)//300*4 + int(torch.nn.functional.relu(outputs[0])*1600)//400
                return min(11, segment), float(outputs[0]), float(outputs[1])


if __name__ == "__main__":
    model, mesh, cap = setupModels()
    while True:
        print(predictGazeInLoop(model,mesh,cap), flush=True)
