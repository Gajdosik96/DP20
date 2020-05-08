import random
import numpy as np
import cv2

from utils.visualizer import Drawer
from utils.pickleHelper import PickleHelper
from utils.loadPbtxt import LoadPbtxt
from pathlib import Path

_FEATURES = ["FACE", "COCO"]


class PlotProcess:
    def __init__(self, path_to_pkl_coco, path_to_pkl_face, coco_label: str="data/labels/mscoco_label_map.pbtxt", width: int=320, height: int=320):
        self.basename_rgb = Path("dataset/colorization_task_result/final")
        self.basename_bw = Path("data/final_data/bw_in_orig_size")

        self.coco_label = LoadPbtxt().parser(path=coco_label)

        self.data_coco = PickleHelper.load(str(path_to_pkl_coco).replace("\\", "/"))
        self.data_face = PickleHelper.load(str(path_to_pkl_face).replace("\\", "/"))

        self.width = width
        self.height = height

    def random_n_random(self, coco: bool=False, face: bool=True, samples_to_show: int=1, prob_tresh_coco: float=0.5, prob_tresh_face: float=0.5):
        for i in range(samples_to_show):
            pick_random_index = random.randint(0, len(self.data_coco.keys()) - 1)
            pick_random_index = 286
            if coco:
                img_name = list(self.data_coco.keys())[pick_random_index]
                print(img_name)
                img_data = self.data_coco[img_name]

                rgb_path = str(self.basename_rgb / img_name).replace("\\", "/")
                bw_path = str(self.basename_bw / img_name).replace("\\", "/")

                load_img_rgb = cv2.imread(rgb_path)
                load_img_bw = cv2.imread(bw_path)

                rgb_data = img_data['rgb']
                bw_data = img_data['bw']

                rgb_frame, detections_rgb = self.draw_coco_data_to_img(frame=load_img_rgb, img_data=rgb_data, prob_tresh=prob_tresh_coco, feature="COCO")
                bw_frame, detections_bw = self.draw_coco_data_to_img(frame=load_img_bw, img_data=bw_data, prob_tresh=prob_tresh_coco, feature="COCO")

                self.show(frame_rgb=rgb_frame, frame_bw=bw_frame, feature="COCO", sample=pick_random_index, detections_bw=detections_bw, detections_rgb=detections_rgb)

            if face:
                img_name = list(self.data_face.keys())[pick_random_index]
                img_data = self.data_face[img_name]

                rgb_path = str(self.basename_rgb / img_name).replace("\\", "/")
                bw_path = str(self.basename_bw / img_name).replace("\\", "/")

                load_img_rgb = cv2.imread(rgb_path)
                load_img_bw = cv2.imread(bw_path)

                rgb_data = img_data['rgb']
                bw_data = img_data['bw']

                rgb_frame, detections_rgb = self.draw_coco_data_to_img(frame=load_img_rgb, img_data=rgb_data, prob_tresh=prob_tresh_face, feature="FACE")
                bw_frame, detections_bw = self.draw_coco_data_to_img(frame=load_img_bw, img_data=bw_data, prob_tresh=prob_tresh_face, feature="FACE")

                self.show(frame_rgb=rgb_frame, frame_bw=bw_frame, feature="FACE", sample=pick_random_index, detections_bw=detections_bw, detections_rgb=detections_rgb)


    def show(self, frame_bw: np.array = None, frame_rgb: np.array = None, feature: str = None, sample: int = None, detections_bw: int = None, detections_rgb: int = None):
        if feature not in _FEATURES:
            print(f"Incorrect sample type !!!")

        if feature is "COCO":
            cv2.namedWindow(f"{feature}_bw_{sample}_det_{detections_bw}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{feature}_bw_{sample}_det_{detections_bw}", self.width, self.height)
            cv2.imshow(f"{feature}_bw_{sample}_det_{detections_bw}", frame_bw)

            cv2.namedWindow(f"{feature}_rgb_{sample}_det_{detections_rgb}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{feature}_rgb_{sample}_det_{detections_rgb}", self.width, self.height)
            cv2.imshow(f"{feature}_rgb_{sample}_det_{detections_rgb}", frame_rgb)

            cv2.waitKey()
            cv2.destroyWindow(f"{feature}_rgb_{sample}_det_{detections_rgb}")
            cv2.destroyWindow(f"{feature}_bw_{sample}_det_{detections_bw}")

        if feature is "FACE":
            cv2.namedWindow(f"{feature}_bw_{sample}_det_{detections_bw}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{feature}_bw_{sample}_det_{detections_bw}", self.width, self.height)
            cv2.imshow(f"{feature}_bw_{sample}_det_{detections_bw}", frame_bw)

            cv2.namedWindow(f"{feature}_rgb_{sample}_det_{detections_rgb}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{feature}_rgb_{sample}_det_{detections_rgb}", self.width, self.height)
            cv2.imshow(f"{feature}_rgb_{sample}_det_{detections_rgb}", frame_rgb)

            cv2.waitKey()
            cv2.destroyWindow(f"{feature}_rgb_{sample}_det_{detections_rgb}")
            cv2.destroyWindow(f"{feature}_bw_{sample}_det_{detections_bw}")

    def draw_coco_data_to_img(self, frame: np.ndarray, img_data, prob_tresh: float, feature: str = None):
        if feature not in _FEATURES:
            print(f"Incorrect sample type !!!")

        if feature is "COCO":
            num_detections, detection_boxes, detection_scores, detection_classes = img_data["num_detections"], img_data[
                "detection_boxes"], img_data["detection_scores"], img_data["detection_classes"]

            print(f"Boxes: {detection_boxes[0][:3]}, Scores: {detection_scores[0][:3]}")

            if num_detections[0] < 1:
                return frame, 0

            width = frame.shape[1]
            height = frame.shape[0]
            for box, score, c in zip(detection_boxes[0], detection_scores[0], detection_classes[0]):
                if score < prob_tresh:
                    continue
                color = Drawer.create_unique_color_uchar(tag=int(c))
                name = f"{self.coco_label[int(c)]}_{score:.2f}"

                x, y, w, h = int(box[1] * width), int(box[0] * height), int((box[3] - box[1]) * width), int((box[2] - box[0]) * height)
                Drawer.rectangle(frame=frame, x=x, y=y, w=w, h=h, label=name, color=color, thickness=4)
            return frame, num_detections[0]

        if feature is "FACE":

            boxes, scores = img_data[0], img_data[1]

            print(f"Boxes: {boxes[:3]}, Scores: {scores[:3]}")


            if len(boxes) < 1:
                return frame, 0

            for box, score in zip(boxes, scores):
                if score < prob_tresh:
                    continue
                color = Drawer.create_unique_color_uchar(tag=1)
                name = f"Face_{score:.2f}"

                Drawer.rectangle(frame=frame, x=box.left(), y=box.top(), w=box.width(), h=box.height(), label=name, color=color, thickness=4)
            return frame, len(boxes)



if __name__ == '__main__':
    path_to_pkl_coco = Path("data/detections_task_output/coco_90_class_obj_detections/COCO.pkl")
    path_to_pkl_face = Path("data/detections_task_output/face_detection/FACE.pkl")

    plotter = PlotProcess(path_to_pkl_coco, path_to_pkl_face)
    plotter.random_n_random(coco=True, face=False, samples_to_show=1, prob_tresh_coco=0.7, prob_tresh_face=-0.5)

