import dlib
import xml.etree.ElementTree as ET

from pathlib import Path
from helpers.pickleHelper import PickleHelper


def intersect_over_union(box1: dlib.rectangle, box2: dlib.rectangle):
    if box1.contains(box2):
        overall_area = box1.area()
    elif box2.contains(box1):
        overall_area = box2.area()
    elif box1.intersect(box2).area():
        overall_area = (box1.area() - box1.intersect(box2).area()) + (box2.area() - box1.intersect(box2).area()) + box1.intersect(box2).area()
    else:
        return 0.0

    return box1.intersect(box2).area() / overall_area


def fill_matrix_with_iou(corr_matrix, bbox_gt, bbox_dt):
    for g_i, b_gt in enumerate(bbox_gt):
        for d_i, b_dt in enumerate(bbox_dt):
            corr_matrix[g_i][d_i] = intersect_over_union(b_gt, b_dt)


def annotations2pkl(path2annot_file: str, output_path: str):
    pkl = {}
    for file in path2annot_file.glob("*.xml"):
        load_file = ET.parse(file)

        fn = load_file.find("filename").text

        new_rects = list()
        for xml_obj in load_file.findall("object"):
            bbox = xml_obj.find("bndbox")

            left, top, right, bottom = round(float(bbox.find("xmin").text)), round(
                float(bbox.find("ymin").text)), round(float(bbox.find("xmax").text)), round(
                float(bbox.find("ymax").text))

            new_face_rect = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
            new_rects.append(new_face_rect)

        pkl[fn] = new_rects

    if pkl:
        PickleHelper.save(path=(output_path/Path(f"gt_250_faces.pkl")).__str__(), data=pkl)

if __name__ == '__main__':
    PATH_TO_ANNOT_FILE = Path("data/final_data/practical_part/face_ann")
    OUTPUT_PATH = Path("data/final_data/practical_part/face_ann_in_pkl")

    annotations2pkl(path2annot_file=PATH_TO_ANNOT_FILE, output_path=OUTPUT_PATH)

    # data = PickleHelper.load(path=(OUTPUT_PATH/Path(f"gt_250_faces.pkl")).__str__())
    # print("End")