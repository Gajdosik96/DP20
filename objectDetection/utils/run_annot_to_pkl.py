import xml.etree.ElementTree as ET
import dlib

from pathlib import Path
from helpers.pickleHelper import PickleHelper

PATH_TO_ANNOT_FILE = Path("data/final_data/practical_part/face_ann")
OUTPUT_PATH = Path("data/final_data/practical_part/face_ann_in_pkl")


def run_script():
    pkl = {}
    for file in PATH_TO_ANNOT_FILE.glob("*.xml"):
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
        PickleHelper.save(path=(OUTPUT_PATH/Path(f"gt_250_faces.pkl")).__str__(), data=pkl)


if __name__ == '__main__':
    run_script()

    # data = PickleHelper.load(path=(OUTPUT_PATH/Path(f"gt_250_faces.pkl")).__str__())
    # print("End")

