import cv2
from pathlib import Path
import numpy as np
import collections


def __conts_to_rects(conts):
    result = np.column_stack((conts.min(axis=(0,1)), conts.max(axis=(0,1))))

    minx, maxx, miny, maxy = result[0][0], result[0][1], result[1][0], result[1][1]

    print(f"minx: {minx}, maxx: {maxx}, miny: {miny}, maxy: {maxy}")

    return minx, miny, maxx, maxy


def postProcess(mask, orig = None, name: str=None):
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
    mask2grey = mask * 255

    filtered = mask2grey
    for i in range(10):
        filtered = cv2.medianBlur(filtered, 3)

    # Vypln diery v maske
    kernel = np.ones((25, 25), np.uint8)
    open = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel, iterations=4)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Get points
    rel_left, rel_top, rel_right, rel_bottom = getMaximalAppearanceOfEachAxis(mask=closing)
    print(f"left: {rel_left}, top: {rel_top}, right: {rel_right}, bottom: {rel_bottom}")

    # orig = orig[int(rel_top * orig.shape[1]):int(rel_bottom * orig.shape[1]), int(rel_left * orig.shape[0]):int(rel_right * orig.shape[0])]
    orig = orig[rel_top:rel_bottom, rel_left:rel_right]

    return orig


def getMaximalAppearanceOfEachAxis(mask):
    top = mask.argmax(axis=(0))
    left = mask.argmax(axis=(1))
    bottom = (mask.shape[0] - 1) - mask[::-1, :-1].argmax(axis=(0))
    right = (mask.shape[1] - 1) - mask[::-1, ::-1].argmax(axis=(1))

    top_values = mask.max(axis=(0))
    left_values = mask.max(axis=(1))
    bottom_values = mask[::-1, :-1].max(axis=(0))
    right_values = mask[::-1, ::-1].max(axis=(1))

    top[top_values == 0] = -1
    left[left_values == 0] = -1
    bottom[bottom_values == 0] = -1
    right[right_values == 0] = -1

    top_counter = collections.Counter(top)
    left_counter = collections.Counter(left)
    bottom_counter = collections.Counter(bottom)
    right_counter = collections.Counter(right)

    left, top, right, bottom = AxisCornersHelp(left_counter), AxisCornersHelp(top_counter), AxisCornersHelp(right_counter), AxisCornersHelp(bottom_counter)
    return left, top, right, bottom


def AxisCornersHelp(counter):
    value = 0
    while 1:
        top_freq = counter.most_common(len(counter.values()))
        for item in top_freq:
            if item[0] == -1:
                continue
            value = item[0]
            break
        break
    return value


if __name__ == '__main__':
    # Path to folder with output segmentated mask
    path2mfolder = Path("dataset/unet_segmetation_task/adepts/adepts_mask")
    # Path to folder with original mask
    path2folder = Path("dataset/unet_segmetation_task/adepts/adepts")

    get_mask = lambda x: path2mfolder / f"{x.stem}_m{x.suffix}"

    for img in path2folder.glob("*.jpg"):
        path2mask = str(get_mask(img))

        mask = cv2.imread(path2mask.replace("\\", "/"), 0)
        image = cv2.imread(str(img).replace("\\", "/"))

        print(f"Processing image: {img}, shape: {image.shape}")
        print(f"Processing mask: {path2mask}, shape: {mask.shape}")

        final_image = postProcess(mask=mask, orig=image, name=img.stem)

        cv2.imwrite(f"dataset/unet_segmetation_task/adepts/after2/{img.name}", final_image)
        # break


