from config import ROI_X1, ROI_Y1, ROI_X2, ROI_Y2, ROI2_X1, ROI2_Y1, ROI2_X2, ROI2_Y2

def is_person_shaped(x1, x2, y1, y2):
    w = x2 - x1
    h = y2 - y1
    if w == 0:
        return False
    ratio = h / w
    return ratio > 1.2 and h < 800

def boxes_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    return True

def in_bay(cx, cy, bay=1):
    if bay == 1:
        return ROI_X1 < cx < ROI_X2 and ROI_Y1 < cy < ROI_Y2
    return ROI2_X1 < cx < ROI2_X2 and ROI2_Y1 < cy < ROI2_Y2