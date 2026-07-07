import cv2
import torch
from torchvision import transforms
from config import CHARS
from LPRNet import build_lprnet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def deskew_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return plate_crop
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = plate_crop.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(plate_crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def upscale_plate(plate_crop, scale=4):
    return cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)



def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
    num_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if num_contours:
        contour_area = [cv2.contourArea(c) for c in num_contours]
        max_cntr_index = np.argmax(contour_area)
        max_cnt = num_contours[max_cntr_index]
        x, y, w, h = cv2.boundingRect(max_cnt)
        final_img = thresh[y:y+h, x:x+w]
        return final_img
    else:
        return thresh


lpr_model = build_lprnet(lpr_max_len=11, phase=False, class_num=37, dropout_rate=0.5)
lpr_model.load_state_dict(torch.load(
    r"runs\detect\train-4\weights\best_lprnet.pth",
    map_location=device
))
lpr_model = lpr_model.to(device)
lpr_model.eval()

lpr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((24, 94)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def decode_ctc(preds):
    preds     = preds.squeeze(0)
    preds_idx = torch.argmax(preds, dim=0).cpu().numpy()
    blank     = len(CHARS)
    result    = []
    prev      = blank
    for idx in preds_idx:
        if idx != prev and idx < len(CHARS):
            result.append(CHARS[idx])
        prev = idx
    return ''.join(result)

def read_plate_lpr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""
    try:
        deskewed = deskew_plate(plate_crop)
        upscaled = upscale_plate(deskewed, scale=4)
        rgb      = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        tensor   = lpr_transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = lpr_model(tensor)
        return decode_ctc(preds)
    except Exception as e:
        print(f"[LPR ERROR] {e}")
        return ""

def get_plate_color(plate_crop):
    if plate_crop.size == 0:
        return "Unknown"
    resized  = cv2.resize(plate_crop, (120, 40))
    hsv      = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv)[:3]
    hue, sat, val = mean_hsv
    if sat < 80 and val > 100:
        return "White"
    elif 35 <= hue <= 85 and sat > 50:
        return "Green"
    elif hue <= 10 or hue >= 160:
        return "Red"
    elif 100 <= hue <= 140 and sat > 50:
        return "Blue"
    elif 15 <= hue <= 34 and sat > 50:
        return "Yellow"
    else:
        return "Unknown"