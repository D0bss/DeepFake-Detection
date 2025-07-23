from fpdf import FPDF
import datetime
import numpy as np
import cv2
from PIL import Image

def clear_all():
    return None, None, None, None, None, None, None, ""

def draw_boxes_on_image(image, boxes, labels):
    image_np = np.array(image).copy()

    for (box, label) in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return Image.fromarray(image_np)



