import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr

def detect_pools(input_image):
    model = YOLO("weights/best.pt")

    results = model.predict(source=input_image, conf=0.6, save=False, verbose=False)

    image = cv2.imread(input_image)

    for result in results:
        for mask in result.masks.xy: 
            mask = np.array(mask, np.int32)  
            mask = mask.reshape((-1, 1, 2)) 
            cv2.polylines(image, [mask], isClosed=True, color=(0, 0, 255), thickness=2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb

def gradio_interface(input_image):
    output_image = detect_pools(input_image)
    return output_image

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(label="Input Image", type="filepath"),
    outputs=gr.Image(label="Output Image with Detected Pools"),
    title="Pool Detection with YOLOv8",
    description="Upload an image to detect pools using a fine-tuned YOLOv8 model. The output will be an image with detected pools outlined in red."
)

iface.launch()