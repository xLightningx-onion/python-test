from ultralytics import YOLO
from IPython.display import [display, Image]

!yolo task=detect mode=predict model=yolov8n.py conf=0.25  source='rear-view-truck.jpeg'
