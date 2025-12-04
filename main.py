import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = 'blaze_face_short_range.tflite'

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)
with FaceDetector.create_from_options(options) as detector:
   cap = cv2.VideoCapture(0)
   timestamp = 0
   while cap.isOpened():
       ret,frame = cap.read()
       if not ret:
           break
       
       timestamp +=1
       frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
       mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
       face_detected_result = detector.detect_for_video(mp_image,timestamp)

       if face_detected_result.detections:
           for detection in face_detected_result.detections:
               bbox = detection.bounding_box
               x,y,w,h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

               cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cv2.imshow('Face Detection', frame)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
cap.release()
cv2.destroyAllWindows() 