import cv2 as cv
import math
import numpy as np
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

pixels_to_cm = 0.5
annotator = sv.BoxAnnotator()
prev_positions = {}
total_distance_moved = {}

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def process_and_render(predictions: dict, video_frame: VideoFrame, video_writer):
    try:
        global prev_positions
        global total_distance_moved
        
        print("Predictions:", predictions)
        
        predictions_list = predictions.get('predictions', [])
        for i, prediction in enumerate(predictions_list):
            print("Prediction:", prediction)
            
            class_name = prediction.get('class', 'Unknown')
            confidence = prediction.get('confidence', 0)
            x, y = prediction.get('x', 0), prediction.get('y', 0)
            width, height = prediction.get('width', 0), prediction.get('height', 0)
            
            print(f"Class: {class_name}\nConfidence: {confidence}")
            print(f"BBox:\n x={x},\n y={y}, \n width={width}, \n height={height}")
            
            cv.putText(video_frame.image, f"{class_name}: ({int(x)},{int(y)})", (10, 20*(i+1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (15, 99, 162), 1)
            
            object_key = f"{class_name}_{i}"

            prev_x, prev_y = prev_positions.get(object_key, (x, y))
            distance_moved_px = calculate_distance(prev_x, prev_y, x, y)
            distance_moved_cm = distance_moved_px * pixels_to_cm
            
            if object_key not in total_distance_moved:
                total_distance_moved[object_key] = 0
            
            total_distance_moved[object_key] += distance_moved_cm
            prev_positions[object_key] = (x, y)
        
            cv.putText(video_frame.image, f"Distancia: {total_distance_moved[object_key]:.2f} cm", (650, 20*(i+1)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (25, 129, 9), 1)
        
        labels = [f"{p['class']}" for p in predictions["predictions"]]
        detections = sv.Detections.from_inference(predictions)
        image = annotator.annotate(
            scene=video_frame.image.copy(), detections=detections, labels=labels
        )
        
        cv.imshow("Predictions", image)
        video_writer.write(image)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q') or not pipeline.video_capture.isOpened():
            cv.destroyAllWindows()
            exit()

    except Exception as e:
        print(f"Error processing frame: {e}")

pipeline = InferencePipeline.init(
    model_id="patos-jatbh/1",
    api_key="5mXTsxXzcJ27MD62MS4F",
    video_reference="pollitos.mp4",
    on_prediction=lambda predictions, video_frame: process_and_render(predictions, video_frame, video_writer)
)

cap = cv.VideoCapture('pollitos.mp4')


frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
output_file = 'pollosprocesados.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
video_writer = cv.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

pipeline.start()
pipeline.join()

cap.release()
video_writer.release()
cv.destroyAllWindows()