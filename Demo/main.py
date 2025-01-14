import datetime

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

model=YOLO('../models/best.pt')
model_2=YOLO('../models/yolov8m-seg.pt')
model_3=YOLO('../models/yolov8m-seg.pt')

def draw_text(text,m_aoi_d_x1,m_aoi_d_y1,annotated_frame,color=(255,0,0),bg_color=(255,255,255)):
    """text draw"""

    text_position = (m_aoi_d_x1 + 50, m_aoi_d_y1 + 40)

    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    # Define the rectangle background
    rect_top_left = (text_position[0], text_position[1] - text_height - baseline)
    rect_bottom_right = (text_position[0] + text_width, text_position[1] + baseline)

    # Draw the rectangle (filled with a background color, e.g., white)
    cv2.rectangle(annotated_frame, rect_top_left, rect_bottom_right, bg_color, cv2.FILLED)

    # Draw the text over the rectangle
    cv2.putText(annotated_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 1)
    return annotated_frame

def main():
    """Main function"""
    # Replace with your video source, e.g., '0' for webcam or a video file path
    video_source = "./video/demo.mov"
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    target_fps = 10
    video_writer = cv2.VideoWriter("./out/output.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               target_fps,
                               (1200,720))
    frame_skip = int(fps / target_fps)
    frame_count = 0
    tracker=False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed. looping back")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            # break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        frame=cv2.resize(frame,(1200,720))
        x_l, y_l, w_l, h_l = 222, 428, 27, 27  # AOI for Light left
        x1_l, y1_l = int(x_l-w_l/2), int(y_l-h_l/2)
        x2_l, y2_l = int(x_l+w_l/2), int(y_l+h_l/2)
        cv2.rectangle(frame, (x1_l, y1_l), (x2_l, y2_l), (255, 0, 0), 2)
        x_r, y_r, w_r, h_r = 285, 430, 27, 27  # AOI for Light right
        x1_r, y1_r = int(x_r-w_r/2), int(y_r-h_r/2)
        x2_r, y2_r = int(x_r+w_r/2), int(y_r+h_r/2)
        cv2.rectangle(frame, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 2)
        tracks=model.track(frame,conf=0.5, iou=0.40, persist=True,
                         device='mps',retina_masks=True, show=False)
        annotated_frame=tracks[0].plot(labels=False,line_width=2)
        m_aoi_u_x1, m_aoi_u_y1, m_aoi_u_x2, m_aoi_u_y2 = 280, 50, 320, 260
        aoi_u_polygon = Polygon([(m_aoi_u_x1, m_aoi_u_y1), (m_aoi_u_x2, m_aoi_u_y1), (m_aoi_u_x2, m_aoi_u_y2), (m_aoi_u_x1, m_aoi_u_y2)])
        cv2.rectangle(annotated_frame, (m_aoi_u_x1, m_aoi_u_y1), (m_aoi_u_x2, m_aoi_u_y2), (255, 255, 255), 1) # BOOM UP GATE AOI
        m_aoi_d_x1, m_aoi_d_y1, m_aoi_d_x2, m_aoi_d_y2 = 300, 550, 680, 550
        aoi_d_polygon = Polygon([(m_aoi_d_x1, m_aoi_d_y1), (m_aoi_d_x2, m_aoi_d_y1), (m_aoi_d_x2, m_aoi_d_y2), (m_aoi_d_x1, m_aoi_d_y2)])
        cv2.rectangle(annotated_frame, (m_aoi_d_x1, m_aoi_d_y1), (m_aoi_d_x2, m_aoi_d_y2), (0, 0, 255), 1) # BOOM DOWN GATE AOI

        polygon_x = [277, 351, 400, 411, 500, 544, 609, 653, 744, 829, 927, 976, 827, 820, 839, 804, 713, 681,
616, 600, 606, 602, 572, 504, 493, 476, 291]
        polygon_y = [524, 589, 647, 664, 661, 654, 643, 633, 624, 606, 594, 580, 548, 541, 522, 499, 501, 513,
 510 ,501 ,499 ,487 ,480, 494, 489, 478 ,480]
        
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [np.array([polygon_x, polygon_y]).T], color=(255, 255, 0))
        cv2.addWeighted(overlay, 0.3, annotated_frame, 1 - 0.5, -2, annotated_frame)
        human=model_2.track(annotated_frame,classes=[0],conf=0.5, iou=0.40, persist=True,
                    device='mps',retina_masks=True, show=False)
        for track in tracks:
            for obj in track.boxes.xyxy:
                bbox_polygon = Polygon([(obj[0], obj[1]), (obj[2], obj[1]), (obj[2], obj[3]), (obj[0], obj[3])])
                if bbox_polygon.intersects(aoi_u_polygon):
                    print("Boom Gate is Up.")
                    text = "BOOM GATE UP"
                    draw_text(text,m_aoi_u_x1,m_aoi_u_y1,annotated_frame)

                elif bbox_polygon.intersects(aoi_d_polygon):
                    print("Boom Gate is Down.")
                    text = "BOOM GATE DOWN"
                    draw_text(text,m_aoi_d_x1,m_aoi_d_y1,annotated_frame)
                    danger_zone_roi = Polygon(np.array([polygon_x, polygon_y]).T)
                    # Iterate through the tracks of vehicles (or objects)
                    for t in human:
                        if t:
                            for mask in t.masks:  # Assuming `track.masks` contains the segmentation masks
                                # Extract contours from the binary mask
                                binary_mask = mask.data.cpu().numpy().astype(np.uint8)
                                contours, _ = cv2.findContours(binary_mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                # Convert contours to polygons
                                mask_polygons = [Polygon(c[:, 0, :]) for c in contours if len(c) > 2]

                                # Combine multiple polygons if necessary
                                mask_polygon = unary_union(mask_polygons)

                                # Check if any part of the mask polygon intersects with the danger zone
                                if mask_polygon.intersects(danger_zone_roi):
                                    tracker=True
            if tracker:
                text = "Unsafe Condition"
                draw_text(text=text,m_aoi_d_x1=900,m_aoi_d_y1=140,annotated_frame=annotated_frame,color=(255,255,255),bg_color=(0,0,255))

        vehicles=model_2.track(human[0].plot(labels=False,line_width=2),classes=[1,2,3,5,7,15,16],conf=0.5, iou=0.40, persist=True,
                         device='mps',retina_masks=True, show=False)
        annotated_frame=vehicles[0].plot(labels=False,line_width=2)
        cv2.imshow("Frame",annotated_frame)
        video_writer.write(annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


