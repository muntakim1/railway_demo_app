import datetime
import os
import queue
import shutil
import threading
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

from utils import RayvanIOConnector

client=RayvanIOConnector("52.169.16.32",username="cisco_cam",password="cisco_cam",topic="5301af67678b61d1478e8204318345d33317/cisco_cam")
client.connect()

def detect_red(frame):
    """detect red light"""
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range of red color in HSV
    lower_red = np.array([170, 180, 100]) # Red Range
    upper_red = np.array([180, 255, 255]) # Red Range

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    # cv2.imshow(name+"_mask",mask)
    # print()
    # print(cv2.countNonZero(mask))
    # Check if any red pixels are present
    if cv2.countNonZero(mask) >= 50:
        return True
    else:
        return False

def detect_left_led_working(im0):
    """Function for detecting the left led"""
    x_l, y_l, w_l, h_l = 222, 428, 27, 27  # AOI for Light left
    x1_l, y1_l = int(x_l-w_l/2), int(y_l-h_l/2)
    x2_l, y2_l = int(x_l+w_l/2), int(y_l+h_l/2)
    cv2.rectangle(im0, (x1_l, y1_l), (x2_l, y2_l), (255, 0, 0), 2)
    cropped_left = im0[y1_l:y2_l, x1_l:x2_l]
    red_detected_left = detect_red(cropped_left)
    # cv2.imwrite('left.png',im0)

    return red_detected_left

def detect_right_led_working(im0):
    """Function for detecting the right led"""
    x_r, y_r, w_r, h_r = 285, 430, 27, 27  # AOI for Light right
    x1_r, y1_r = int(x_r-w_r/2), int(y_r-h_r/2)
    x2_r, y2_r = int(x_r+w_r/2), int(y_r+h_r/2)
    cv2.rectangle(im0, (x1_r, y1_r), (x2_r, y2_r), (255, 0, 0), 2)
    cropped_right = im0[y1_r:y2_r, x1_r:x2_r]
    # cv2.imwrite('right.png',cropped_right)

    red_detected_right = detect_red(cropped_right)

    return red_detected_right


def create_video_from_frames(folder_path, output_video_path, frame_rate=1):
    """converting images into a video"""
    images = [img for img in os.listdir(folder_path) if img.startswith("Frame_") and img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Ensure the images are sorted by name
    if not images:
        print(f"No frames_*.png files found in {folder_path}")
        return

    first_frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = first_frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    for image in images:
        img_path = os.path.join(folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video created: {output_video_path}")

def process_frames(frames,timestamp):
    """Process Frame"""
    stuck_buffer=[]
    high_danger=[]
    low_danger=[]
    left_led_working=[]
    right_led_working=[]
    stats=defaultdict()
    stats['time']=str(datetime.datetime.now())
    print(f"Processing {len(frames)} frames")
    for i, frame in enumerate(frames):
        frame=cv2.resize(frame,(1200,720))
        # cv2.imwrite(f'{timestamp}/Frame_{i}.png', frame)
        right_led_working.append(detect_right_led_working(frame))
        left_led_working.append(detect_left_led_working(frame))
        s_b,h_d,l_d=detect_boomgate_status(frame,timestamp,i)
        stuck_buffer.append(s_b)
        high_danger.append(h_d)
        low_danger.append(l_d)
    if any(all(stuck_buffer[i:i+8]) for i in range(len(stuck_buffer) - 7)):
        stats['BOOM_GATE_STUCK']=True
        stats['BOOM_GATE_STUCK_FILE_NAME']=f'output/boom_gate_stuck{timestamp}.mp4'
        create_video_from_frames(timestamp,f'output/boom_gate_stuck{timestamp}.mp4',frame_rate=1)
    if any(high_danger) and any(all(low_danger[i:i+5]) for i in range(len(low_danger) - 4)):
        stats['DANGER_ZONE_ALERT']='HIGH'
        stats['DANGER_ZONE_ALERT_FILE_NAME']=f'output/high_danger_{timestamp}.mp4'
        create_video_from_frames(timestamp,f'output/high_danger_{timestamp}.mp4',frame_rate=1)
    if not any(high_danger) and any(all(low_danger[i:i+5]) for i in range(len(low_danger) - 4)):
        stats['DANGER_ZONE_ALERT']='LOW'
        stats['DANGER_ZONE_ALERT_FILE_NAME']=f'output/low_danger_{timestamp}.mp4'
        create_video_from_frames(timestamp,f'output/low_danger_{timestamp}.mp4',frame_rate=1)
    if any(all(right_led_working[i:i+5]) for i in range(len(right_led_working) - 4)) or any(all(left_led_working[i:i+5]) for i in range(len(left_led_working) - 4)):
        stats['LED_ALERT']=True
        stats['DANGER_ZONE_ALERT_FILE_NAME']=f'output/light_fused_{timestamp}.mp4'
        create_video_from_frames(timestamp,f'output/light_fused_{timestamp}.mp4',frame_rate=1)
    else:
        shutil.rmtree(timestamp)
        stuck_buffer=[]
    print("Sending Telementry data.")
    print(stats)
    if len(stats)>1:
        client.send_telemetry(
            stats
        )
    


def get_state_boom_gate(annotated_frame,tracks):
    """Detect the boom gate a"""
    m_aoi_u_x1, m_aoi_u_y1, m_aoi_u_x2, m_aoi_u_y2 = 270, 0, 350, 260
    aoi_u_polygon = Polygon([(m_aoi_u_x1, m_aoi_u_y1), (m_aoi_u_x2, m_aoi_u_y1), (m_aoi_u_x2, m_aoi_u_y2), (m_aoi_u_x1, m_aoi_u_y2)])
    cv2.rectangle(annotated_frame, (m_aoi_u_x1, m_aoi_u_y1), (m_aoi_u_x2, m_aoi_u_y2), (255, 255, 255), 1) # BOOM UP GATE AOI
    m_aoi_d_x1, m_aoi_d_y1, m_aoi_d_x2, m_aoi_d_y2 = 270, 550, 700, 550
    aoi_d_polygon = Polygon([(m_aoi_d_x1, m_aoi_d_y1), (m_aoi_d_x2, m_aoi_d_y1), (m_aoi_d_x2, m_aoi_d_y2), (m_aoi_d_x1, m_aoi_d_y2)])
    cv2.rectangle(annotated_frame, (m_aoi_d_x1, m_aoi_d_y1), (m_aoi_d_x2, m_aoi_d_y2), (0, 0, 255), 1) # BOOM DOWN GATE AOI
    stuck_buffer=False
    boom_gate_close_buffer=False
    for track in tracks:
        for obj in track.boxes.xyxy:
            bbox_polygon = Polygon([(obj[0], obj[1]), (obj[2], obj[1]), (obj[2], obj[3]), (obj[0], obj[3])])
            if bbox_polygon.intersects(aoi_u_polygon):
                print("Boom Gate is Up.")
                text = "BOOM GATE UP"
                text_position = (m_aoi_u_x1+100, m_aoi_u_y1+150)
                cv2.putText(annotated_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 0, 0), 2)

            elif bbox_polygon.intersects(aoi_d_polygon):
                print("Boom Gate is Down.")
                text = "BOOM GATE DOWN"
                text_position = (m_aoi_d_x1+100, m_aoi_d_y1+40)
                cv2.putText(annotated_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                boom_gate_close_buffer=True
            else:
                stuck_buffer=True
                boom_gate_close_buffer=True

    return annotated_frame,stuck_buffer,boom_gate_close_buffer

def detect_boomgate_status(im0,timestamp,i):
    """Detect Boom Gate"""

    polygon_x = [296, 327, 422, 487, 512, 540, 608, 629, 629, 622, 633, 701, 728, 793, 833, 842, 840, 819,
                    882, 954, 870, 766, 694, 438, 403, 352, 336, 292, 287]
    polygon_y = [494, 492, 480, 478, 499, 487, 475, 487, 501, 510, 515, 515, 508, 496, 510, 519, 526, 543,
                    550, 575, 582, 598, 603, 661, 668, 629, 575, 524, 501]
    model = YOLO('./models/best.pt')
    model_2 = YOLO('./models/yolov8m-seg.pt')
    tracks = model.track(im0, conf=0.5, iou=0.40, persist=True,
                         device='mps',retina_masks=True, show=False)
    
    overlay = im0.copy()
    cv2.fillPoly(overlay, [np.array([polygon_x, polygon_y]).T], color=(255, 255, 0))
    cv2.addWeighted(overlay, 0.3, im0, 1 - 0.5, -2, im0)
    for track in tracks:
        for box in track.boxes:
            # Extract the coordinates and class information
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]}"  # Label with confidence
            
            # Draw the bounding box
            cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw the label
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    im0,stuck_buffer,boom_gate_close_buffer=get_state_boom_gate(im0,tracks)
    tracks_vechicles = model_2.track(im0, conf=0.5, iou=0.35, classes=[0,1,2,3,5,7,8,15,16],persist=True,
                            device='mps',retina_masks=True, show=False)
    high_danger=False
    low_danger=False

    if boom_gate_close_buffer:
        danger_zone_roi = Polygon(np.array([polygon_x, polygon_y]).T)
        # Iterate through the tracks of vehicles (or objects)
        for track in tracks_vechicles:
            for mask in track.masks:  # Assuming `track.masks` contains the segmentation masks
                # Extract contours from the binary mask
                binary_mask = mask.data.cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Convert contours to polygons
                mask_polygons = [Polygon(c[:, 0, :]) for c in contours if len(c) > 2]

                # Combine multiple polygons if necessary
                mask_polygon = unary_union(mask_polygons)

                # Check if any part of the mask polygon intersects with the danger zone
                if mask_polygon.intersects(danger_zone_roi):
                    print(mask_polygon.intersection(danger_zone_roi).area,danger_zone_roi.area)
                    text = "Unsafe Condition"
                    text_position = (900, 140)
                    cv2.putText(im0, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    high_danger = True
                    
                

    if not boom_gate_close_buffer and (detect_right_led_working(im0) or detect_left_led_working(im0)):
        danger_zone_roi = Polygon(np.array([polygon_x, polygon_y]).T)
        # Iterate through the tracks of vehicles (or objects)
        for track in tracks_vechicles:
            if track:
                for mask in track.masks:  # Assuming `track.masks` contains the segmentation masks
                    # Extract contours from the binary mask
                    binary_mask = mask.data.cpu().numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Convert contours to polygons
                    mask_polygons = [Polygon(c[:, 0, :]) for c in contours if len(c) > 2]

                    # Combine multiple polygons if necessary
                    mask_polygon = unary_union(mask_polygons)

                    # Check if any part of the mask polygon intersects with the danger zone
                    if mask_polygon.intersects(danger_zone_roi):
                        # print(mask_polygon.intersection(danger_zone_roi).area,danger_zone_roi.area)
                        low_danger = True

    
    # tracks_2[0].plot()
    cv2.imwrite('Frame.png', tracks_vechicles[0].plot(labels=False,line_width=1))

    cv2.imwrite(f'{timestamp}/Frame_{i}.png', tracks_vechicles[0].plot(labels=False,line_width=1))
    return stuck_buffer,high_danger,low_danger

def main():
    """Main function"""
    # Replace with your video source, e.g., '0' for webcam or a video file path
    video_source = "./video/demo.mov"
    cap = cv2.VideoCapture(video_source)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    target_fps =3
    queue_size=15
    frame_skip = int(fps / target_fps)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    
    # Queue to store frames for processing
    frame_queue = queue.Queue(maxsize=queue_size)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed. looping back")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frame_count += 1
        if frame_count % frame_skip !=0:
            continue

        # Store the frame in the queue
        if frame_queue.qsize() <= queue_size:
            frame_queue.put(frame)

        if frame_queue.qsize()==queue_size:
            frames=[]
            while not frame_queue.empty():
                frames.append(frame_queue.get())
            try:
                timestamp=datetime.datetime.now().strftime("%Y%d%m_%H%M%S")
                if not os.path.exists(timestamp):
                    os.makedirs(timestamp)
                frame_process_thread = threading.Thread(target=process_frames,args=(frames,timestamp))
                frame_process_thread.start()
                frame_queue.queue.clear()
                frame_process_thread.join()
            except TypeError:
                print("Some Error!")
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
