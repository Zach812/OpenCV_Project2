import cv2
import os
import mediapipe as mp
import argparse

output_dir = os.path.join('.', 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_image(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data

            # bounding box of the face
            bbox = location_data.relative_bounding_box
            
            x1, y1, w, h = bbox.xmin , bbox.ymin , bbox.width , bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            # x1 = int(x1)
            # y1 = int(y1)
            # w = int(w)
            # h = int(h)

            # Showing rectangle around face
            # img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return img

args = argparse.ArgumentParser()

args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default=None)

args =args.parse_args()

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    # Read image
    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
        # Processing Images
        img = process_image(img, face_detection)
        # save image
        cv2.imwrite(os.path.join(output_dir, 'man_smile_output.jpg'), img)
    
    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), 
                                        cv2.VideoWriter_fourcc(*'mp4v'), 
                                        25, # might want to specify same as video input
                                        (frame.shape[1], frame.shape[0])
                                        )

        while ret:
            frame = process_image(frame, face_detection)
            
            output_video.write(frame)

            ret, frame = cap.read()


        cap.release()
        output_video.release()

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        while ret:
            frame = process_image(frame, face_detection)
            cv2.imshow('frame', frame)
            # This is 25ms because the 40 fps
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()


        cap.release()