import cv2
import os
from ultralytics import YOLO

class MultiModelDetector:
    def __init__(self, models):
        self.models = models

    def detect_and_draw(self, frame, threshold=0.5):
        results_list = [model(frame)[0].boxes.data.tolist() for model in self.models]

        for results in zip(*results_list):
            max_score = max(result[4] for result in results)

            if max_score > threshold:
                best_result = max(results, key=lambda x: x[4])
                x1, y1, x2, y2, score, class_id = best_result

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, f"{self.models[results.index(best_result)].names[int(class_id)].upper()} {score:.2f}",
                    (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        while ret:
            self.detect_and_draw(frame, threshold=0)

            cv2.imshow("Detection Model", frame)  # Display the frame

            out.write(frame)
            ret, frame = cap.read()

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

model1 = YOLO('runs/detect/train/weights/best.pt')
model2 = YOLO('runs/detect/train2/weights/best.pt')


video_path = 'videos/angled.mp4'
video_path_out = '{}_models{}_out.mp4'.format(video_path, '2_th0.3')

multi_model_detector = MultiModelDetector([model2])
multi_model_detector.run(video_path, video_path_out)
