import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread("Erp_Photo.jpg")
event = threading.Event()

def check_frame(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False
    finally:
        event.set()

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_frame, args=(frame.copy(),)).start()

        counter += 1

        event.wait()  # Wait for the check_frame thread to finish
        event.clear()  # Reset the event for the next iteration

        if face_match:
            cv2.putText(frame, "Match!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match!!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()