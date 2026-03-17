

from tkinter import *
import threading
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2

# ---------------- MAIN WINDOW ----------------
main = Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("700x600")
main.config(bg="#f0f2f5")

card = Frame(main, bg="white", bd=2, relief="ridge",
             width=500, height=450)
card.place(relx=0.5, rely=0.5, anchor="center")
card.pack_propagate(False)

running = False

# ---------------- EAR FUNCTION ----------------
def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------- MOR FUNCTION ----------------
def MOR(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return ((A + B) / 2.0) / C

# ---------------- CAMERA LOOP ----------------
def camera_loop():
    global running

    webcamera = cv2.VideoCapture(0)
    predictor_path = "SVMclassifier.dat"

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 10
    MOU_AR_THRESH = 0.75

    COUNTER = 0
    yawnStatus = False
    yawns = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    while running:
        ret, frame = webcamera.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        prev_yawn_status = yawnStatus

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mouEAR = MOR(mouth)

            # Draw contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,255), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,255), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0,255,0), 1)

            # Eye detection
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Eyes Closed", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)
            else:
                COUNTER = 0
                cv2.putText(frame, "Eyes Open", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            # Yawn detection
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning Detected!", (10,90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                yawnStatus = True
                cv2.putText(frame, "Yawn Count: " + str(yawns+1),
                            (10,120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
            else:
                yawnStatus = False

            if prev_yawn_status and not yawnStatus:
                yawns += 1

            # Show EAR & MAR
            cv2.putText(frame, "EAR: {:.2f}".format(ear),
                        (480,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(frame, "MAR: {:.2f}".format(mouEAR),
                        (480,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Driver Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcamera.release()
    cv2.destroyAllWindows()
    status_label.config(text="Status: Stopped", fg="red")
    running = False

# ---------------- BUTTON FUNCTIONS ----------------
def startMonitoring():
    global running
    if not running:
        running = True
        status_label.config(text="Status: Monitoring...", fg="blue")
        threading.Thread(target=camera_loop).start()

def stopMonitoring():
    global running
    running = False

# ---------------- UI ----------------
Label(card, text="Driver Drowsiness Detection",
      font=("Arial",20,"bold"),
      bg="white", fg="#2c3e50").pack(pady=20)

Button(card, text="Start Monitoring",
       command=startMonitoring,
       bg="#2e86de", fg="white",
       font=("Arial",12,"bold"),
       width=20).pack(pady=10)

Button(card, text="Stop Monitoring",
       command=stopMonitoring,
       bg="#e74c3c", fg="white",
       font=("Arial",12,"bold"),
       width=20).pack(pady=10)

status_label = Label(card, text="Status: Idle",
                     font=("Arial",12,"bold"),
                     fg="green", bg="white")
status_label.pack(pady=20)

main.mainloop()
