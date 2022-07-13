"""
$ pip install --upgrade opencv-contrib-python python-vlc
(python-vlc requires vlc  *64bit* installed.)

deep learning face detection:
https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

python-vlc issue: keep window open while playing
https://stackoverflow.com/questions/43272532/python-vlc-wont-start-the-player

threading:
https://realpython.com/intro-to-python-threading/
https://www.askpython.com/python/oops/threading-with-classes

T-Rex Video: https://www.youtube.com/watch?v=Ml5ABdE1HtM
"""

import numpy as np
import time
import vlc
import cv2
from threading import Thread


class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream", resolution=(800, 600), fps=30):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src + cv2.CAP_DSHOW)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stream.set(cv2.CAP_PROP_FPS, fps)

        (self.grabbed, self.frame) = self.stream.read()

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class ThreadedVLC(Thread):
    def init_player(self, src, fullscreen=True):
        self.player = vlc.MediaPlayer()
        self.player.set_media(vlc.Media(src))  # or just using vlc.MediaPlayer(src)
        if fullscreen is True:
            self.player.toggle_fullscreen()

    def run(self):
        self.player.play()
        time.sleep(1)  # wait while VLC is starting
        while self.player.is_playing():
            time.sleep(0.1)  # wait while video is playing
        self.player.stop()


show_cam = True
cam_framerate = 15
cam_resolution = (640, 480)
confidence_threshold = 0.4

play_video = True
fullscreen = True
video_path = "video/Dino.mp4"

basepath = "res10_ssd_face-detector/"
model = "res10_300x300_ssd_iter_140000.caffemodel"
prototxt = "res10_300x300_ssd.prototxt"

# init face detection model
net = cv2.dnn.readNetFromCaffe(basepath + prototxt, basepath + model)

# init the video stream
print("starting webcam...")
print("press [Q] to exit.")
vs = WebcamVideoStream(src=0, resolution=cam_resolution, fps=cam_framerate).start()

# init variables for detecting upcoming faces
before_found = current_found = False

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < confidence_threshold:
            continue

        # else if face found:
        current_found = True

        if show_cam is True:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # PLAY VIDEO
    if before_found is False and current_found is True:
        print("Face detected!")
        if play_video is True:
            threaded_player = ThreadedVLC()
            threaded_player.init_player(video_path, fullscreen=fullscreen)
            threaded_player.start()

    # reset *found* variables for next frame
    before_found = current_found
    current_found = False

    # limit cam processing frame rate
    time.sleep(1 / cam_framerate)

    if show_cam is True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# cleanup
cv2.destroyAllWindows()
vs.stop()
