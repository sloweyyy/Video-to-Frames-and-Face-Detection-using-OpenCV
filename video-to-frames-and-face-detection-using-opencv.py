# %% [markdown]
# # Convert Data into Frames and Face Detection
# 
# This notebook is split into 2 parts.  First, I am going to convert a video to still frames using OpenCV.  Next I am going to use OpenCV to put a bounding box around faces

# %% [code]


# %% [markdown]
# ## Convert Video into Frames

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:32.596498Z","iopub.execute_input":"2023-09-15T13:46:32.596864Z","iopub.status.idle":"2023-09-15T13:46:36.600453Z","shell.execute_reply.started":"2023-09-15T13:46:32.596835Z","shell.execute_reply":"2023-09-15T13:46:36.599131Z"}}
import cv2
import numpy as np

vidObj = cv2.VideoCapture('est.mp4')

count = 0

while True:

    success, image = vidObj.read()

    if success:
        cv2.imwrite(f"frame{count}.jpg", image)
    else:
        break

    count += 1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:36.602354Z","iopub.execute_input":"2023-09-15T13:46:36.602630Z","iopub.status.idle":"2023-09-15T13:46:37.744690Z","shell.execute_reply.started":"2023-09-15T13:46:36.602591Z","shell.execute_reply":"2023-09-15T13:46:37.742581Z"}}
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for i in range(0, 1300, 100):
    img = mpimg.imread(f'./frame{i}.jpg')
    imgplot = plt.imshow(img)
    plt.show()


# %% [markdown]
# ## Detect Faces
# 
# I will use this [great notebook](https://www.kaggle.com/serkanpeldek/face-detection-with-opencv) to detect faces

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.745652Z","iopub.status.idle":"2023-09-15T13:46:37.746119Z"}}
class FaceDetector():

    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.1,
               minNeighbors=5,
               minSize=(5, 5)):
        # function return rectangle coordinates of faces for given image
        rects = self.faceCascade.detectMultiScale(image,
                                                  scaleFactor=scaleFactor,
                                                  minNeighbors=minNeighbors,
                                                  minSize=minSize)
        return rects


# %% [markdown]
# Download pretrained model from [here](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.747125Z","iopub.status.idle":"2023-09-15T13:46:37.747529Z"}}
# Frontal face of haar cascade loaded
frontal_cascade_path = "/kaggle/input/casscadeclassifier/haarcascade_frontalface_default.xml"

# Detector object created
fd = FaceDetector(frontal_cascade_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.748368Z","iopub.status.idle":"2023-09-15T13:46:37.748725Z"}}
my_image = cv2.imread("frame1100.jpg")


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.749594Z","iopub.status.idle":"2023-09-15T13:46:37.749968Z"}}
def get_my_image():
    return np.copy(my_image)


def show_image(image):
    plt.figure(figsize=(18, 15))
    # Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.750828Z","iopub.status.idle":"2023-09-15T13:46:37.751184Z"}}
show_image(get_my_image())


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.752101Z","iopub.status.idle":"2023-09-15T13:46:37.752449Z"}}
def detect_face(image, scaleFactor, minNeighbors, minSize):
    # face will detected in gray image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = fd.detect(image_gray,
                      scaleFactor=scaleFactor,
                      minNeighbors=minNeighbors,
                      minSize=minSize)

    for x, y, w, h in faces:
        # detected faces shown in color image
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 255, 0), 3)

    show_image(image)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-09-15T13:46:37.753322Z","iopub.status.idle":"2023-09-15T13:46:37.753665Z"}}
detect_face(image=get_my_image(),
            scaleFactor=1.9,
            minNeighbors=3,
            minSize=(30, 30))