import cv2

# Variables
width, height = 1280, 720
folderPath = "Presentation"


# Camera Set-up
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


# Get the list of Presentation Images
pathImages = sorted(os.listdir(folderPath), key = len)
print(pathImages)

imgNumber = 0

while True:
    success, img = cap.read()
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break