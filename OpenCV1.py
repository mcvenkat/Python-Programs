import cv2, time

img = cv2.imread('C:\\Users\\PC\\Downloads\\VK ration card.jpg',1)

print(img.shape)

cv2.imshow('Legend', img)

resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

print(resized.shape)

#cv2.imshow('Legend', resized)

#cv2.waitKey(2000)

#cv2.destroyAllWindows()

#Face Detection

#Create a CascadeClassifier Object
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read the image as it is
#img = cv2.imread('')

#Reading the image as grayscale
gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
print(gray_img)

#Search coordinates of the image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors= 5)

print(type(faces))
print(faces)

for x, y, w, h in faces:
    img = cv2.rectangle(resized, (x,y), (x + w, y + h), (0, 255, 0), 3)

#First Frame Capture
video = cv2.VideoCapture(0)

check, frame = video.read()

print(check)
print(frame)

cv2.imshow('Capturing', frame)
cv2.waitKey(0)

time.sleep(5)

#Capture Entire Video
a = 1
while True:
    a = a + 1
    check, frame = video.read()
    print(frame)

    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Capturing', gray2)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(a)

video.release()

cv2.destroyAllWindows()

#Motion Detection

first_frame = None

while True:
    gray3 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.GaussianBlur(gray3, (21,21), 0)

    if first_frame is None:
        first_frame = gray3
        continue

    delta_frame = cv2.absdiff(first_frame, gray3)

    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations = 0)
    (_cnts_)= cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
             continue

        (x, y, w, h) = cv2.boundingRect(contour)

        cv2.rectangle(frame, x, y, x + w, y + h, 0, 255, 0, 3)
    cv2.imshow('frame', frame)
    cv2.imshow('Capturing', gray3)
    cv2.imshow('delta', delta_frame)
    cv2.imsow('thresh', thresh_delta)

#Storing Time Values
first_frame = None
status_list = [None, None]
times=[]
df = pandas.dataFrame(columns = ['Start', 'End'])
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    gray4 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.GaussianBlur(gray4, (21,21), 0)
    (_cnts_)= cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1
    # Record datetime in a list when change occurs
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    for i in range(0, len(times), 2):
        df = df.append({'Start:' times[i], 'End:' times[i+1]}, ignore_index = True)

df.to_csv('Times.csv')
video.release()

cv2.destroyAllWindows()
