import cv2

vidcap = cv2.VideoCapture('vidd.mp4')
#vidcap.set(1,3)
print(vidcap)
success, image = vidcap.read()
print(success)
count = 0
success = True
while success:
    success, image = vidcap.read()
    cv2.imwrite('./ev_Best_Route_Planning/deeptesla/epochs/image/frame%d.jpg' % count, image)
    count += 1
