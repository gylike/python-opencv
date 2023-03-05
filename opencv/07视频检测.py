#导入cv模块
import cv2 as cv
def face_detect_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  #转换为灰度
#加载分类器：做出人脸识别的关键  opencv已经训练好的，自带的
    face_detect = cv.CascadeClassifier('D:/Develop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray) #gray为放置的图像   1.01为缩放倍数   5检测次数（5次都要检测到）  0  默认  （100，100）最小人脸像素  （300，300）最大人脸像素大小
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)

#读取摄像头
cap = cv.VideoCapture(0)
#cap = cv.VideoCapture('***.mp4')#读取视频

#循环
while True:
    flag,frame=  cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()
#释放摄像头
cap.release()