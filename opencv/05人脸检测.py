#导入cv模块
import cv2 as cv
def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  #转换为灰度
#加载分类器：做出人脸识别的关键  opencv自带
    face_detect = cv.CascadeClassifier('D:/Develop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray,1.01,5,0,(100,100),(1000,1000)) #gray为放置的图像   1.01为缩放倍数   5检测次数（5次都要检测到）  0  默认  （100，100）最小人脸像素  （300，300）最大人脸像素大小
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)
#读取图片
img = cv.imread('face1.jpg')
#检测函数
face_detect_demo()
#等待
while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()