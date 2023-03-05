#导入cv模块
import cv2 as cv
def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)  #转换为灰度
#加载分类器：做出人脸识别的关键  opencv已经训练好的，自带的
    face_detect = cv.CascadeClassifier('D:/Develop/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray) #去掉参数为使用默认配置
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)
#读取图片
img = cv.imread('face2.jpg')
#检测函数
face_detect_demo()
#等待
while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()