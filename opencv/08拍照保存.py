#导入cv模块
import cv2 as cv
#读取摄像头
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture(0)
falg = 1
num = 1
#循环
while(cap.isOpened()):#检测是否在开启状态
    ret_flag,Vshow = cap.read()#得到每帧图像
    cv.imshow("Capture_Test",Vshow)#显示图像
    k = cv.waitKey(1) & 0xFF#按键判断
    if k ==ord('s'):#保存
        cv.imwrite("D:/File/python/mycodetest/opencv/data/jm/"+str(num)+".name"+".jpg",Vshow)
        print("success to save"+str(num)+".jpg")
        print("---------------------")
        num += 1
    elif k == ord(' '):#空格退出
        break

#释放摄像头
cap.release()
#释放内存
cv.destroyAllWindows()