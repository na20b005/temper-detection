import cv2
import numpy as np

cap = cv2.VideoCapture(0)
_,f = cap.read()

avg1 = np.float32(f)
avg2 = np.float32(f)
c = 10
while(1):
    c += 1
    ksize = int(c/10)
    _,f = cap.read()
    f=cv2.blur(f,ksize= (ksize,ksize))
    cv2.accumulateWeighted(f,avg1,0.1)
    cv2.accumulateWeighted(f,avg2,0.01)

    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    k = cv2.waitKey(20)

    # print(np.mean(np.abs(res1-res2)) / np.mean(res1))
    print(np.var(f))
    print()

    if k == 27:
        break

cv2.destroyAllWindows()
c.release()