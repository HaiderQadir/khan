from pynput.mouse import Button, Controller
from cv2 import cv2
import numpy as np
import wx
mouse=Controller()

Resapp=wx.App(False)

(x_axis,y_axis)=wx.GetDisplaySize()
#(cam_x,cam_y)=(x_axis,y_axis)
(cam_x,cam_y)=(460,370)
cam = cv2.VideoCapture(0)

lower_Bound=np.array([33,80,40])
upper_Bound=np.array([102,255,255])

one_matrix_kP=np.ones((5,5))
one_matrix_kC=np.ones((20,20))

pinchFlag=0 #for opening the masks
while True:
    ret_val, img = cam.read() 
    #img = cv2.flip(img, 1)

    #ret, image=camera.read()
    img=cv2.resize(img,(460,370)) #width then height

  #convert BGR to HSV
    image_HSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(image_HSV,lower_Bound,upper_Bound)

    #morphology
    mask_Open=cv2.morphologyEx(mask,cv2.MORPH_OPEN,one_matrix_kP)
    mask_Close=cv2.morphologyEx(mask_Open,cv2.MORPH_CLOSE,one_matrix_kC)
    maskFinal=mask_Close

    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,conts,-1,(255,0,0),3)

    if(len(conts)==1):
        x,y,w,h=cv2.boundingRect(conts[0])
        
        if(pinchFlag==0):
            pinchFlag=1
            mouse.press(Button.right)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cx=x+w//2
        
        cy=y+h//2
        cv2.circle(img,(cx,cy),(w+h)//4,(0,0,255),2)
        mouseLoc=(x_axis-(cx*x_axis//cam_x), cy*y_axis//cam_y)
        mouse.position=mouseLoc 
        while mouse.position!=mouseLoc:
            pass
    
    
    #cv2.imshow('mask', mask)
    cv2.imshow('|DCUGD| Capture', img)
    #cv2.imshow('mask 2', mask_Open)
  #if cv2.waitKey(2) == 27: 
   #   break  # esc to quit
      
    if cv2.waitKey(2) == 98:
      break
       
cv2.destroyAllWindows()

   
   
 