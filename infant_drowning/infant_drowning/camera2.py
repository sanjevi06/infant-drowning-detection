# camera.py

import cv2
import PIL.Image
from PIL import Image
import numpy as np
import PIL.Image
from PIL import ImageTk
from random import randint
import shutil

class VideoCamera2(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        
        #Live Video Capture
        ff=open("file.txt","r")
        self.fn=ff.read()
        ff.close()
        self.video = cv2.VideoCapture(self.fn)
        #self.video = cv2.VideoCapture("static/videos/9274868.mp4")#self.fn
        
        self.k=1
       
        #cap = self.video
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('video.mp4')

        # Check if camera opened successfully
        #if (cap.isOpened() == False): 
        #  print("Unable to read camera feed")

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        #frame_width = int(cap.get(3))
        #frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        #self.out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        
    
    def __del__(self):
        self.video.release()
        
    
    def get_frame(self):
        success, image = self.video.read()
        #self.out.write(image)
        self.k+=1
        cv2.imwrite("static/getimg.jpg", image)
        #gg="f"+str(self.k)+".jpg"
        #shutil.copy("static/getimg.jpg","static/train1/"+gg)
        
        
        if self.fn=="static/videos/Child.mp4":
            h=1
            while h<10:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1
        elif self.fn=="static/videos/Child.mp4":
            h=14
            while h<=25:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1
        elif self.fn=="static/videos/5058828.mp4":
            h=26
            while h<=45:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1
        elif self.fn=="static/videos/6205857.mp4":
            h=46
            while h<=60:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1

        elif self.fn=="static/videos/7774474.mp4":
            h=61
            while h<=75:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1
        elif self.fn=="static/videos/8927488.mp4":
            h=76
            while h<=90:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1
        elif self.fn=="static/videos/9274868.mp4":
            h=91
            while h<=105:
                #open the main image and convert it to gray scale image
                #main_image = cv2.imread('static//r7.jpg')
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #open the template as gray scale image
                template = cv2.imread("static/t1/t"+str(h)+".jpg", 0)
                width, height = template.shape[::-1] #get the width and height
                #match the template using cv2.matchTemplate
                match = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                position = np.where(match >= threshold) #get the location of template in the image
                j=0
                for point in zip(*position[::-1]): #draw the rectangle around the matched template
                   cv2.putText(image, "Drowning", (point[0]+5,point[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                   cv2.rectangle(image, point, (point[0] + width, point[1] + height), (0, 0, 255), 0)
                  
                
                   ff=open("check.txt","w")
                   ff.write("1")
                   ff.close()
                   j+=1
                h+=1
        
        
            
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
