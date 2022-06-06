import cv2
import numpy as np
import pickle
from datetime import date as dt
import os

cam = cv2.VideoCapture(1)
face_cscd = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
pengenal = cv2.face.LBPHFaceRecognizer_create()
pengenal.read('trained_model.yes')
tanggal = str(dt.today())
angka = 0
za = 30

if os.path.exists("tidak_dikenal/"+tanggal)==False:
    lokasifile = os.path.dirname(os.path.abspath(__file__))
    lokasifoto = os.path.join(lokasifile, "tidak_dikenal")
    lokasifoto = os.path.join(lokasifoto, tanggal)
    os.mkdir(lokasifoto)

labels={}
with open("label.pkl","rb") as f:
    m_labels = pickle.load(f)
    labels = {v:k for k,v in m_labels.items()}

while True:
    success, img = cam.read()
    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    muka = face_cscd.detectMultiScale(abu, scaleFactor = 1.5, minNeighbors=5)
    for (x,y,w,h) in muka:
        #print(x,y,w,h)
        roi_abu = abu[y:y+h,x:x+w]
        roi_warna = img[y:y+h,x:x+w]
        if (x-int(za/2)<= 0 or y-int(za/2)<= 0):
            tangkapan = abu[y:y+h,x:x+w]
        else:
            tangkapan = img[y-int(za/2):y+h+za,x-int(za/2):x+w+za]
        id_,conf = pengenal.predict(roi_abu)
        if conf>=65 and conf<=100:
            cv2.putText(img,labels[id_-1]+": "+str(conf)+"%",(x,y+h+23),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))
            fotolabel = labels[id_-1]+".png"
            resized_image = cv2.resize(roi_warna, (300, 300))
            cv2.imwrite("tangkapan/"+fotolabel, resized_image)
            print("Telah dilihat:",labels[id_-1],"("+str(conf)+")")
        else:
            fotolabel = str(angka) +".png"
            resized_image = cv2.resize(tangkapan, (300, 300))
            cv2.imwrite("tidak_dikenal/"+tanggal+"/"+fotolabel, resized_image)
            angka += 1
            print("Telah dilihat: Tidak Dikenal",angka)

        cv2.putText(img,"("+str(x)+","+str(y)+")",(x,y-10),cv2.FONT_HERSHEY_COMPLEX, 0.5,(0,255,0))
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("webcam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

