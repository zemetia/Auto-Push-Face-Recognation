import os 
import numpy as np
from PIL import Image
import cv2
import pickle

lokasifile = os.path.dirname(os.path.abspath(__file__))
lokasifoto = os.path.join(lokasifile, "foto")

face_cscd = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
pengenal = cv2.face.LBPHFaceRecognizer_create()

id_skrg = 0
ids = {}
y_label = []
x_train = []


for root,dirs,files in os.walk(lokasifoto):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()

            if not label in ids:
                ids[label] = id_skrg
                id_skrg+=1
                
            pil_image = Image.open(path).convert("L")
            gbr_array = np.array(pil_image, "uint8")
            muka = face_cscd.detectMultiScale(gbr_array, scaleFactor = 1.5, minNeighbors=5)
            for(x,y,w,h) in muka:
                roi = gbr_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(id_skrg)

with open("label.pkl","wb") as f:
    pickle.dump(ids,f)
    
pengenal = cv2.face.LBPHFaceRecognizer_create()
pengenal.train(x_train,np.array(y_label))
pengenal.save("trained_model.yes")

                
