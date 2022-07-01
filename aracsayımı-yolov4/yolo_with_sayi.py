import numpy as np
import cv2



kamera= cv2.VideoCapture('otoban.mp4')

# Yolov4'e dair config ve weights dosyalarını yükledik
net = cv2.dnn.readNet( "yolov4-tiny.cfg","yolov4-tiny.weights") 
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Coconames sınıflarını yükledik
classes = []
with open("coco.names", "r") as f:  
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]



to=0 # cikis yapan araba sayisi
gi=0 #giris yapan araba sayisi
while True:

    
    ret,img=kamera.read()

    img = cv2.resize(img, (416,416))
    height, width, channels = img.shape
    
  
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #buradaki True BGR'I RGB'e geçirmeye yarar. 0.00392 değeri ise 1/255 den gelir.0,0,0 değeri ise her RGB kanalı için ortalama alma değerini 0 tutmaktadır.

    net.setInput(blob)
    outs = net.forward(output_layers)
    
    
    
    
    class_ids = [] #sınıflarımızı tutan liste
    confidences = []#bulunan nesnelerin güven değerlerini tutan liste
    boxes = []#sınırlayıcı kutularımızı tutan liste
    for out in outs:
        for detection in out:
            scores = detection[5:]
            #detection[5:] -> ilk 4 öge center_x, center_y, genişlik ve yüksekliktir. Son öğe ise sınırlayıcı kutuların nesneyi çevrelediğindeki güven faktörüdür.
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                # Nesne tanımlama
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Kutu koordinatları
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    
#üst üste binen ve güven eşiği daha düşük olan, nesneyi tam olarak çerçevelemeyen tüm kutuları ortadan kaldırabilmek için maksimum olmayını bastırma işlemi (nmsThreshold)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

    
    sayi=0  # o anki  araba sayısı için
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            if label=='car':
                confidences.sort(reverse=True)
                conf = str(confidences[0]*100)
        #label car classına eşitse gezilen kutular değerlerine göre tersine sıralanır(en yüksekten en düşüğe) ve bunun sonucunda en yüksek güven skorlu class elde tutulur.

                color=(255,255,255)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img,conf,(x,y),font,0.5,(0,0,255),2)
                sayi=sayi+1  
                if x<200 and y+h//2>354 and y+h//2<361: # çıkış yapılan konum, bu konumlar videodan videoya değişir
                    to=to+1
                    
                if x>200 and y+h//2>340 and y+h//2<351: # giriş yapılan konum, bu konumlar videodan videoya değişir
                    gi=gi+1
                    print(x)
                #cv2.putText(img, label, (x, y + 30), font, 1, color, 3)
    cv2.line(img,(0,355),(400,355),(255,0,0),2)
    cv2.putText(img,"Cikis:"+ str(to), (0, 40), font, 1.5, (0,0,255), 2)
    cv2.putText(img,"Giris:"+ str(gi), (220, 40), font, 1.5, (0,255,255), 2)
    cv2.imshow("Image", img)
    #print(img.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
kamera.release()
cv2.destroyAllWindows()