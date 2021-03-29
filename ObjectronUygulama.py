
# %%
# Kütüphaneleri Yükleyelim
import cv2
import matplotlib.pyplot as plt
import numpy as np

#mediapipe kütüphanesi ve metotlarını ekleyelim
import mediapipe as mp
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils


# %%

# Resim okut
image = cv2.imread("sandalye.jpg")

# objectron nesnesini konfigüre et
with mp_objectron.Objectron(
    static_image_mode=True,
    max_num_objects=5,
    min_detection_confidence=0.5,
    model_name='Chair') as objectron:


    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Çizgileri çizecek metot
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.detected_objects[0].landmarks_2d, mp_objectron.BOX_CONNECTIONS)
    # 3 boyutlu aksisi çizecek metot
    # mp_drawing.draw_axis(annotated_image, results.detected_objects[0].rotation, results.detected_objects[0].translation)

    #Çizilmiş Resmi gösterene kodlar
    cv2.imshow('image window', annotated_image)
    cv2.waitKey(0)
    cv2.destoyAllWindows() 
# %%






































