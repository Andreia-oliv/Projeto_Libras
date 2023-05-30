import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import tensorflow as tf

def nothing(x):
    pass

image_x, image_y = 64,64


#informamos o modelo a ser usado
classifier = load_model('models/cnn_model_LIBRAS_palavras_28_05_2023_20_25.h5')

#classes = 21
classes = 10
''''
modelo = {'0' : 'A', '1' : 'B', '2' : 'C' , '3': 'D', 
 '4': 'E', '5':'F', '6':'G', '7': 'I', '8':'L', '9':'M', 
 '10':'N', '11': 'O', '12':'P', '13':'Q', '14':'R', 
 '15':'S', '16':'T', '17':'U', '18':'V', '19':'W','20':'Y'}
 
modelo = {'0' : '0', '1' : '1', '2' : '2' , '3': '3', 
  '4': '4', '5':'5', '6':'6', '7': '7', '8':'8', '9':'9'}
'''

modelo = {'0' : 'adulto', '1' : 'america', '2' : 'casa' ,
          '3': 'gasolina', '4': 'juntos', '5':'lei',
          '6':'palavra', '7': 'pedra', '8':'pequeno', '9':'verbo'}

#le a imagem da camera e converte para o
# formato que o keras/tensorflow entendem para executar
def predictor():
       test_image = tf.keras.utils.load_img('./temp/img.png', target_size=(64, 64))
       test_image = tf.keras.utils.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)

       maior, class_index = -1, -1

       #esse for informa a probabilidade de qual
       # a letra a ser visualizada na webcam
       #dado o maior valor, ele determina qual letra é
       for x in range(classes):
           if result[0][x] > maior:
              maior = result[0][x]
              class_index = x

       return [result, modelo[str(class_index)]]


#inicia a webcam
cam = cv2.VideoCapture(0)

img_counter = 0

img_text = ['','']

#a partir daqui o while cria as caixas e
# personaliza o terminal para mostrar o resultado
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0),
                        thickness=2, lineType=8, shift=0)

    imcrop = img[102:298, 427:623]

    cv2.putText(frame, str(img_text[1]), (30, 400),
                cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0))

    cv2.imshow("test_net", frame)
    cv2.imshow("mask", imcrop)

    img_name = "./temp/img.png"
    save_img = cv2.resize(imcrop, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    img_text = predictor()
    #print(str(img_text[0]))

    print(img_text[0])
    #output = np.ones((150, 150, 3)) * 255
    output = np.ones((150, 600, 3)) * 255
    cv2.putText(output, str(img_text[1]), (15, 130),
                cv2.FONT_HERSHEY_TRIPLEX, 4, (0, 0, 0))
    cv2.imshow("PREDICT", output)


    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()
