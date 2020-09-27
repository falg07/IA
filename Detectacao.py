#Importa as bibliotecas necessárias
import numpy as np
import argparse
import cv2 

# construção do argument parse
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt",
                                  help='Path to text network file: '
                                       'MobileNetSSD_deploy.prototxt for Caffe model or '
                                       )
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel",
                                 help='Path to weights: '
                                      'MobileNetSSD_deploy.caffemodel for Caffe model or '
                                      )
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
args = parser.parse_args()

# Rótulos da rede.
classNames = { 0: 'background',
    1: 'Avião', 2: 'Bicicleta', 3: 'Passaro', 4: 'Barco',
    5: 'Garrafa', 6: 'Onibus', 7: 'Carro', 8: 'Gato', 9: 'Cadeira',
    10: 'Vaca', 11: 'Mesa de Jantar', 12: 'Cachorro', 13: 'Cavalo',
    14: 'Moto', 15: 'Pessoa', 16: 'Vaso de Planta',
    17: 'Ovelha', 18: 'Sofá', 19: 'Trem', 20: 'tvmonitor' }

# Abra o arquivo de vídeo ou o dispositivo de captura.
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

#Carregar Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 180)  # espelha a imagem
    frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

    # MobileNet requer dimensões fixas para imagem (ns) de entrada
    # portanto, temos que garantir que ele seja redimensionado para 300x300 pixels.
    # definir um fator de escala para a imagem porque os objetos em rede têm tamanhos diferentes.
    # Realizamos uma subtração média (127,5, 127,5, 127,5) para normalizar a entrada;
    # depois de executar este comando, nosso "blob" agora tem a forma:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    #Defina para colocar em rede o blob de entrada
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()

    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]

    # Para obter a classe e localização do objeto detectado,
    # Existe um índice de correção para classe, localização e confiança
    # valor no array @detections.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            # Draw location of object  
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))

            # Draw label and confidence of prediction in frame resized
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label) #print class and confidence

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
