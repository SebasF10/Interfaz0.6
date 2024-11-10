# reconocimiento.py

import cv2
import os
from formularios.reconocimiento import asistencia


def ejecutar_reconocimiento():
    data_path = 'C:/Users/USER/Desktop/Interfaz-0.5-main/Eye_System/formularios/Data'
    image_path = os.listdir(data_path)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('prueba.xml')

    cap = cv2.VideoCapture(0)

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceClassif.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            rostro = gray[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (200, 200), interpolation=cv2.INTER_CUBIC)

            result = face_recognizer.predict(rostro)

            cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 80:
                codigo_est = image_path[result[0]]  # Se usa el índice result[0] para obtener el código del estudiante
                cv2.putText(frame, '{}'.format(codigo_est), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Registrar asistencia
                asistencia.registrar_asistencia(codigo_est)
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)

        # Salimos con la tecla 'ESC'
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()