import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import os
import threading
from datetime import datetime
import pymysql

class RecognitionApp:
    def __init__(self):
        self.is_running = False
        self.face_recognizer = None
        self.cap = None
        self.registros_hoy = set()  # Conjunto para almacenar los códigos ya registrados hoy
    
    def iniciar_reconocimiento(self, label_camera):
        if self.is_running:
            return
        
        self.is_running = True
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('prueba.xml')
        self.cap = cv2.VideoCapture(0)
        
        # Iniciar el reconocimiento en un hilo separado
        threading.Thread(target=self.proceso_reconocimiento, args=(label_camera,), daemon=True).start()
    
    def detener_reconocimiento(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
    
    def registrar_asistencia(self, codigo_est):
        try:
            # Obtener la fecha actual
            fecha_actual = datetime.now().strftime('%Y-%m-%d')
            registro_key = f"{codigo_est}_{fecha_actual}"
            
            # Verificar si ya se registró hoy
            if registro_key not in self.registros_hoy:
                # Conexión a la base de datos remota
                conexion = pymysql.connect(
                    host='b4qhbwwqys2nhher1vul-mysql.services.clever-cloud.com',
                    port=3306,
                    db='b4qhbwwqys2nhher1vul',
                    user='upvge9afjesbmmgv',
                    password='BS2bxJNACO1XYEmWBqA0'
                )

                cursor = conexion.cursor()

                # Verificar si el estudiante existe
                consulta_estudiante = "SELECT * FROM estudiante WHERE codigo_est = %s"
                cursor.execute(consulta_estudiante, (codigo_est,))
                estudiante = cursor.fetchone()

                if estudiante:
                    # Obtener hora actual
                    hora_actual = datetime.now().strftime("%H:%M:%S")

                    # Insertar asistencia en la tabla
                    insertar_asistencia = "INSERT INTO ingreso (codigo_est, fecha, hora) VALUES (%s, %s, %s)"
                    cursor.execute(insertar_asistencia, (codigo_est, fecha_actual, hora_actual))
                    conexion.commit()

                    print(f"Asistencia registrada para el estudiante: {estudiante[2]} {estudiante[1]} (Grupo: {estudiante[3]}, Jornada: {estudiante[4]})")
                    
                    # Agregar a los registros de hoy
                    self.registros_hoy.add(registro_key)
                    return True
                else:
                    print("Estudiante no encontrado.")
                    return False

                cursor.close()
                conexion.close()
            return False
        except Exception as e:
            print(f"Error al registrar asistencia: {e}")
            return False
    
    def proceso_reconocimiento(self, label_camera):
        data_path = 'C:/Users/USER/Desktop/Interfaz-0.5-main/Eye_System/formularios/Data'
        image_path = os.listdir(data_path)
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceClassif.detectMultiScale(gray, 1.2, 4, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                rostro = gray[y:y+h, x:x+w]
                rostro = cv2.resize(rostro, (200, 200), interpolation=cv2.INTER_CUBIC)
                result = self.face_recognizer.predict(rostro)
                
                if result[1] < 80:
                    codigo_est = image_path[result[0]]
                    # Intentar registrar asistencia
                    if self.registrar_asistencia(codigo_est):
                        color = (0, 255, 0)  # Verde si se registró
                        texto = f'{codigo_est} - Registrado'
                    else:
                        color = (255, 255, 0)  # Amarillo si ya estaba registrado
                        texto = f'{codigo_est} - Ya registrado'
                    
                    cv2.putText(frame, texto, (x, y-25), 2, 1.1, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                else:
                    cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Convertir el frame para mostrarlo en la interfaz
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            label_camera.configure(image=photo)
            label_camera.image = photo

def mostrar_ventana_reconocimiento(frame_principal, callback_return):
    # Limpiar el frame principal
    for widget in frame_principal.winfo_children():
        widget.destroy()
    
    # Crear instancia de la aplicación de reconocimiento
    app = RecognitionApp()
    
    # Crear un frame contenedor
    frame_reconocimiento = ctk.CTkFrame(frame_principal, fg_color="transparent")
    frame_reconocimiento.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Título
    label_titulo = ctk.CTkLabel(
        frame_reconocimiento,
        text="Reconocimiento Facial",
        font=("Roboto", 24, "bold")
    )
    label_titulo.pack(pady=20)
    
    # Frame para la cámara
    frame_camera = ctk.CTkFrame(frame_reconocimiento, width=640, height=480)
    frame_camera.pack(pady=20)
    
    # Label para la cámara (ahora mostrará el video)
    label_camera = ctk.CTkLabel(frame_camera, text="")
    label_camera.place(relx=0.5, rely=0.5, anchor="center")
    
    # Botones de control
    frame_botones = ctk.CTkFrame(frame_reconocimiento, fg_color="transparent")
    frame_botones.pack(pady=20)
    
    btn_iniciar = ctk.CTkButton(
        frame_botones,
        text="Iniciar Reconocimiento",
        font=("Roboto", 14),
        command=lambda: app.iniciar_reconocimiento(label_camera),
        width=200
    )
    btn_iniciar.pack(side="left", padx=10)
    
    btn_detener = ctk.CTkButton(
        frame_botones,
        text="Detener",
        font=("Roboto", 14),
        command=app.detener_reconocimiento,
        width=200
    )
    btn_detener.pack(side="left", padx=10)
    
    # Botón para volver
    btn_volver = ctk.CTkButton(
        frame_reconocimiento,
        text="Volver al Menú Principal",
        font=("Roboto", 14),
        command=lambda: [app.detener_reconocimiento(), callback_return(frame_principal)],
        width=200
    )
    btn_volver.pack(pady=20)

if __name__ == "__main__":
    app = ctk.CTk()
    app.title("Sistema de Reconocimiento Facial")
    app.geometry("800x600")
    
    def dummy_callback(frame):
        app.quit()
    
    mostrar_ventana_reconocimiento(app, dummy_callback)
    app.mainloop()