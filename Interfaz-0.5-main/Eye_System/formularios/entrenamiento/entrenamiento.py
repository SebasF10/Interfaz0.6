import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

def load_image(image_path):
    """Carga una imagen y la prepara para el entrenamiento"""
    try:
        img = cv2.imread(image_path, 0)
        # Redimensionar la imagen para consistencia y rendimiento
        img = cv2.resize(img, (150, 150))
        # Aplicar ecualización de histograma para mejorar el contraste
        img = cv2.equalizeHist(img)
        return img
    except Exception as e:
        print(f"Error al cargar imagen {image_path}: {str(e)}")
        return None

def process_person_directory(person_path, label):
    """Procesa todas las imágenes de una persona"""
    faces = []
    labels = []
    try:
        image_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Usar ThreadPoolExecutor para cargar imágenes en paralelo
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Crear lista de rutas completas
            image_paths = [os.path.join(person_path, img) for img in image_files]
            # Cargar imágenes en paralelo
            loaded_images = list(executor.map(load_image, image_paths))
            
            # Filtrar imágenes válidas y agregar etiquetas
            for img in loaded_images:
                if img is not None:
                    faces.append(img)
                    labels.append(label)
                    
        return faces, labels
    except Exception as e:
        print(f"Error al procesar directorio {person_path}: {str(e)}")
        return [], []

def ejecutar_entrenamiento():
    print("Iniciando entrenamiento...")
    start_time = time.time()
    
    dataPath = 'C:/Users/USER/Desktop/Interfaz-0.5-main/Eye_System/formularios/Data'
    peopleList = os.listdir(dataPath)
    
    if not peopleList:
        print("No se encontraron carpetas de personas para entrenar")
        return
    
    all_faces = []
    all_labels = []
    
    # Mostrar progreso
    total_people = len(peopleList)
    print(f"Procesando datos de {total_people} personas...")
    
    # Procesar cada carpeta de persona
    for i, nameDir in enumerate(peopleList):
        print(f"Procesando persona {i+1}/{total_people}: {nameDir}")
        personPath = os.path.join(dataPath, nameDir)
        
        # Procesar directorio de la persona
        faces, labels = process_person_directory(personPath, i)
        
        all_faces.extend(faces)
        all_labels.extend(labels)
        
        # Mostrar progreso
        print(f"- Procesadas {len(faces)} imágenes para {nameDir}")
    
    if not all_faces:
        print("No se encontraron imágenes válidas para entrenar")
        return
    
    print("\nComenzando entrenamiento del modelo...")
    
    # Convertir listas a arrays de numpy
    faces_array = np.array(all_faces)
    labels_array = np.array(all_labels)
    
    # Crear y entrenar el reconocedor
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=100.0
    )
    
    try:
        face_recognizer.train(faces_array, labels_array)
        
        # Guardar el modelo con compresión
        face_recognizer.write('prueba.xml')
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\nEntrenamiento completado exitosamente:")
        print(f"- Total de personas: {total_people}")
        print(f"- Total de imágenes procesadas: {len(all_faces)}")
        print(f"- Tiempo de entrenamiento: {training_time:.2f} segundos")
        print("- Modelo guardado como 'prueba.xml'")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        return