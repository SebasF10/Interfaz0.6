# form_tomar_datos.py

import pymysql
import customtkinter as ctk
from tkinter import messagebox
import threading
import time
from formularios.base_de_datos import base_de_datos
from formularios.entrenamiento import entrenamiento
from formularios.reconocimiento import reconocimiento

# Definición de colores modernos
COLOR_PRINCIPAL = "#1a1b26"     # Azul oscuro
COLOR_SECUNDARIO = "#f0f5ff"    # Azul claro
COLOR_ACENTO = "#2d4f7c"        # Azul medio
COLOR_HOVER = "#1a365d"         # Azul oscuro para hover

class VentanaCarga(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana
        self.geometry("400x250")
        self.title("Cargando...")
        
        # Frame principal
        self.frame = ctk.CTkFrame(self)
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Etiqueta de carga
        self.label = ctk.CTkLabel(
            self.frame,
            text="Entrenando, por favor espere...",
            font=("Roboto", 16, "bold")
        )
        self.label.pack(pady=20)
        
        # Barra de progreso
        self.progressbar = ctk.CTkProgressBar(self.frame)
        self.progressbar.pack(pady=20)
        self.progressbar.set(0)
        
    def actualizar_progreso(self, valor):
        self.progressbar.set(valor)
        
def abrir_ventana_toma_de_datos(frame_principal, regresar_callback):
    # Limpiar frame principal
    for widget in frame_principal.winfo_children():
        widget.destroy()
        
    def iniciar_captura():
        nombre_carpeta = codigo_est_entry.get()
        if not nombre_carpeta:
            messagebox.showerror("Error", "Debe ingresar un código de usuario.")
            return
        base_de_datos.ejecutar(nombre_carpeta)
        print("Captura de fotos iniciada...")

    def iniciar_entrenamiento():
        ventana_carga = VentanaCarga()
        
        def entrenamiento_thread():
            try:
                # Simular progreso
                for i in range(100):
                    ventana_carga.after(50 * i, lambda v=i/100: ventana_carga.actualizar_progreso(v))
                
                entrenamiento.ejecutar_entrenamiento()
                ventana_carga.after(5000, ventana_carga.destroy)
                messagebox.showinfo("Entrenamiento Completo", "El entrenamiento se ha completado exitosamente.")
            except Exception as e:
                print(f"Error en entrenamiento: {str(e)}")
                ventana_carga.destroy()
                
        threading.Thread(target=entrenamiento_thread).start()

    def guardar_datos():
        try:
            conn = pymysql.connect(
                host='b4qhbwwqys2nhher1vul-mysql.services.clever-cloud.com',
                port=3306,
                db='b4qhbwwqys2nhher1vul',
                user='upvge9afjesbmmgv',
                password='BS2bxJNACO1XYEmWBqA0'
            )
            
            cur = conn.cursor()
            sql = "INSERT INTO estudiante (codigo_est, apellidos, nombres, grupo, jornada) VALUES (%s, %s, %s, %s, %s)"
            
            cur.execute(sql, (
                codigo_est_entry.get(),
                apellidos_entry.get(),
                nombres_entry.get(),
                grupo_entry.get(),
                jornada_var.get()
            ))
            
            conn.commit()
            messagebox.showinfo("Éxito", "Datos guardados correctamente")
            
        except Exception as e:
            print(f"Error al guardar datos: {str(e)}")
            messagebox.showerror("Error", "No se pudieron guardar los datos")
        finally:
            if 'conn' in locals() and conn.open:
                cur.close()
                conn.close()

    # Frame principal
    main_frame = ctk.CTkFrame(frame_principal, fg_color="transparent")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)
    
    # Título
    titulo = ctk.CTkLabel(
        main_frame,
        text="Tomar Datos del Estudiante",
        font=("Roboto", 24, "bold"),
        text_color=COLOR_PRINCIPAL
    )
    titulo.pack(pady=20)
    
    # Frame para el formulario
    form_frame = ctk.CTkFrame(main_frame)
    form_frame.pack(fill="x", padx=20, pady=10)
    
    # Grid para organizar los campos
    form_frame.grid_columnconfigure((1, 3, 5), weight=1)
    
    # Campos del formulario
    campos = [
        ("Código del Estudiante:", 0, 0),
        ("Apellidos:", 0, 2),
        ("Nombres:", 0, 4),
        ("Grupo:", 1, 0)
    ]
    
    entries = {}
    for (label_text, row, col) in campos:
        ctk.CTkLabel(
            form_frame,
            text=label_text,
            font=("Roboto", 14)
        ).grid(row=row, column=col, padx=5, pady=10, sticky="e")
        
        entry = ctk.CTkEntry(
            form_frame,
            height=35,
            font=("Roboto", 14)
        )
        entry.grid(row=row, column=col+1, padx=5, pady=10, sticky="ew")
        entries[label_text] = entry
    
    codigo_est_entry = entries["Código del Estudiante:"]
    apellidos_entry = entries["Apellidos:"]
    nombres_entry = entries["Nombres:"]
    grupo_entry = entries["Grupo:"]
    
    # Jornada
    ctk.CTkLabel(
        form_frame,
        text="Jornada:",
        font=("Roboto", 14)
    ).grid(row=1, column=2, padx=5, pady=10, sticky="e")
    
    jornada_var = ctk.StringVar(value="Mañana")
    jornada_menu = ctk.CTkOptionMenu(
        form_frame,
        values=["Mañana", "Tarde"],
        variable=jornada_var,
        width=150,
        height=35,
        font=("Roboto", 14)
    )
    jornada_menu.grid(row=1, column=3, padx=5, pady=10, sticky="ew")
    
    # Frame para botones
    button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
    button_frame.pack(fill="x", padx=20, pady=20)
    
    # Botones
    botones = [
        ("Guardar Datos", guardar_datos),
        ("Iniciar Captura", iniciar_captura),
        ("Entrenar", iniciar_entrenamiento),
        ("Regresar", regresar_callback)
    ]
    
    for i, (texto, comando) in enumerate(botones):
        ctk.CTkButton(
            button_frame,
            text=texto,
            command=comando,
            width=200,
            height=40,
            font=("Roboto", 14),
            fg_color=COLOR_ACENTO,
            hover_color=COLOR_HOVER,
            corner_radius=10
        ).pack(pady=10)