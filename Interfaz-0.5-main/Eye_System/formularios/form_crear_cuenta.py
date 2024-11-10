# form_crear_cuanta.py

import customtkinter as ctk
from tkinter import messagebox
import requests

# Definición de colores modernos
COLOR_PRINCIPAL = "#1a1b26"     # Azul oscuro
COLOR_SECUNDARIO = "#f0f5ff"    # Azul claro 
COLOR_ACENTO = "#2d4f7c"        # Azul medio
COLOR_HOVER = "#1a365d"         # Azul oscuro para hover
COLOR_CUERPO_PRINCIPAL = COLOR_SECUNDARIO

def crear_cuenta_admin():
    # Datos para la creación de la cuenta de administrador
    datos = {
        "correo": correo_entry.get(),
        "password": contraseña_entry.get(),
    }

    try:
        response = requests.post("https://app-3520e06f-74cd-43ba-9e12-7429c4f4e834.cleverapps.io/registro_admins.php", data=datos, verify=False)
        
        if response.status_code == 200:
            respuesta_json = response.json()
            if respuesta_json.get("status") == "success":
                messagebox.showinfo("Cuenta Creada", "La cuenta de administrador ha sido creada exitosamente.")
                limpiar_campos()
            else:
                messagebox.showerror("Error", respuesta_json.get("message", "Error al crear la cuenta de administrador."))
        else:
            messagebox.showerror("Error", f"Error de comunicación: {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")
        messagebox.showerror("Error", f"Error al crear la cuenta: {str(e)}")

def crear_cuenta_user():
    # Datos para la creación de la cuenta de usuario
    datos = {
        "email": correo_entry.get(),
        "contraseña": contraseña_entry.get(),
    }

    try:
        response = requests.post("https://app-3520e06f-74cd-43ba-9e12-7429c4f4e834.cleverapps.io/registro_users.php", data=datos, verify=False)
        
        if response.status_code == 200:
            respuesta_json = response.json()
            if respuesta_json.get("status") == "success":
                messagebox.showinfo("Cuenta Creada", "La cuenta de usuario ha sido creada exitosamente.")
                limpiar_campos()
            else:
                messagebox.showerror("Error", respuesta_json.get("message", "Error al crear la cuenta de usuario."))
        else:
            messagebox.showerror("Error", f"Error de comunicación: {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")
        messagebox.showerror("Error", f"Error al crear la cuenta: {str(e)}")

def limpiar_campos():
    correo_entry.delete(0, 'end')
    contraseña_entry.delete(0, 'end')

def abrir_ventana_crear_cuenta(parent_frame, regresar_callback):
    # Limpiar frame principal
    for widget in parent_frame.winfo_children():
        widget.destroy()

    # Estilos y fuentes
    fuente_titulo = ("Roboto", 24, "bold")
    fuente_subtitulo = ("Roboto", 18, "bold")
    fuente_normal = ("Roboto", 14)

    # Frame principal
    main_frame = ctk.CTkFrame(parent_frame, fg_color=COLOR_CUERPO_PRINCIPAL)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # Título principal
    titulo = ctk.CTkLabel(main_frame, text="Crear Nueva Cuenta", font=fuente_titulo, text_color=COLOR_PRINCIPAL)
    titulo.pack(pady=20)

    # Frame para los campos de entrada
    campos_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_CUERPO_PRINCIPAL)
    campos_frame.pack(pady=10)

    # Campo de correo
    correo_label = ctk.CTkLabel(campos_frame, text="Correo Electrónico:", font=fuente_normal, text_color=COLOR_PRINCIPAL)
    correo_label.pack(pady=5)

    global correo_entry
    correo_entry = ctk.CTkEntry(campos_frame, font=fuente_normal, height=45, width=300)
    correo_entry.pack(pady=5)

    # Campo de contraseña
    contrasena_label = ctk.CTkLabel(campos_frame, text="Contraseña:", font=fuente_normal, text_color=COLOR_PRINCIPAL)
    contrasena_label.pack(pady=5)

    global contraseña_entry
    contraseña_entry = ctk.CTkEntry(campos_frame, show="*", font=fuente_normal, height=45, width=300)
    contraseña_entry.pack(pady=5)

    # Frame para los botones de tipo de cuenta
    botones_tipo_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_CUERPO_PRINCIPAL)
    botones_tipo_frame.pack(pady=20)

    # Subtítulo para selección de tipo
    tipo_label = ctk.CTkLabel(botones_tipo_frame, text="Seleccione el tipo de cuenta:", font=fuente_subtitulo, text_color=COLOR_PRINCIPAL)
    tipo_label.pack(pady=10)

    # Botón para crear cuenta de administrador
    boton_crear_admin = ctk.CTkButton(
        botones_tipo_frame, 
        text="Crear Cuenta de Administrador",
        command=crear_cuenta_admin,
        width=250,
        height=40,
        corner_radius=10,
        fg_color=COLOR_ACENTO,
        hover_color=COLOR_HOVER
    )
    boton_crear_admin.pack(pady=5)

    # Botón para crear cuenta de usuario
    boton_crear_user = ctk.CTkButton(
        botones_tipo_frame, 
        text="Crear Cuenta de Usuario",
        command=crear_cuenta_user,
        width=250,
        height=40,
        corner_radius=10,
        fg_color=COLOR_ACENTO,
        hover_color=COLOR_HOVER
    )
    boton_crear_user.pack(pady=5)

    # Botón para regresar
    boton_regresar = ctk.CTkButton(
        main_frame,
        text="Regresar",
        command=regresar_callback,
        width=200,
        height=40,
        corner_radius=10,
        fg_color=COLOR_ACENTO,
        hover_color=COLOR_HOVER
    )
    boton_regresar.pack(pady=20)