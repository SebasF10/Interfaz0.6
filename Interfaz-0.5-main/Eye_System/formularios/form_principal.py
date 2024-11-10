# form_principal.py

import customtkinter as ctk
from formularios.form_maestro_design import FormularioMaestroDesign
from formularios.form_administradores_design import FormularioAdministradorDesign
import requests
from tkinter import messagebox

# Definición de colores (tema moderno)
COLOR_MITAD_IZQUIERDA = "#1a1b26"  # Azul oscuro
COLOR_MITAD_DERECHA = "#f0f5ff"    # Azul claro

class FormCrearIniciarSesion(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configurar tema
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        # Inicializar frames antes de usarlos
        self.frame_administrador = None
        self.frame_usuario = None
        
        # Inicializar botones como atributos de clase
        self.button_administrador = None
        self.button_usuario = None
        
        self.config_window()
        self.paneles()
        self.controles_botones()
        
        # Crear frames después de la inicialización
        self.frame_administrador = ctk.CTkFrame(self.izquierda, fg_color=COLOR_MITAD_IZQUIERDA)
        self.frame_usuario = ctk.CTkFrame(self.derecha, fg_color=COLOR_MITAD_DERECHA)
        
        self.ocultar_formularios()

    def config_window(self):
        self.title('Sistema de Inicio de Sesión')
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height-40}+0+0")
        self.resizable(True, True)

    def paneles(self):
        self.cuerpo_principal = ctk.CTkFrame(self)
        self.cuerpo_principal.pack(fill='both', expand=True)

        self.izquierda = ctk.CTkFrame(self.cuerpo_principal, fg_color=COLOR_MITAD_IZQUIERDA)
        self.izquierda.pack(side='left', fill='both', expand=True)

        self.derecha = ctk.CTkFrame(self.cuerpo_principal, fg_color=COLOR_MITAD_DERECHA)
        self.derecha.pack(side='right', fill='both', expand=True)

    def controles_botones(self):
        self.button_administrador = ctk.CTkButton(
            self.izquierda,
            text="Iniciar Como Administrador",
            command=self.mostrar_formulario_administrador,
            width=200,
            height=40,
            corner_radius=10,
            hover_color="#2d4f7c"
        )
        self.button_administrador.pack(expand=True, padx=50, pady=20)

        self.button_usuario = ctk.CTkButton(
            self.derecha,
            text="Iniciar Como Usuario",
            command=self.mostrar_formulario_usuario,
            width=200,
            height=40,
            corner_radius=10,
            fg_color="#2d4f7c",
            hover_color="#1a365d"
        )
        self.button_usuario.pack(expand=True, padx=50, pady=20)

    def mostrar_formulario_administrador(self):
        if self.frame_administrador:
            self.ocultar_formularios()
            self.frame_administrador.pack(fill='both', expand=True)
            # Ocultar el botón de administrador
            self.button_administrador.pack_forget()
            
            for widget in self.frame_administrador.winfo_children():
                widget.destroy()

            ctk.CTkLabel(
                self.frame_administrador,
                text="Iniciar Como Administrador",
                font=("Helvetica", 24, "bold"),
                text_color="white"
            ).pack(pady=(20, 20))

            self.entry_admin_correo = ctk.CTkEntry(
                self.frame_administrador,
                width=300,
                height=40,
                placeholder_text="correo@ejemplo.com",
                corner_radius=10
            )
            self.entry_admin_correo.pack(pady=(5, 15))

            self.entry_admin_contraseña = ctk.CTkEntry(
                self.frame_administrador,
                width=300,
                height=40,
                placeholder_text="Contraseña",
                show="•",
                corner_radius=10
            )
            self.entry_admin_contraseña.pack(pady=(5, 20))

            self.button_iniciar_admin = ctk.CTkButton(
                self.frame_administrador,
                text="Iniciar Sesión",
                command=self.iniciar_sesion_admin,
                width=200,
                height=40,
                corner_radius=10,
                hover_color="#2d4f7c"
            )
            self.button_iniciar_admin.pack(pady=20)

            # Botón para volver
            self.button_volver_admin = ctk.CTkButton(
                self.frame_administrador,
                text="Volver",
                command=self.cerrar_formulario_administrador,
                width=200,
                height=40,
                corner_radius=10,
                hover_color="#2d4f7c"
            )
            self.button_volver_admin.pack(pady=10)

    def mostrar_formulario_usuario(self):
        if self.frame_usuario:
            self.ocultar_formularios()
            self.frame_usuario.pack(fill='both', expand=True)
            # Ocultar el botón de usuario
            self.button_usuario.pack_forget()
            
            for widget in self.frame_usuario.winfo_children():
                widget.destroy()

            ctk.CTkLabel(
                self.frame_usuario,
                text="Iniciar Como Usuario",
                font=("Helvetica", 24, "bold"),
                text_color="#1a1b26"
            ).pack(pady=(20, 20))

            self.entry_usuario_correo = ctk.CTkEntry(
                self.frame_usuario,
                width=300,
                height=40,
                placeholder_text="correo@ejemplo.com",
                corner_radius=10
            )
            self.entry_usuario_correo.pack(pady=(5, 15))

            self.entry_usuario_contraseña = ctk.CTkEntry(
                self.frame_usuario,
                width=300,
                height=40,
                placeholder_text="Contraseña",
                show="•",
                corner_radius=10
            )
            self.entry_usuario_contraseña.pack(pady=(5, 20))

            self.button_iniciar_usuario = ctk.CTkButton(
                self.frame_usuario,
                text="Iniciar Sesión",
                command=self.iniciar_sesion_user,
                width=200,
                height=40,
                corner_radius=10,
                fg_color="#2d4f7c",
                hover_color="#1a365d"
            )
            self.button_iniciar_usuario.pack(pady=20)

            # Botón para volver
            self.button_volver_usuario = ctk.CTkButton(
                self.frame_usuario,
                text="Volver",
                command=self.cerrar_formulario_usuario,
                width=200,
                height=40,
                corner_radius=10,
                fg_color="#2d4f7c",
                hover_color="#1a365d"
            )
            self.button_volver_usuario.pack(pady=10)

    def cerrar_formulario_administrador(self):
        self.frame_administrador.pack_forget()
        # Mostrar nuevamente el botón de administrador
        self.button_administrador.pack(expand=True, padx=50, pady=20)

    def cerrar_formulario_usuario(self):
        self.frame_usuario.pack_forget()
        # Mostrar nuevamente el botón de usuario
        self.button_usuario.pack(expand=True, padx=50, pady=20)

    def ocultar_formularios(self):
        if self.frame_administrador:
            self.frame_administrador.pack_forget()
            # Mostrar el botón de administrador si estaba oculto
            self.button_administrador.pack(expand=True, padx=50, pady=20)
        if self.frame_usuario:
            self.frame_usuario.pack_forget()
            # Mostrar el botón de usuario si estaba oculto
            self.button_usuario.pack(expand=True, padx=50, pady=20)

    def iniciar_sesion_admin(self):
        datos = {
            "correo": self.entry_admin_correo.get(),
            "password": self.entry_admin_contraseña.get()
        }

        try:
            response = requests.post("https://app-3520e06f-74cd-43ba-9e12-7429c4f4e834.cleverapps.io/login_admin.php", data=datos)
            if response.status_code == 200:
                respuesta_json = response.json()
                if respuesta_json.get("status") == "success":
                    messagebox.showinfo("Inicio de Sesión", "Sesión iniciada exitosamente.")
                    self.iniciar_sesion_administrador()
                else:
                    messagebox.showerror("Error", respuesta_json.get("message", "Correo electrónico o contraseña incorrectos."))
            else:
                messagebox.showerror("Error", f"Código de estado: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def iniciar_sesion_user(self):
        datos = {
            "email": self.entry_usuario_correo.get(),
            "contraseña": self.entry_usuario_contraseña.get()
        }

        try:
            response = requests.post("https://app-3520e06f-74cd-43ba-9e12-7429c4f4e834.cleverapps.io/login_users.php", data=datos)
            if response.status_code == 200:
                respuesta_json = response.json()
                if respuesta_json.get("status") == "success":
                    messagebox.showinfo("Inicio de Sesión", "Sesión iniciada exitosamente.")
                    self.iniciar_sesion_usuario()
                else:
                    messagebox.showerror("Error", respuesta_json.get("message", "Correo electrónico o contraseña incorrectos."))
            else:
                messagebox.showerror("Error", f"Código de estado: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

    def iniciar_sesion_usuario(self):
        self.destroy()
        app = FormularioMaestroDesign()
        app.mainloop()

    def iniciar_sesion_administrador(self):
        self.destroy()
        admin_app = FormularioAdministradorDesign()
        admin_app.mainloop()

if __name__ == "__main__":
    app = FormCrearIniciarSesion()
    app.mainloop()