# form_maestro_design.py

# form_maestro_design.py

import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
from formularios.form_reconocimiento import mostrar_ventana_reconocimiento
from formularios.form_panel_datos import mostrar_panel_datos
from formularios.form_salir import salir_aplicacion

# Definición de colores
COLOR_BARRA_SUPERIOR = "#1a1b26"    
COLOR_MENU_LATERAL = "#2d4f7c"      
COLOR_CUERPO_PRINCIPAL = "#f0f5ff"  

class FormularioMaestroDesign(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Inicializar variables de instancia
        self.menu_lateral = None
        self.cuerpo_principal = None
        self.logo_image = None
        self.perfil_image = None
        
        # Configurar tema
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        # Configurar ventana
        self.configurar_ventana()
        
        # Cargar imágenes
        self.cargar_imagenes()
        
        # Crear paneles
        self.crear_paneles()
        
        # Configurar controles
        self.configurar_controles()
        
        # Cargar pantalla principal
        self.cargar_pantalla_principal()

    def configurar_ventana(self):
        self.title('Eye System - Maestro')
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{screen_width}x{screen_height-40}+0+0")
        self.resizable(True, True)

    def cargar_imagenes(self):
        try:
            base_path = Path(__file__).parent.parent / "imagenes"
            
            logo_path = base_path / "Logo.png"
            logo_pil = Image.open(logo_path)
            self.logo_image = ctk.CTkImage(
                light_image=logo_pil,
                dark_image=logo_pil,
                size=(800, 600)
            )
            
            perfil_path = base_path / "Perfil.png"
            perfil_pil = Image.open(perfil_path)
            self.perfil_image = ctk.CTkImage(
                light_image=perfil_pil,
                dark_image=perfil_pil,
                size=(100, 100)
            )
        except Exception as e:
            print(f"Error al cargar las imágenes: {e}")
            self.logo_image = None
            self.perfil_image = None

    def crear_paneles(self):
        # Barra superior
        self.barra_superior = ctk.CTkFrame(self, fg_color=COLOR_BARRA_SUPERIOR, height=60)
        self.barra_superior.pack(side='top', fill='x')
        
        # Contenedor principal
        self.contenedor_principal = ctk.CTkFrame(self, fg_color="transparent")
        self.contenedor_principal.pack(side='top', fill='both', expand=True)
        
        self.contenedor_principal.grid_columnconfigure(1, weight=1)
        self.contenedor_principal.grid_rowconfigure(0, weight=1)
        
        # Menú lateral
        self.menu_lateral = ctk.CTkFrame(self.contenedor_principal, fg_color=COLOR_MENU_LATERAL, width=250)
        self.menu_lateral.grid(row=0, column=0, sticky="nsew")
        self.menu_lateral.grid_propagate(False)
        
        # Cuerpo principal
        self.cuerpo_principal = ctk.CTkFrame(self.contenedor_principal, fg_color=COLOR_CUERPO_PRINCIPAL)
        self.cuerpo_principal.grid(row=0, column=1, sticky="nsew")

    def configurar_controles(self):
        # Configurar barra superior
        self.label_titulo = ctk.CTkLabel(
            self.barra_superior,
            text="Eye System - Maestro",
            font=("Roboto", 20, "bold"),
            text_color="white"
        )
        self.label_titulo.pack(side='left', padx=20)
        
        # Botón de menú
        self.button_menu = ctk.CTkButton(
            self.barra_superior,
            text="≡",
            width=40,
            command=self.toggle_panel,
            fg_color=COLOR_MENU_LATERAL
        )
        self.button_menu.pack(side='left', padx=10)
        
        # Configurar menú lateral
        self.configurar_menu_lateral()

    def configurar_menu_lateral(self):
        if self.perfil_image:
            label_perfil = ctk.CTkLabel(
                self.menu_lateral,
                image=self.perfil_image,
                text=""
            )
            label_perfil.pack(pady=20)

        botones_info = [
            ("Reconocimiento", lambda: self.abrir_pantalla(mostrar_ventana_reconocimiento)),
            ("Panel De Datos", lambda: self.abrir_pantalla(mostrar_panel_datos)),
            ("Salir", lambda: salir_aplicacion(self))
        ]

        for texto, comando in botones_info:
            btn = ctk.CTkButton(
                self.menu_lateral,
                text=texto,
                command=comando,
                width=200,
                height=40,
                fg_color=COLOR_MENU_LATERAL,
                border_width=2,
                border_color="white"
            )
            btn.pack(pady=10, padx=20)

    def abrir_pantalla(self, funcion):
        for widget in self.cuerpo_principal.winfo_children():
            widget.destroy()
            
        frame_nuevo = ctk.CTkFrame(self.cuerpo_principal, fg_color=COLOR_CUERPO_PRINCIPAL)
        frame_nuevo.pack(fill='both', expand=True)
        
        boton_regresar = ctk.CTkButton(
            frame_nuevo,
            text="← Regresar",
            command=self.cargar_pantalla_principal,
            width=100,
            height=30,
            fg_color=COLOR_MENU_LATERAL
        )
        boton_regresar.pack(anchor='nw', padx=20, pady=20)
        
        funcion(frame_nuevo, self.cargar_pantalla_principal)

    def toggle_panel(self, mostrar=None):
        if mostrar is None:
            mostrar = not self.menu_lateral.winfo_ismapped()
        
        if mostrar:
            self.menu_lateral.grid(row=0, column=0, sticky="nsew")
        else:
            self.menu_lateral.grid_remove()

    def cargar_pantalla_principal(self, frame_principal=None):
        self.toggle_panel(True)
        if frame_principal is None:
            frame_principal = self.cuerpo_principal
            
        for widget in frame_principal.winfo_children():
            widget.destroy()
            
        if self.logo_image:
            frame_contenedor = ctk.CTkFrame(frame_principal, fg_color="transparent")
            frame_contenedor.place(relx=0.5, rely=0.5, anchor='center', relwidth=0.9, relheight=0.9)
            
            label_logo = ctk.CTkLabel(
                frame_contenedor,
                image=self.logo_image,
                text=""
            )
            label_logo.pack(expand=True, fill="both")

if __name__ == "__main__":
    app = FormularioMaestroDesign()
    app.mainloop()