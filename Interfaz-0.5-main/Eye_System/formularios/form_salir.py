# form_salir.py

import tkinter as tk

def salir_aplicacion(root):
    # Importar dentro de la función para evitar la dependencia circular
    from formularios.form_principal import FormCrearIniciarSesion

    # Destruir la ventana actual
    root.destroy()

    # Crear una nueva instancia de la interfaz principal
    ventana_principal = FormCrearIniciarSesion()
    ventana_principal.mainloop()

