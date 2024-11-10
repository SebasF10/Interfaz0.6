# form_panel_datos.py

import customtkinter as ctk
from PIL import Image, ImageTk
import pymysql
from datetime import datetime
import csv
from typing import List, Tuple, Optional
import logging
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DatabaseConnection:
    def __init__(self):
        self.connection_params = {
            'host': 'b4qhbwwqys2nhher1vul-mysql.services.clever-cloud.com',
            'port': 3306,
            'db': 'b4qhbwwqys2nhher1vul',
            'user': 'upvge9afjesbmmgv',
            'password': 'BS2bxJNACO1XYEmWBqA0'
        }
        
    def __enter__(self):
        self.connection = self.get_connection()
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
    
    def get_connection(self) -> Optional[pymysql.Connection]:
        try:
            return pymysql.connect(**self.connection_params)
        except Exception as e:
            logging.error(f"Error de conexión a la base de datos: {e}")
            return None

class DataManager:
    @staticmethod
    def fetch_asistencia(limit: int = 100) -> List[Tuple]:
        with DatabaseConnection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT a.id, CONCAT(e.nombres, ' ', e.apellidos) as nombre, 
                               a.fecha, a.estado 
                        FROM asistencia a
                        JOIN estudiante e ON a.estudiante_id = e.id 
                        ORDER BY a.fecha DESC 
                        LIMIT %s
                    """, (limit,))
                    return cursor.fetchall()
            except Exception as e:
                logging.error(f"Error al obtener datos de asistencia: {e}")
                return []

    @staticmethod
    def fetch_estudiantes() -> List[Tuple]:
        with DatabaseConnection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM estudiante")
                    return cursor.fetchall()
            except Exception as e:
                logging.error(f"Error al obtener estudiantes: {e}")
                return []

    @staticmethod
    def fetch_usuarios() -> List[Tuple]:
        with DatabaseConnection() as conn:
            if not conn:
                return []
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(DISTINCT estudiante_id) 
                        FROM asistencia 
                        WHERE fecha >= CURDATE()
                    """)
                    return cursor.fetchall()
            except Exception as e:
                logging.error(f"Error al obtener usuarios: {e}")
                return []

    @staticmethod
    def export_to_csv(datos: List[Tuple], filename: Optional[str] = None) -> str:
        if not filename:
            fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exportacion_asistencia_{fecha_actual}.csv"
        
        try:
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Nombre", "Fecha", "Estado"])
                writer.writerows(datos)
            logging.info(f"Datos exportados exitosamente a {filename}")
            return filename
        except Exception as e:
            logging.error(f"Error al exportar datos: {e}")
            raise

class DashboardPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Sección de Estadísticas
        self.create_stats_section()
        
        # Sección de Tabla de Asistencia
        self.create_attendance_table()
        
        # Sección de Gráficos de Asistencia
        self.create_attendance_charts()
        
        # Botones de Control
        self.create_control_buttons()
        
    def create_stats_section(self):
        stats_frame = ctk.CTkFrame(self)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        # Obtener datos para estadísticas
        estudiantes = DataManager.fetch_estudiantes()
        usuarios = DataManager.fetch_usuarios()
        asistencia = DataManager.fetch_asistencia()
        
        # Crear tarjetas de estadísticas
        self.create_stat_card(stats_frame, "Total Estudiantes", len(estudiantes), 0)
        self.create_stat_card(stats_frame, "Asistencia Hoy", 
                            usuarios[0][0] if usuarios else 0, 1)
        self.create_stat_card(stats_frame, "Total Registros", 
                            len(asistencia), 2)
        
    def create_stat_card(self, parent, title: str, value: int, column: int):
        card = ctk.CTkFrame(parent)
        card.grid(row=0, column=column, padx=10, pady=5, sticky="nsew")
        
        ctk.CTkLabel(card, text=title, font=("Roboto", 12)).pack(pady=5)
        ctk.CTkLabel(card, text=str(value), font=("Roboto", 20, "bold")).pack(pady=5)
        
    def create_attendance_table(self):
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Encabezados
        columns = ["ID", "Nombre", "Fecha", "Estado"]
        for i, col in enumerate(columns):
            ctk.CTkLabel(
                self.table_frame,
                text=col,
                font=("Roboto", 12, "bold")
            ).grid(row=0, column=i, padx=5, pady=5, sticky="w")
            
        self.actualizar_tabla()
        
    def actualizar_tabla(self):
        # Limpiar tabla existente (excepto encabezados)
        for widget in self.table_frame.grid_slaves():
            if int(widget.grid_info()["row"]) > 0:
                widget.destroy()
                
        # Obtener datos frescos
        datos = DataManager.fetch_asistencia()
        
        # Llenar tabla
        for i, row in enumerate(datos, start=1):
            for j, value in enumerate(row):
                if isinstance(value, datetime):
                    value = value.strftime("%Y-%m-%d %H:%M")
                    
                ctk.CTkLabel(
                    self.table_frame,
                    text=str(value),
                    font=("Roboto", 12)
                ).grid(row=i, column=j, padx=5, pady=2, sticky="w")
                
    def create_attendance_charts(self):
        self.chart_frame = ctk.CTkFrame(self)
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Obtener datos para gráficos
        self.data = DataManager.fetch_asistencia()
        self.total_asistencia = sum(row[2] for row in self.data)
        self.total_ausencias = sum(row[3] for row in self.data)
        
        if len(self.data) > 0:
            self.promedio_ausencias = self.total_ausencias / len(self.data)
        else:
            self.promedio_ausencias = 0
        
        # Crear gráficos
        self.create_attendance_line_chart()
        self.create_absences_bar_chart()
        self.create_attendance_pie_chart()
        
    def create_attendance_line_chart(self):
        self.figura_asistencia = Figure(figsize=(6, 4), facecolor="#3E3E3E")
        self.grafico_asistencia = self.figura_asistencia.add_subplot(111)
        self.grafico_asistencia.plot([row[1] for row in self.data], [row[2] for row in self.data], color='cyan')
        self.grafico_asistencia.set_xlabel('Fecha', color='white')
        self.grafico_asistencia.set_ylabel('Asistencia', color='white')
        self.grafico_asistencia.set_title('Asistencia a lo Largo del Tiempo', color='white')
        self.grafico_asistencia.tick_params(axis='x', colors='white')
        self.grafico_asistencia.tick_params(axis='y', colors='white')
        
        self.canvas_asistencia = FigureCanvasTkAgg(self.figura_asistencia, self.chart_frame)
        self.canvas_asistencia.draw()
        self.canvas_asistencia.get_tk_widget().pack(side="left", padx=10, pady=10)
        
    def create_absences_bar_chart(self):
        self.figura_ausencias = Figure(figsize=(6, 4), facecolor="#3E3E3E")
        self.grafico_ausencias = self.figura_ausencias.add_subplot(111)
        self.grafico_ausencias.bar([row[0] for row in self.data], [row[3] for row in self.data], color='orange')
        self.grafico_ausencias.set_xlabel('Estudiante', color='white')
        self.grafico_ausencias.set_ylabel('Ausencias', color='white')
        self.grafico_ausencias.set_title('Ausencias por Estudiante', color='white')
        self.grafico_ausencias.tick_params(axis='x', colors='white')
        self.grafico_ausencias.tick_params(axis='y', colors='white')
        
        self.canvas_ausencias = FigureCanvasTkAgg(self.figura_ausencias, self.chart_frame)
        self.canvas_ausencias.draw()
        self.canvas_ausencias.get_tk_widget().pack(side="left", padx=10, pady=10)
        
    def create_attendance_pie_chart(self):
        if self.total_asistencia > 0 or self.total_ausencias > 0:
            self.figura_pie = Figure(figsize=(6, 4), facecolor="#3E3E3E")
            self.grafico_pie = self.figura_pie.add_subplot(111)
            self.grafico_pie.pie(
                [self.total_asistencia, self.total_ausencias],
                labels=["Asistencia", "Ausencias"],
                autopct='%1.1f%%',
                colors=["#00FF00", "#FF4500"]
            )
            self.canvas_pie = FigureCanvasTkAgg(self.figura_pie, self.chart_frame)
            self.canvas_pie.draw()
            self.canvas_pie.get_tk_widget().pack(side="left", padx=10, pady=10)
        
    def create_control_buttons(self):
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkButton(
            button_frame,
            text="Actualizar Datos",
            command=self.actualizar_tabla,
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="Exportar a CSV",
            command=self.exportar_datos,
            width=150
        ).pack(side="left", padx=5)
        
    def exportar_datos(self):
        datos = DataManager.fetch_asistencia()
        if datos:
            try:
                filename = DataManager.export_to_csv(datos)
                self.show_message(f"Datos exportados a {filename}")
            except Exception as e:
                self.show_message(f"Error al exportar: {str(e)}", "error")
                
    def show_message(self, message: str, msg_type: str = "info"):
        color = "red" if msg_type == "error" else "green"
        label = ctk.CTkLabel(
            self,
            text=message,
            text_color=color,
            font=("Roboto", 12)
        )
        label.pack(pady=5)
        self.after(3000, label.destroy)

def mostrar_panel_datos(frame_principal: ctk.CTkFrame, callback_return):
    # Limpiar frame principal
    for widget in frame_principal.winfo_children():
        widget.destroy()
        
    # Crear y mostrar el dashboard
    dashboard = DashboardPanel(frame_principal)
    dashboard.pack(fill="both", expand=True)
    
    # Botón para volver
    ctk.CTkButton(
        frame_principal,
        text="Volver al Menú Principal",
        command=lambda: callback_return(frame_principal),
        width=200
    ).pack(pady=10)

if __name__ == "__main__":
    app = ctk.CTk()
    app.title("Panel de Asistencia Escolar")
    app.configure(bg="#2E2E2E")
    mostrar_panel_datos(app, lambda x: x.destroy())
    app.mainloop()