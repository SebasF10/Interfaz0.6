import pymysql
from datetime import datetime

# Registrar asistencia y conectar en la misma función
def registrar_asistencia(codigo_est):
    try:
        # Conexión a la base de datos remota
        conexion = pymysql.connect(host='b4qhbwwqys2nhher1vul-mysql.services.clever-cloud.com', port= 3306, db='b4qhbwwqys2nhher1vul', user='upvge9afjesbmmgv', password='BS2bxJNACO1XYEmWBqA0')

        cursor = conexion.cursor()

        # Verificar si el estudiante existe
        consulta_estudiante = "SELECT * FROM estudiante WHERE codigo_est = %s"
        cursor.execute(consulta_estudiante, (codigo_est,))
        estudiante = cursor.fetchone()

        if estudiante:
            # Obtener fecha y hora actual
            fecha_actual = datetime.now().strftime("%Y-%m-%d")
            hora_actual = datetime.now().strftime("%H:%M:%S")

            # Insertar asistencia en la tabla
            insertar_asistencia = "INSERT INTO ingreso (codigo_est, fecha, hora) VALUES (%s, %s, %s)"
            cursor.execute(insertar_asistencia, (codigo_est, fecha_actual, hora_actual))
            conexion.commit()

            print(f"Asistencia registrada para el estudiante: {estudiante[2]} {estudiante[1]} (Grupo: {estudiante[3]}, Jornada: {estudiante[4]})")
        else:
            print("Estudiante no encontrado.")
    
    finally:
        try:
            if cursor:
                cursor.close()
        except NameError:
            pass
        try:
            if conexion:
                conexion.close()
        except NameError:
            pass