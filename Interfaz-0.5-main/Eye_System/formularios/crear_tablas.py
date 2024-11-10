# crear_tablas.py

import pymysql

def crear_tablas():
    # Parámetros de conexión
    connection_params = {
        'host': 'b4qhbwwqys2nhher1vul-mysql.services.clever-cloud.com',
        'port': 3306,
        'db': 'b4qhbwwqys2nhher1vul',
        'user': 'upvge9afjesbmmgv',
        'password': 'BS2bxJNACO1XYEmWBqA0'
    }

    try:
        # Establecer conexión
        conn = pymysql.connect(**connection_params)
        cursor = conn.cursor()

        # SQL para crear tabla estudiante
        crear_tabla_estudiante = """
        CREATE TABLE IF NOT EXISTS estudiante (
            id INT AUTO_INCREMENT PRIMARY KEY,
            codigo_est VARCHAR(20) NOT NULL,
            apellidos VARCHAR(100) NOT NULL,
            nombres VARCHAR(100) NOT NULL,
            grupo VARCHAR(50) NOT NULL,
            jornada VARCHAR(20) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        # SQL para crear tabla asistencia
        crear_tabla_asistencia = """
        CREATE TABLE IF NOT EXISTS asistencia (
            id INT AUTO_INCREMENT PRIMARY KEY,
            estudiante_id INT,
            nombre VARCHAR(200) NOT NULL,
            fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
            estado VARCHAR(20) NOT NULL,
            FOREIGN KEY (estudiante_id) REFERENCES estudiante(id)
        );
        """

        # Ejecutar las consultas
        print("Creando tabla estudiante...")
        cursor.execute(crear_tabla_estudiante)
        
        print("Creando tabla asistencia...")
        cursor.execute(crear_tabla_asistencia)

        # Confirmar los cambios
        conn.commit()
        print("¡Tablas creadas exitosamente!")

    except Exception as e:
        print(f"Error al crear las tablas: {e}")

    finally:
        if 'conn' in locals():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    crear_tablas()