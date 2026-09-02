from database import get_connection


connection = get_connection()

print("PostgreSQL bağlantısı başarılı!")

connection.close()