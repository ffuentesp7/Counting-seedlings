import socket # in order to know if there is a connection
import os # to read the images in the folder
import MySQLdb as dbapi 
import sys 
from os import remove

print("With internet connection")
directorio_imagenes = 'Rasberry images folder path'
contenido = os.listdir(directorio_imagenes)
imagenes = []
for fichero in contenido:
 if os.path.isfile(os.path.join(directorio_imagenes, fichero)) and fichero.endswith('.jpg'):
  try:
   fin = open(directorio_imagenes+fichero,"rb") #open file enter database
   img = fin.read() #read file
   fin.close() #close file

  except IOError, e: #exception in case of error
   print ("failure to obtain the image")
   break

  try:
   conn = dbapi.connect(host='ip server',user='user_credential', passwd='password_credential', db='database') #database connection
   cursor = conn.cursor() 
   cursor.execute("INSERT INTO imagenes SET img = '%s'" % \
   dbapi.escape_string(img)) #executes image input sentence
   conn.commit() #close transaction
   cursor.close() 
   conn.close() #close connection
   remove(directorio_imagenes+fichero)

  except IOError, e: #generates exception in case of error
   print("failure to connect to the database")
   break

