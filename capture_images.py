from picamera import PiCamera
from time import sleep
import datetime

now = datetime.datetime.now()

ruta = "/home/pi/Desktop/imagenes/"+str(now)+".jpg"

camara = PiCamera()
camara.start_preview()
sleep(5)
camara.capture(ruta)
camara.stop_preview()


