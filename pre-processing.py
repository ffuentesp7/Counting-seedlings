import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import imutils
import heapq


def homography(X,H, N, M, W):

  Hinv  = np.linalg.inv(H)
  Xp    = np.zeros((N,M,W),np.uint8)

  m = np.ones((N*M,3))
  t = 0
  for i in range(N):
    for j in range(M):
      m[t,0:2] = [j,i]
      t = t+1

  mph = np.dot(Hinv,m.T)  # Transfromacion de M a Mp
  mp  = np.divide(mph[0:2,:],mph[2,:])
  mpf = np.fix(mp).astype(int)

  ip = mpf[1,:]
  jp = mpf[0,:]
  ktj = np.logical_and(jp>=0,jp<M)
  kti = np.logical_and(ip>=0,ip<N)
  kt  = np.logical_and(kti,ktj)

  t = 0
  for i in range(N):
    for j in range(M):
      if kt[t]:
        Xp[i,j] = X[ip[t],jp[t]]
      t = t+1
  return Xp


def homography_matrix(m,mp):
  (x ,y ) = m
  (xp,yp) = mp
  n       = len(x)
  A       = np.zeros((2*n,9))
  for i in range(n):
    j = i*2
    A[j  ,:] = [x[i], y[i], 1,     0,     0, 0, -x[i]*xp[i], -y[i]*xp[i], -xp[i]]
    A[j+1,:] = [  0  ,   0  , 0, x[i], y[i], 1, -x[i]*yp[i], -y[i]*yp[i], -yp[i]]
  [U,S,V] = np.linalg.svd(A)
  h       = V[-1,:]
  H       = np.vstack([h[0:3], h[3:6], h[6:9]])
  return H


def buscarValores(valor, listaPuntosX, listaPuntosY, ArrayPuntosFinales):

	if(valor[0] == valor[1]):
		result = np.where(np.array(listaPuntosX) == valor[0])
		
		y1 = listaPuntosY[result[0][0]];
		y2 = listaPuntosY[result[0][1]];

		if(y1 < y2):
			ArrayPuntosFinales.append((valor[0], y1))
			ArrayPuntosFinales.append((valor[0], y2))
		else:
			ArrayPuntosFinales.append((valor[0], y2))
			ArrayPuntosFinales.append((valor[0], y1))

	else:
			indexY1 = listaPuntosX.index((valor[0]))
			indexY2 = listaPuntosX.index((valor[1]))

			y1 = listaPuntosY[indexY1];
			y2 = listaPuntosY[indexY2];

			if(y1 < y2):
				ArrayPuntosFinales.append((valor[0], y1))
				ArrayPuntosFinales.append((valor[1], y2))
			else:
				ArrayPuntosFinales.append((valor[1], y2))
				ArrayPuntosFinales.append((valor[0], y1))

	return ArrayPuntosFinales


def ordenarPuntos(listaPuntosX, listaPuntosY, ArrayPuntosFinales): 

	x1_2 = heapq.nsmallest(2, listaPuntosX)
	ArrayPuntosFinales = buscarValores(x1_2, listaPuntosX, listaPuntosY, ArrayPuntosFinales)

	x3_4 = heapq.nlargest(2, listaPuntosX)
	ArrayPuntosFinales = buscarValores(x3_4, listaPuntosX, listaPuntosY, ArrayPuntosFinales)

	return ArrayPuntosFinales


def procesarImagen(imagen, ruta_salida, nombre_imagen):

	X = cv2.imread(imagen); # reading in opencv format (BGR)
	(N,M,W) = X.shape

	#rangos de color rojo en el espacio HSV
	rojoBajo1 = np.array([0,140,90], np.uint8)
	rojoAlto1 = np.array([8,255,255], np.uint8)
	rojoBajo2 = np.array([160,140,90], np.uint8)
	rojoAlto2 = np.array([180,255,255], np.uint8)

	#transform to other color spaces
	imageGrey = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
	imageHSV = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)

	#DETECTING THE COLOR RED
	maskRojo1 = cv2.inRange(imageHSV, rojoBajo1, rojoAlto1)
	maskRojo2 = cv2.inRange(imageHSV, rojoBajo2, rojoAlto2)
	mask = cv2.add(maskRojo1, maskRojo2)

	# reduce image noise
	mask = cv2.medianBlur(mask, 5)
	
	kernel = np.ones((7,7), np.uint8);
	mask = cv2.erode(mask, kernel, iterations=1)

	'''
	plt.figure(figsize=(15,15))
	plt.imshow(mask,cmap='gray')
	plt.show()
	'''

	canny = cv2.Canny(mask, 10, 150)

	cnts,jerarquia = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	#Moments to calculate the centroid of my circles

	listaPuntosX = [];
	listaPuntosY = [];
	for i in range(4):
		ctn = cnts[i];
		Mom = cv2.moments(ctn)
		cX = int(Mom["m10"] / Mom["m00"])
		cY = int(Mom["m01"] / Mom["m00"])
		listaPuntosX.append(cX)
		listaPuntosY.append(cY)

	ArrayPuntosFinales = [];

	ArrayPuntosFinales = ordenarPuntos(listaPuntosX, listaPuntosY, ArrayPuntosFinales)

	(x1,y1) = ArrayPuntosFinales[0]
	(x2,y2) = ArrayPuntosFinales[1]
	(x3,y3) = ArrayPuntosFinales[2]
	(x4,y4) = ArrayPuntosFinales[3]
	
	cv2.circle(X, (x1, y1), 3, (255,255,0), -1)
	cv2.circle(X, (x2, y2), 3, (255,255,0), -1)
	cv2.circle(X, (x3, y3), 3, (255,255,0), -1)
	cv2.circle(X, (x4, y4), 3, (255,255,0), -1)

	'''
	plt.figure(figsize=(15,15))
	plt.imshow(X,cmap='gray')
	plt.show()
	'''
	
	# Generating the square in the image to be cropped.

	xp = np.array([x1,x2,x3,x4])
	yp = np.array([y1,y2,y3,y4])
	i  = np.array([0,1,3,2,0])

	'''
	plt.figure(figsize=(15,15))
	plt.imshow(X,cmap='gray')
	plt.plot(xp[i],yp[i])
	plt.scatter(xp,yp,c='red')
	plt.show()
	'''

	# Coordinates of the resulting image
	x = np.array([0,M,0,M])
	y = np.array([0,0,N,N])

	mm = np.vstack([x, y, np.array([1, 1, 1, 1])])
	H2 = homography_matrix((x,y),(xp,yp))
	mmp = np.dot(H2,mm)
	mmp = mmp/mmp[-1,:]
	Hinv  = np.linalg.inv(H2)
	imagen_homografiada = homography(X,Hinv, N,M,W)

	#EQUALIZATION OF THE GREEN CHANNEL OF THE HSV IMAGE.

	Khsv = cv2.cvtColor(imagen_homografiada, cv2.COLOR_BGR2HSV)
	Khsv[:,:,1] = cv2.equalizeHist(imagen_homografiada[:,:,1])
	imagen_homografiada_nueva = cv2.cvtColor(Khsv, cv2.COLOR_HSV2BGR)
	
	cv2.imwrite(os.path.join(ruta_salida,nombre_imagen), imagen_homografiada_nueva)
	
	print('image successfully processed');

ruta_imagenes = "/Image_path_with_red_manual_vertex/"
salida_imagenes = '/path_pre-processed_images/';
nombre_archivos = os.listdir(ruta_imagenes);


for fichero in nombre_archivos:
    if os.path.isfile(os.path.join(ruta_imagenes, fichero)) and fichero.endswith('.jpg'):
    	imagen = ruta_imagenes+fichero;
    	print(imagen);

    	procesarImagen(imagen, salida_imagenes, fichero);
