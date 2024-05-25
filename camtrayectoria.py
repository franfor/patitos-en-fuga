import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros para el flujo óptico de Lucas-Kanade
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

# Parámetros para la detección de características ShiTomasi
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Inicialización
cap = cv.VideoCapture('pollitos.mp4')
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

# Listas para almacenar las posiciones 3D estimadas
trajectory_3d = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Calcular el flujo óptico
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Seleccionar los puntos buenos
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) < 5 or len(good_old) < 5:
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        continue

    try:
        # Calcular la matriz esencial y la pose de la cámara
        E, mask = cv.findEssentialMat(good_new, good_old, focal=1.0, pp=(0.5*frame.shape[1], 0.5*frame.shape[0]), method=cv.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv.recoverPose(E, good_new, good_old)
        
        # Acumular la trayectoria de la cámara
        if len(trajectory_3d) == 0:
            trajectory_3d.append((0, 0, 0))  # Punto inicial
        else:
            x, y, z = trajectory_3d[-1]
            dx, dy, dz = t.flatten()
            trajectory_3d.append((x + dx, y + dy, z + dz))
    except cv.error as e:
        print(f"Error al recuperar la pose: {e}")
        continue
    
    # Dibujar las trayectorias
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
    
    cv.imshow('frame', frame)
    
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# Convertir la trayectoria a un array numpy para graficar
trajectory_3d = np.array(trajectory_3d)

# Graficar la trayectoria 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], label='Trajectoria de la cámara')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
