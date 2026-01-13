import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

# ==========================
# Fungsi cari nama warna terdekat
# ==========================
def closest_color(rgb):
    min_dist = float('inf')
    closest_name = None

    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        dist = (rgb[0] - r_c) ** 2 + (rgb[1] - g_c) ** 2 + (rgb[2] - b_c) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name



# ==========================
# Ambil warna dominan dari ROI
# ==========================
def get_dominant_colors(image, k=3):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img)

    return kmeans.cluster_centers_.astype(int)


# ==========================
# Buka Kamera
# ==========================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Kamera tidak dapat diakses")
    exit()

print("Tekan 'q' untuk keluar")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ==========================
    # ROI (Area Tengah)
    # ==========================
    roi_size = 200
    x1 = w // 2 - roi_size // 2
    y1 = h // 2 - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ==========================
    # Deteksi warna dominan
    # ==========================
    colors = get_dominant_colors(roi, k=3)

    # ==========================
    # Tampilkan kotak warna
    # ==========================
    for i, color in enumerate(colors):
        bgr = tuple(int(c) for c in color[::-1])
        name = closest_color(tuple(color))

        start_x = 10
        start_y = 10 + i * 60
        end_x = start_x + 50
        end_y = start_y + 50

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), bgr, -1)
        cv2.putText(
            frame,
            name,
            (end_x + 10, start_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow("Real-Time Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
