import cv2
import numpy as np
import webcolors

# ==========================
# Optimasi: Cache Database Warna
# ==========================
# Kita simpan data warna di awal agar tidak memanggil fungsi webcolors berulang kali
CSS3_NAMES = webcolors.names("css3")
CSS3_RGB_VALUES = [webcolors.name_to_rgb(n) for n in CSS3_NAMES]

def closest_color(rgb):
    """
    Mencari nama warna terdekat menggunakan jarak Euclidean.
    """
    min_dist = float('inf')
    closest_name = None
    
    # Konversi input ke array numpy untuk perhitungan vektor (lebih cepat)
    r, g, b = rgb
    
    # Kita iterasi manual (atau bisa pakai KDTree untuk lebih cepat jika list sangat besar)
    for i, name in enumerate(CSS3_NAMES):
        r_c, g_c, b_c = CSS3_RGB_VALUES[i]
        dist = (r - r_c)**2 + (g - g_c)**2 + (b - b_c)**2
        
        if dist < min_dist:
            min_dist = dist
            closest_name = name
            
    return closest_name

# ==========================
# Fungsi K-Means dengan OpenCV
# ==========================
def get_dominant_colors_cv2(image, k=3):
    """
    Menggunakan cv2.kmeans yang jauh lebih cepat dari sklearn
    """
    # 1. Resize gambar agar hitungan lebih ringan (Downscaling)
    # Mengubah 200x200 (40.000 pixel) menjadi 50x50 (2.500 pixel)
    # Warna dominan tetap akurat, tapi 16x lebih cepat.
    img_small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    
    img = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    
    # Harus convert ke float32 untuk cv2.kmeans
    img = np.float32(img)

    # Kriteria penghentian: (tipe, max_iter, epsilon)
    # Berhenti jika iterasi mencapai 10 atau akurasi mencapai 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Flags
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # Jalankan K-Means
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, flags)
    
    # Konversi kembali ke uint8
    return centers.astype(int)

# ==========================
# Main Program
# ==========================
camera = cv2.VideoCapture(0)

# Set resolusi kamera (opsional, untuk memperingan beban)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("❌ Kamera tidak dapat diakses")
    exit()

print("Tekan 'q' untuk keluar")

# Variabel untuk Frame Skipping
frame_counter = 0
skip_frames = 5  # Hitung warna setiap 5 frame sekali
cached_colors = []      # Menyimpan warna terakhir
cached_names = []       # Menyimpan nama warna terakhir

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
    
    # Gambar kotak ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # ==========================
    # Proses Deteksi (Frame Skipping)
    # ==========================
    # Hanya jalankan algoritma berat jika sisa bagi frame_counter dengan skip_frames adalah 0
    if frame_counter % skip_frames == 0:
        cached_colors = get_dominant_colors_cv2(roi, k=3)
        cached_names = [closest_color(tuple(c)) for c in cached_colors]
    
    frame_counter += 1

    # ==========================
    # Tampilkan Hasil (Menggunakan Data Cache)
    # ==========================
    # Jika cached_colors kosong (frame pertama), skip dulu
    if len(cached_colors) > 0:
        for i, color in enumerate(cached_colors):
            bgr = tuple(int(c) for c in color[::-1]) # RGB to BGR
            name = cached_names[i]

            start_x = 10
            start_y = 10 + i * 60
            end_x = start_x + 50
            end_y = start_y + 50

            # Gambar kotak warna
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), bgr, -1)
            
            # Outline kotak agar terlihat jika warnanya hitam/putih
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (200, 200, 200), 1)

            # Teks bayangan (outline) agar terbaca di background apapun
            cv2.putText(frame, name, (end_x + 10, start_y + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            # Teks utama
            cv2.putText(frame, name, (end_x + 10, start_y + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Real-Time Color Detection (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()