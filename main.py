import cv2
from colorthief import ColorThief

# Membuka kamera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

print("Tekan 's' untuk mengambil gambar dan mendeteksi warna dominan.")
print("Tekan 'q' untuk keluar.")

# Warna dominan default (hitam)
dominant_colors = [(0, 0, 0), (0, 0, 0)]

while True:
    # Membaca frame dari kamera
    ret, image = camera.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Menampilkan warna dominan di sudut kiri atas
    frame_height, frame_width, _ = image.shape
    patch_size = 50  # Ukuran patch warna

    for i, color in enumerate(dominant_colors):
        start_x = i * (patch_size + 5)  # Memberi jarak antara patch warna
        end_x = start_x + patch_size
        start_y, end_y = 0, patch_size

        # Menambahkan patch warna ke frame
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color[::-1], -1)  # BGR

    # Menampilkan frame
    cv2.imshow('Live Camera', image)

    # Deteksi input tombol
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Menyimpan gambar saat tombol 's' ditekan
        filename = 'captured_image.jpg'
        cv2.imwrite(filename, image)
        print(f"Gambar disimpan sebagai {filename}")

        # Mendeteksi warna dominan menggunakan ColorThief
        def detect_colors():
            colorthief = ColorThief(filename)
            # Mengambil palet warna (2 warna dominan)
            colors = colorthief.get_palette(color_count=2)
            return colors

        dominant_colors = detect_colors()
        print("Warna dominan terdeteksi:", dominant_colors)

    elif key == ord('q'):
        # Keluar saat tombol 'q' ditekan
        print("Keluar dari program.")
        break

# Membersihkan resource
camera.release()
cv2.destroyAllWindows()
