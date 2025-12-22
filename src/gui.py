import cv2

def show(image_path, label, confidence, backend, latency):
    img = cv2.imread(image_path)

    cv2.putText(img, f"{label} ({confidence:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.putText(img, f"{backend} | {latency:.2f} ms",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2)

    cv2.imshow("Vehicle Classification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
