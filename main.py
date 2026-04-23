# Installation (einmal ausführen)
# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2

def main():
    # Modell laden (wird automatisch heruntergeladen)
    model = YOLO("yolov8n.pt")

    # Bildpfad (anpassen!)
    image_path = "image.jpg"

    # Bild laden
    image = cv2.imread(image_path)

    if image is None:
        print("Fehler: Bild konnte nicht geladen werden.")
        return

    # Inferenz durchführen
    results = model(image)

    # Ergebnisse zeichnen
    annotated_frame = results[0].plot()

    # Anzeigen
    cv2.imshow("YOLO Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: speichern
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Ergebnis gespeichert unter: {output_path}")

if __name__ == "__main__":
    main()
