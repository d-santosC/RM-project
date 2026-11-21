
import cv2
import pytesseract

#caminho do tesseract
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream")
        return

    print("Recognition of Writing")
    print("Press 'q' or 'ESC' to exit")

    #configuração do tesseract para reconhecer apenas letras
    #oem 3: usar o melhor mecanismo de OCR
    #psm 10: tratar a imagem como uma unica letra / psm 8: tratar a imagem como uma unica palavra
    custom_config = (
        r'--oem 3 --psm 8 '
        r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        r'-c load_system_dawg=0 '
        r'-c load_freq_dawg=0'
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        #flip da imagem para efeito de espelho
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        #definir uma caixa central para colocar a escrita
        # box_size = 250
        # x1 = (w - box_size) // 2
        # y1 = (h - box_size) // 2
        # x2 = x1 + box_size
        # y2 = y1 + box_size

        box_w, box_h = 500, 300
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        roi = frame[y1:y2, x1:x2]

        roi = cv2.flip(roi, 1)

        #processamento da imagem
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #escala de cinzentos
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0) #reduzir ruido

        #binarização adaptativa
        _, thresh = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        #remover pequenos pontos de ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,151,3)

        recognized_text = ""

        #encontrar os limites da escrita
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contours = [c for c in contours if cv2.contourArea(c) > 50]

        if contours:

            #recortar apenas a letra
            boxes = [cv2.boundingRect(c) for c in contours]

            x_min = min(b[0] for b in boxes)
            y_min = min(b[1] for b in boxes)
            x_max = max(b[0] + b[2] for b in boxes)
            y_max = max(b[1] + b[3] for b in boxes)

            #recortar a palavra
            word = thresh[y_min:y_max, x_min:x_max]

            #adicionar margem à volta da palavra
            word = cv2.copyMakeBorder(
                word,
                30, 30, 30, 30,
                cv2.BORDER_CONSTANT,
                value=0
            )

            #aumentar a resolução da imagem para melhorar o reconhecimento
            word = cv2.resize(
                word,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            ocr_image = cv2.bitwise_not(word) #inverter cores para o tesseract

            #rconhecimento OCR
            try:
                text = pytesseract.image_to_string(
                    ocr_image,
                    config=custom_config
                )

                recognized_text = text.strip()

            except Exception as e:
                print("OCR Error:", e)

            cv2.imshow("Processed Word", ocr_image)
            cv2.namedWindow("Processed Word", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Processed Word", 600, 200)

        #feedback visual
        color = (0, 255, 0) if recognized_text else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"Word: {recognized_text if recognized_text else '---'}"

        #mostrar texto reconhecido
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        cv2.putText(
            frame, "Place the writing in the box",
            (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        #windows
        cv2.imshow("Writing Recognition", frame)
        cv2.imshow("Threshold ROI", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
