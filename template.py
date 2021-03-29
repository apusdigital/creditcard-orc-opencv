# Credit card digits detector - https://github.com/alexcamargoweb/creditcard-orc-opencv.
# Reconhecimento de dígitos de um cartão de crédito utilizando Python e OpenCV.
# Adrian Rosebrock, Credit card OCR with OpenCV and Python. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/.
# Acessado em: 29/03/2021.
# Arquivo: template.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate python_ocr

# importa os pacotes necessários
from imutils import contours
import numpy as np
import imutils
import cv2

# imagem de entrada
IMAGE = './inputs/5.png'
# imagem OCR-A de referência (dígitos de 0 a 9)
REFERENCE = './reference/ocr_a_reference.png'
# define um dicionário para mapear o primeiro dígito do cartão
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# carrega a imagem OCR-A de referência
ref = cv2.imread(REFERENCE)
# converte a imagem para escala de cinza
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# converte os dígitos da imagem em branco e fundo em preto
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# busca as bordas da imagem OCR-A (bordas dos dígitos)
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
# ordena da esquerda para a direita
refCnts = contours.sort_contours(refCnts, method = "left-to-right")[0]
# inicializa um dicionário para mapear o dígito ROI
digits = {}

# faz um loop sobre os contornos da imagem OCR-A de referência
for (i, c) in enumerate(refCnts):
    # processa a bouding box para o dígito, extrai e redimensiona (tamanho fixo)
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # atualiza o dicionário de dígitos, mapeando o ROI
    digits[i] = roi

# inicializa um kernel estrutural retangular
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# inicializa um kernel estrutural quadrado
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# carrega a imagem de entrada
image = cv2.imread(IMAGE)
# redimensiona para 300px mantendo a proporção
image = imutils.resize(image, width = 300)
# converte para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# aplica um operador morfológico tophat (whitehat) para encontrar a luz
# de regiões contra um fundo escuro (ou seja, os números do cartão de crédito)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# processa o gradiente Scharr do tophat da imagem
gradX = cv2.Sobel(tophat, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# reescala o resto da imagem para num intervalo de [0, 255]
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# aplica uma operação de fechamento usando o kernel retangular para ajudar
# a fechar as lacunas entre os dígitos do número do cartão de crédito
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# aplica o método de limiar de Otsu para binarizar a imagem
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# aplica uma segunda operação de fechamento à imagem binária, novamente
# para ajudar a fechar lacunas entre as regiões de número do cartão de crédito
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

# encontra os contornos de limite na imagem
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# inicializa a lista de locais dos dígitos
locs = []

# faz um loop sobre os contornos
for (i, c) in enumerate(cnts):
    # calcula a bouding box do contorno
    (x, y, w, h) = cv2.boundingRect(c)
    # usa as coordenadas para derivar a proporção
    ar = w / float(h)
    # já que os cartões de crédito são fixados em 4 grupos de 4 dígitos
    # é possível destacar os potenciais contornos baseados na proporção
    if ar > 2.5 and ar < 4.0:
        # os contornos podem ser destacados em largura e altura mínima/máxima
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # acrescenta a bouding box do grupo de dígitos na lista de localizações
            locs.append((x, y, w, h))

# ordena a localização dos dígitos da esquerda para a direita
locs = sorted(locs, key = lambda x:x[0])
# inicializa a lista dos dígitos classificados
output = []

# faz um loop sobre os 4 grupos de 4 dígitos
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # inicializa a lista do grupo de 4 dígitos
    groupOutput = []
    # extrai o ROI do grupo de 4 dígitos da imagem (escala de cinza)
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # aplica o limite para segmentar os dígitos do fundo do cartão de crédito
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # detecta os contornos de cada dígito individual no grupo
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = imutils.grab_contours(digitCnts)
    # ordena os contornos do dígito da esquerda para a direita
    digitCnts = contours.sort_contours(digitCnts, method = "left-to-right")[0]

    # faz um loop sobre os contornos dos dígitos
    for c in digitCnts:
        # calcula a bouding box do dígito individual
        (x, y, w, h) = cv2.boundingRect(c)
        # extrai o dígito
        roi = group[y:y + h, x:x + w]
        # redimensiona para ter o mesmo tamanho fixo da imagem de referência OCR-A
        roi = cv2.resize(roi, (57, 88))
        # inicializa uma lista dos scores do template
        scores = []

        # faz um loop sobre o dígito de referência e o ROI do dígito
        for (digit, digitROI) in digits.items():
            # aplica uma correlação baseada no template
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            # pega o score
            (_, score, _, _) = cv2.minMaxLoc(result)
            # atualiza lista de score
            scores.append(score)

        # a classificação para o ROI é a referência
        # do dígito com maior score no template
        groupOutput.append(str(np.argmax(scores)))

        # desenha a classificação dos dígitos acima do grupo
        cv2.rectangle(image, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # atualiza a lista dos dígitos de output
    output.extend(groupOutput)

# exibe as informações do cartão de crédito na tela
print("Bandeira: {}".format(FIRST_NUMBER[output[0]]))
print("Número: {}".format("".join(output)))
cv2.imshow("Credit card digits detector", image)
cv2.waitKey(0)