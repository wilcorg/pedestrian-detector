"""
Roman Szlachtun, 272330

Mierzy dokładność działania algorytmów Kollera, HOG-SVM oraz YOLO według
metryk testowych, jak precyzja, czułość i stosunek ramek TP do ludzi TP.

Czas działania algorytmów Kollera, HOG-SVM jest mierzony w
zoptymalizowanych implementacjach w C++.

Ponieważ YOLO nie ma oficjalnie wspieranego interfejsu z C++,
wydajność była mierzona w Python.
"""

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Final

import ultralytics.models
from PIL import Image, ImageDraw

import cv2
import numpy as np
from ultralytics import YOLO

from DirectoryAnchor import DATASET_DIR, SVM_MODEL_DIR, YOLO_MODEL_DIR, PROJECT_ROOT_DIR
from hogsvm.SvmConfig import SvmConfig
from koller.bw.py.Koller1 import KollerBw
from koller.yuv.py.Koller3 import KollerYuv as Koller

cv2.setUseOptimized(True)

# Przedziały klatek są domknięte
TESTING_FRAMES_RANGE: Final[Tuple[int, int]] = (4050, 4500)

"""
Definicje używanych struktur danych
"""
@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

@dataclass
class InferredBox:
    box: Box
    owner: int | None
    bestIntersection: float

@dataclass(frozen=True)
class Results:
    tp: int
    fp: int
    fn: int

@dataclass
class Predictions:
    tp: List[int]
    fp: List[int]
    fn: List[int]

@dataclass
class BorderMargins:
    topPercent: int
    rightPercent: int
    bottomPercent: int
    leftPercent: int
    intersectThreshold: float

"""
Klasa, owijąjąca implementację każdego detektora.
Inicjalizuje wybrany algorytm i przekazuje polecenie detekcji do wybranego detektora.
"""
class Detector:
    def __init__(self):
        self.model: cv2.HOGDescriptor | ultralytics.models.YOLO | KollerBw | Koller | None = None

    @staticmethod
    def initHog(npzPath: Path):
        detector = Detector()
        d = np.load(str(npzPath))
        w = d["w"].astype(np.float32).ravel()
        rho = float(d["rho"])
        detectorVec = np.hstack([w, -rho]).astype(np.float32)

        cfg = SvmConfig()
        model = cv2.HOGDescriptor(_winSize=cfg.windowSize, _blockSize=cfg.blockSize, _blockStride=cfg.blockStride, _cellSize=cfg.pixelPerCell, _nbins=cfg.orientationNumber)
        model.setSVMDetector(detectorVec)
        detector.model = model
        return detector

    @staticmethod
    def initYolo(modelPath: Path):
        detector = Detector()
        model = YOLO(str(modelPath), task='detect')
        detector.model = model
        return detector

    @staticmethod
    def initKollerBw(downscaleFactor: float):
        n = 20
        detector = Detector()
        detector.model = KollerBw(
            deltaThreshold=15,
            alpha1=0.003,
            alpha2=0.0003,
            minAreaFraction=15e-4,
        )

        """
        Algorytmy Kollera biorą klatki dla modelu tła ze zbioru walidacyjnego.
        Jest to dozwolone, ponieważ algorytmy nie potrzebują uczenia się obiektów.
        """
        initImgs = []
        start = 4050 - n * 25
        for i in range(start, 4050, 25):
            imgPath = DATASET_DIR / 'frames' / f"{i+1}.png"
            initImgs.append(imgPath)

        detector.model.backgroundModelInit(frameList=initImgs, downscaleFactor=downscaleFactor)
        return detector

    @staticmethod
    def initKoller(downscaleFactor: float):
        n = 20
        detector = Detector()
        detector.model = Koller(
            deltaUV=5,
            deltaY=15,
            alpha1=0.003,
            alpha2=0.0003,
            minAreaFraction=15e-4,
        )

        initImgs = []
        start = 4050 - n * 25
        for i in range(start, 4050, 25):
            imgPath = DATASET_DIR / 'frames' / f"{i+1}.png"
            initImgs.append(imgPath)

        detector.model.backgroundModelInit(frameList=initImgs, downscaleFactor=downscaleFactor)
        return detector

    """
    Metoda, odpowiedzialna za uruchomienie detekcji wybranego detektora.
    Na koniec detekcji obiektów z obrazu metoda zwraca listę wykrytych ramek oraz czas działania.
    
    Czas działania jest stosowany tylko dla podglądu i nie jest przetwarzany w statystykach.
    """
    def predict(self, img: np.ndarray) -> Tuple[List[InferredBox], float]:
        recognitions: List[InferredBox] = []
        if isinstance(self.model, cv2.HOGDescriptor):
            startTime = time.time()
            rects, weights = self.model.detectMultiScale(
                img,
                hitThreshold=0.3,
                winStride=(8, 8),
                padding=(0, 0),
                scale=1.2,
                groupThreshold=1
            )
            elapsed = time.time() - startTime

            for (x, y, w, h), score in zip(rects, weights):
                recognitions.append(InferredBox(box=Box(x=float(x), y=float(y), w=float(w), h=float(h)), owner=None, bestIntersection=0.0))
            return recognitions, elapsed
        elif isinstance(self.model, ultralytics.models.YOLO):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            startTime = time.time()
            results = self.model.predict(img, device='cpu', rect=True, verbose=False)
            elapsed = time.time() - startTime

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1
                    recognitions.append(InferredBox(box=Box(x=float(x1), y=float(y1), w=float(w), h=float(h)), owner=None, bestIntersection=0.0))
            return recognitions, elapsed
        elif isinstance(self.model, KollerBw):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            startTime = time.time()
            boxes = self.model.predict(img)
            elapsed = time.time() - startTime

            for box in boxes:
                recognitions.append(InferredBox(
                    box=Box(x=box.x1, y=box.y1, w=(box.x2 - box.x1), h=(box.y2 - box.y1)), owner=None, bestIntersection=0.0))
            return recognitions, elapsed
        elif isinstance(self.model, Koller):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            startTime = time.time()
            boxes = self.model.predict(img)
            elapsed = time.time() - startTime

            for box in boxes:
                recognitions.append(InferredBox(
                    box=Box(x=box.x1, y=box.y1, w=(box.x2 - box.x1), h=(box.y2 - box.y1)), owner=None, bestIntersection=0.0))
            return recognitions, elapsed
        else:
            return [], 0.0

    def getMode(self) -> str:
        if isinstance(self.model, cv2.HOGDescriptor):
            return "HOG+SVM"
        elif isinstance(self.model, ultralytics.models.YOLO):
            return "YOLO"
        elif isinstance(self.model, KollerBw):
            return "KollerBw"
        elif isinstance(self.model, Koller):
            return "Koller"
        else:
            return "Unknown"

    model: cv2.HOGDescriptor | ultralytics.models.YOLO | KollerBw | Koller | None = None

"""
Oblicza pole przecięcia dwóch ramek
"""
def boxIntersection(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    return inter

"""
Odczytuje oznaczenia zbioru danych
"""
def parseTop(gt_path: Path) -> Dict[int, List[Box]]:
    gt: Dict[int, List[Box]] = defaultdict(list)
    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            frame = int(float(row[1]))
            x = float(row[8])
            y = float(row[9])
            w = float(row[10]) - x
            h = float(row[11]) - y

            if w > 0 and h > 0:
                gt[frame].append(Box(x, y, w, h))
    return gt

"""
Redukuje wymiary oznaczeń zbioru danych wraz ze współczynnikiem downscaleFactor.

downscaleFactor zmniejsza rodzielczość obrazu wejściowego.
"""
def prepareFrameTop(frameNum: int, top: Dict[int, List[Box]], downscaleFactor: float) -> Tuple[List[InferredBox], float]:
    inferred: List[InferredBox] = []
    boxes = top.get(frameNum, [])
    for box in boxes:
        downscaledBox = Box(
            x=box.x / downscaleFactor,
            y=box.y / downscaleFactor,
            w=box.w / downscaleFactor,
            h=box.h / downscaleFactor,
        )
        inferred.append(InferredBox(box=downscaledBox, owner=None, bestIntersection=0.0))
    return inferred, 0.0

"""
Odczyt klatki nagrania i jej potencjalne przeskalowanie w dół.
"""
def prepareFrame(imgPath: Path, downscaleFactor: float) -> np.ndarray:
    img = cv2.imread(str(imgPath), cv2.IMREAD_COLOR_RGB)
    assert img.shape == (1080, 1920, 3)
    img = cv2.resize(img, (int(img.shape[1] / downscaleFactor), int(img.shape[0] / downscaleFactor)))
    return img

"""
Sprawdzenie, czy ramka istotnie wykracza poza pole obserwacji.

Jeśli większość ramki znajduje się poza polem obserwacji, 
detektor ma prawo pomylić się z jej wykryciem.
"""
def isOutsiderDetection(borderMargins: BorderMargins, box: InferredBox, imgShape: Tuple[int, int, int]) -> bool:
    fieldOfObservationBox = Box(
        x=imgShape[1] * (borderMargins.leftPercent / 100.0),
        y=imgShape[0] * (borderMargins.topPercent / 100.0),
        w=imgShape[1] - imgShape[1] * ((borderMargins.leftPercent + borderMargins.rightPercent) / 100.0),
        h=imgShape[0] - imgShape[0] * ((borderMargins.topPercent + borderMargins.bottomPercent) / 100.0)
    )

    observationIntersection = boxIntersection(box.box, fieldOfObservationBox)
    if observationIntersection / (box.box.w * box.box.h) < borderMargins.intersectThreshold:
        return True
    else:
        return False


def main():
    top = parseTop(DATASET_DIR / 'groundtruth.top')
    results: List[Results] = []

    startFrame, endFrame = TESTING_FRAMES_RANGE
    downscaleFactor: Final[float] = 1.0
    borderMargins: Final[BorderMargins] = BorderMargins(topPercent=10, rightPercent=5, bottomPercent=10, leftPercent=5, intersectThreshold=0.5)
    savePlots: Final[bool] = False

    # Wybór detektora dla detekcji

    # model = Detector.initHog(SVM_MODEL_DIR / 'stage4' / "16-32-8-9-l2hys.npz")
    model = Detector.initHog(SVM_MODEL_DIR / 'stage4' / "8-16-8-9-l2hys.npz")
    # model = Detector.initYolo(YOLO_MODEL_DIR / '256x352' / 'best_openvino_model_int8')
    # model = Detector.initKollerBw(downscaleFactor=downscaleFactor)
    # model = Detector.initKoller(downscaleFactor=downscaleFactor)

    saveDir: Path = PROJECT_ROOT_DIR / 'benchmark' / 'results' / f'{model.getMode()}'

    intersectThreshold: float = 0.0
    if model.getMode() == "YOLO":
        intersectThreshold = 0.5
    elif model.getMode() == "HOG+SVM":
        intersectThreshold = 0.5
    elif model.getMode() == "KollerBw" or model.getMode() == "Koller":
        intersectThreshold = 0.2

    if savePlots:
        os.makedirs(saveDir, exist_ok=True)

    boxesPerHumansResults = []

    for imgNum in range(startFrame, endFrame + 1):
        img = prepareFrame(DATASET_DIR / 'frames' / f"{imgNum}.png", downscaleFactor)
        # Pobranie okien predykcji detektora
        predictedBoxes, elapsedSec = model.predict(img)
        # Pobranie oczekiwanych oznaczeń ludzi
        expectedPeople, _ = prepareFrameTop(imgNum, top, downscaleFactor)

        predictions = Predictions(tp=list(), fp=list(), fn=list())
        skippedBoxes: List[Box] = []

        """
        Implementacja algorytmu doboru okien predykcji do oznaczeń ludzi
        Jest dokładnie opisany w Rozdziale 6 pracy dyplomowej 
        """
        for personId in range(len(expectedPeople)):
            for predictedId, predicted in enumerate(predictedBoxes):
                intersectionByPerson: float = boxIntersection(predicted.box, expectedPeople[personId].box) / (expectedPeople[personId].box.w * expectedPeople[personId].box.h)

                """
                Jeśli jest największe przecięcie ramki okna i ozaczenia człowieka,
                które jest większe od minimalnej wartości,
                to oznaczenie człwoieka jest "podpięte" pod okno i okno oraz człowiek stają się TP.
                
                Okno detekcji, wybrane przez człowieka, może ulec zmianie. Wtedy dotychczasowe
                okno przestaje być podpięte pod danego człowieka.
                """
                if intersectionByPerson > intersectThreshold and expectedPeople[personId].bestIntersection < intersectionByPerson:
                    expectedPeople[personId].owner = predictedId
                    expectedPeople[personId].bestIntersection = max(expectedPeople[personId].bestIntersection, intersectionByPerson)

        """
        Część, odpowiadająca za detekcję okien FP oraz oznaczeń ludzi FN.
        """
        personIdsToRemove: List[int] = []
        for personId in range(len(expectedPeople)):
            """
            Jeśli oznaczenie człowieka jest "podpięte" pod okno - człwoiek należy do TP
            """
            if expectedPeople[personId].owner is not None:
                predictions.tp.append(expectedPeople[personId].owner)
            else:
                """
                Jeśli oznaczenie człowieka jednak nie było wykryte,
                sprawdzane jest, czy oznaczenie nie znajduje się poza polem obserwacji.
                
                Jeśli oznaczenie jest poza polem obswerwacji, nie jest dodawane do FN.
                """
                if isOutsiderDetection(borderMargins, expectedPeople[personId], img.shape):
                    personIdsToRemove.append(personId)
                else:
                    predictions.fn.append(personId)

        predictedIdsToRemove: List[int] = []
        for predictedId in range(len(predictedBoxes)):
            """
            Jeśli żadny człowiek nie jest podpięty pod okno detekcji,
            oraz okno jest w zasięgu pola obserwacji,
            dane okno detekcji jest traktowane jako FP.
            
            W zdecydowanej większości przypadków okna FP są tak małe, 
            że zawierają najwyżej jedno człowieka. Dlatego przyjęto, że
            każde okno FP przekłada się na jednego człowieka FP.
            """
            if predictedId not in predictions.tp:
                if isOutsiderDetection(borderMargins, predictedBoxes[predictedId], img.shape):
                    predictedIdsToRemove.append(predictedId)
                else:
                    predictions.fp.append(predictedId)

        """
        Oznaczenie okien predykcji oraz oznaczeń ludzi, znajdujących się
        poza polem obserwacji. Nie są liczone do TP, FP, FN.
        
        Z oczywistych przyczyn detektory nie wykrywają TN, bo byłaby
        bardzo duża liczba detekcji tła.
        """
        for personId in personIdsToRemove:
            skippedBoxes.append(expectedPeople[personId].box)

        for predictedId in predictedIdsToRemove:
            skippedBoxes.append(predictedBoxes[predictedId].box)


        tp=len(predictions.tp)
        fp=len(predictions.fp)
        fn=len(predictions.fn)

        """
        Obliczenie ilość ramek, które zawierają predykcje TP. Są oznaczone na zielono.
        
        Od ogólnej liczby ramek odjęto ramki pominięte oraz ramki FP.
        """
        predictedBoxesWithoutSkipped = len(predictedBoxes) - len(predictedIdsToRemove) - fp

        results.append(Results(tp=tp, fp=fp, fn=fn))
        if tp == 0:
            boxesPerHumansResults.append(0)
        else:
            boxesPerHumansResults.append(predictedBoxesWithoutSkipped / float(tp))

        print(f"[{startFrame}/{imgNum:4d}/{endFrame:4d}] [{elapsedSec:.3f}s] [{model.getMode()}]: GT={len(expectedPeople):2d}, Det={predictedBoxesWithoutSkipped:2d}, TP={tp:2d}, FP={fp:2d}, FN={fn:2d}")
        if savePlots:
            plot(
                predictions=predictions,
                predictedBoxes=predictedBoxes,
                expectedPeople=expectedPeople,
                skippedBoxes=skippedBoxes,
                imgPath=DATASET_DIR / 'frames' / f"{imgNum}.png",
                downscaleFactor=downscaleFactor,
                borderMargins=borderMargins,
                saveDir=saveDir,
            )

    totalTp = sum(r.tp for r in results)
    totalFp = sum(r.fp for r in results)
    totalFn = sum(r.fn for r in results)

    print(f"Średnia precyzja: {totalTp / (totalTp + totalFp):.2%}")
    print(f"Średnia czułość: {totalTp / (totalTp + totalFn):.2%}")
    print(f"Stosunek liczby poprawnych ramek do TP: {np.mean(boxesPerHumansResults).astype(np.float32):.3f}")

"""
Rysuje ramki pola obserwacji, ramki okien predykcji, 
oznaczenia ludzi FN oraz ramki pominięte.

Generuje ładne rysunki, użyte w Rozdziale 6.

Przeskalowuje ramki predykcji z powrotem do rozdzielczości 1920x1080 pikseli,
przez co wyniki są nakładne na identyczne kolorowe klatki nagrania.
"""
def plot(predictions: Predictions, predictedBoxes: List[InferredBox], expectedPeople: List[InferredBox], skippedBoxes: List[Box], imgPath: Path, downscaleFactor: float, borderMargins: BorderMargins, saveDir: Path) -> None:
    render = Image.open(str(imgPath))
    draw = ImageDraw.Draw(render)
    imgShape = cv2.imread(str(imgPath)).shape

    for i in range(len(predictedBoxes)):
        box = predictedBoxes[i].box
        predictedBoxes[i].box = Box(
            x=box.x * downscaleFactor,
            y=box.y * downscaleFactor,
            w=box.w * downscaleFactor,
            h=box.h * downscaleFactor,
        )

    for i in range(len(expectedPeople)):
        box = expectedPeople[i].box
        expectedPeople[i].box = Box(
            x=box.x * downscaleFactor,
            y=box.y * downscaleFactor,
            w=box.w * downscaleFactor,
            h=box.h * downscaleFactor,
        )

    for tpPrediction in predictions.tp:
        box = predictedBoxes[tpPrediction].box
        draw.rectangle((box.x, box.y, box.x + box.w, box.y + box.h), outline="lime", width=4)

    for fpPrediction in predictions.fp:
        box = predictedBoxes[fpPrediction].box
        draw.rectangle((box.x, box.y, box.x + box.w, box.y + box.h), outline="yellow", width=4)

    for fnPrediction in predictions.fn:
        box = expectedPeople[fnPrediction].box
        draw.rectangle((box.x, box.y, box.x + box.w, box.y + box.h), outline="red", width=4)

    for skippedBox in skippedBoxes:
        scaledBox = Box(
            x=skippedBox.x * downscaleFactor,
            y=skippedBox.y * downscaleFactor,
            w=skippedBox.w * downscaleFactor,
            h=skippedBox.h * downscaleFactor,
        )
        draw.rectangle((scaledBox.x, scaledBox.y, scaledBox.x + scaledBox.w, scaledBox.y + scaledBox.h), outline="blue", width=4)


    draw.rectangle([imgShape[1] * borderMargins.leftPercent / 100, imgShape[0] * borderMargins.topPercent / 100, imgShape[1] * (1 - borderMargins.rightPercent / 100), imgShape[0] * (1 - borderMargins.bottomPercent / 100)], outline="black", width=4)

    render.save(f'{saveDir}/{imgPath.name}')

if __name__ == '__main__':
    main()