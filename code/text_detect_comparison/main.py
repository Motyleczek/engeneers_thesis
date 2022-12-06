# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from pytesseract import pytesseract
from pytesseract import Output
from imutils.object_detection import non_max_suppression
import swtloc as swt
import numpy as np
import cv2
import argparse
import time

# line below might be sometimes needed, but shouldnt be if tesseract is installed properly
# pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract"


def pytesseract_text_detect(img_path: str, time_it: bool = False):
    """
    Function takes in path to an image, and displays the image with bounding boxes around detected text.
    Uses the pytesseract package together with opencv to visualise the results.
    Image will be shown untill '0' key presed.

    :param img_path: path to image
    :param return_data: determines if the function should return anything
    :return: None or (img, data), where img is the image with bounding boxes, and data is the read text
    """

    img = cv2.imread(img_path)
    image_data = pytesseract.image_to_data(img, output_type=Output.DICT)

    for i, word in enumerate(image_data["text"]):
        if word != "":
            x, y, w, h = image_data["left"][i], image_data["top"][i], image_data["width"][i], image_data["height"][i]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"Progress: {i}", "\r")
    print("finish")

    if not time_it:
        cv2.imshow("window", img)
        cv2.setWindowTitle("window", "Pytesseract results")
        cv2.waitKey(0)
        return None
    else:
        data = pytesseract.image_to_string(img)
        return img


# connectong bbs
# Sort bounding rects by x coordinate
def getXFromRect(item):
    return item[0]


def connect_bbs_myfun(bbs_list, x_thr=1, y_thr=1):
    # Array of initial bounding rects
    rects = []

    # Bool array indicating which initial bounding rect has
    # already been used
    rectsUsed = []

    # Just initialize bounding rects and set all bools to false
    for cnt in bbs_list:
        rects.append(cnt)
        rectsUsed.append(False)

    rects.sort(key = getXFromRect)

    # Array of accepted rects
    acceptedRects = []

    # Merge threshold for x coordinate distance
    xThr = x_thr
    yThr = y_thr

    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):
            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (candxMin <= currxMax + xThr):
                    if (candyMin <= curryMax + yThr) and (candyMin >= curryMin):
                        # Reset coordinates of current rect
                        currxMax = candxMax
                        curryMin = min(curryMin, candyMin)
                        curryMax = max(curryMax, candyMax)
                    # Merge candidate (bounding rect) is used
                        rectsUsed[subIdx] = True
                else:
                    break
             # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

    return acceptedRects


def swt_text_detect(img_path: str, time_it: bool=False):
    """
    Function takes in path to an image, and displays the image with bounding boxes around detected text.
    Uses the opencv package to both visualise and find boxes with stroge width transform.
    Image will be shown untill '0' key presed.

    :param img_path:
    :return: None
    """

    img = cv2.imread(img_path)
    result_bow, draw_bow, chain_bbs_bow = cv2.text.detectTextSWT(img, True)
    result_wob, draw_wob, chain_bbs_wob = cv2.text.detectTextSWT(img, False)

    img_with_bbs = img.copy()

    # połączyć tazem, usunac
    merged_bbs_bow = connect_bbs_myfun(chain_bbs_bow[0], 20, 20)
    merged_bbs_wob = connect_bbs_myfun(chain_bbs_wob[0], 20, 20)


    for i, elem in enumerate(merged_bbs_bow):
        start_point = elem[0], elem[1]
        end_point = elem[0]+elem[2], elem[1]+elem[3]
        cv2.rectangle(img_with_bbs, start_point, end_point, (255, 0, 0))

    for i, elem in enumerate(merged_bbs_wob):
        start_point = elem[0], elem[1]
        end_point = elem[0]+elem[2], elem[1]+elem[3]
        cv2.rectangle(img_with_bbs, start_point, end_point, (255, 0, 0))

    if not time_it:
        cv2.imshow("window", img_with_bbs)
        cv2.setWindowTitle("window", "SWT results")
        cv2.waitKey(0)
    else:
        return img_with_bbs


# potrzebne do east:
def decode_predictions(scores, geometry, args):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue
            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)


# https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
def east_text_detect(img_path: str, time_it: bool = False):
    """
    Function detects possible text regions and showcases them with the usage of EAST text detector from cv2.

    :param img_path: str, path to analysed image
    :return: None
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, default=img_path,
                    help="path to input image")
    ap.add_argument("-east", "--east", type=str, default="frozen_east_text_detection.pb",
                    help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.1,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320*2,
                    help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320*2,
                    help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    # load the input image and grab the image dimensions
    image = cv2.imread(args["image"])
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
    (rects, confidences) = decode_predictions(scores, geometry, args)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the frame
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # zaleznie od kompresji i resize image nie wykrywa nam wszystkiego - wydluza czas ale za to zwieksza dokladnosc
    if not time_it:
        cv2.imshow("window", orig)
        cv2.setWindowTitle("window", "EAST results")
        cv2.waitKey(0)
    else:
        return orig


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    img_paths = ["labels/label1.png",
                 "labels/label2.png",
                 "labels/label3.png",
                 "labels/label4.png",
                 "labels/label5.png",
                 "labels/label6.png",
                 "labels/label7.png",
                 "labels/label8.png",
                 "labels/label9.png",
                 "labels/label10.png"]

    times_tess = []
    times_swt = []
    times_east = []

    for i, img_path in enumerate(img_paths):
        # żeby tesseract dobrze działał musi być bliżej obrazu zdjęcue, inaczej nic nie wyjdzie
        # pytesseract_text_detect(img_path)
        # swt_text_detect(img_path)
        # east_text_detect(img_path)
        start = time.time()
        tess_img = pytesseract_text_detect(img_path, time_it=True)
        times_tess.append(time.time() - start)

        start = time.time()
        swt_img = swt_text_detect(img_path, time_it=True)
        times_swt.append(time.time() - start)

        start = time.time()
        east_img = east_text_detect(img_path, time_it=True)
        times_east.append(time.time() - start)

        cv2.imwrite(f"labels/tess_results/label{i+1}_tess.png", tess_img)
        cv2.imwrite(f"labels/swt_results/label{i+1}_swt.png", swt_img)
        cv2.imwrite(f"labels/east_results/label{i+1}_east.png", east_img)

    print(times_tess)
    print(times_swt)
    print(times_east)

