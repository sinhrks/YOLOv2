import sys
import click

import cv2
from yolov2_darknet_predict import CocoPredictor


@click.command()
@click.option('-i', default=None)
@click.option('-w', default=640)
@click.option('-h', default=480)
def cmd(i, w, h):
    # 画像を変換
    if i is not None:
        predict_image(i)
        sys.exit(0)
    else:
        predict_camera(w, h)
        sys.exit(0)


def _decorate(img, class_id, label, probs, conf, objectness, box):
    left, top = box.int_left_top()
    cv2.rectangle(img, box.int_left_top(), box.int_right_bottom(),
                  (0, 255, 0), 5)
    text = '%s(%2d%%)' % (label, probs.max() * conf * 100)
    cv2.putText(img, text, (left, top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def predict_image(img):
    # read image
    print("loading image...")
    orig_img = cv2.imread(img)

    predictor = CocoPredictor()
    nms_results = predictor(orig_img)

    # draw result
    for result in nms_results:
        _decorate(orig_img, **result)

    print("save results to yolov2_result.jpg")
    cv2.imwrite("yolov2_result.jpg", orig_img)
    cv2.imshow("w", orig_img)
    cv2.waitKey()


def predict_camera(w, h):
    cap = cv2.VideoCapture(0)

    ret = cap.set(3, w)
    ret = cap.set(4, h)

    coco_predictor = CocoPredictor()

    while(True):
        ret, orig_img = cap.read()
        nms_results = coco_predictor(orig_img)

        # draw result
        for result in nms_results:
            _decorate(orig_img, **result)

        cv2.imshow("w", orig_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    cmd()
