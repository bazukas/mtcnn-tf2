import argparse
import cv2
import time

from mtcnn.mtcnn import MTCNN


def demo(video=None, output=None):
    mtcnn = MTCNN('weights/pnet.h5', 'weights/rnet.h5', 'weights/onet.h5')
    if video is None:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video)

    out = None
    if output is not None:
        fps = vid.get(cv2.CAP_PROP_FPS)
        _, img = vid.read()
        h, w, _ = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output, fourcc, fps, (w, h))

    s = time.time()
    frames = 0
    i = 0
    while True:
        _, img = vid.read()
        if img is None:
            break

        bboxes, landmarks = mtcnn.detect(img)
        frames += 1
        print(bboxes, landmarks)
        img = draw_faces(img, bboxes, landmarks)

        cv2.imshow('demo', img)
        if out is not None:
            out.write(img)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
        i += 1
    t = time.time() - s
    fps = frames / t
    print("FPS: ", fps)
    cv2.destroyAllWindows()


def draw_faces(img, bboxes, landmarks):
    h, w, _ = img.shape
    for box in bboxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
    return img


parser = argparse.ArgumentParser(description='FaceID Demo')
parser.add_argument('--video', help='Video file')
parser.add_argument('--output', help='Output file')


if __name__ == '__main__':
    args = parser.parse_args()
    demo(video=args.video, output=args.output)
