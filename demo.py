import argparse
import cv2
import time

from mtcnn.mtcnn import MTCNN


def image_demo(img_input, img_output):
    """ mtcnn image demo """
    mtcnn = MTCNN('weights/pnet.h5', 'weights/rnet.h5', 'weights/onet.h5')

    img = cv2.imread(img_input)
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, landmarks, scores = mtcnn.detect(img_in)
    img = draw_faces(img, bboxes, landmarks, scores)
    cv2.imwrite(img_output, img)


def video_demo(video=None, output=None):
    """ mtcnn video/webcab demo """
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
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, landmarks, scores = mtcnn.detect(img_in)
        frames += 1
        img = draw_faces(img, bboxes, landmarks, scores)

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


def draw_faces(img, bboxes, landmarks, scores):
    """ draw bounding boxes and facial landmarks on the image """
    for box, landmark, score in zip(bboxes, landmarks, scores):
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
        for i in range(5):
            x = int(landmark[i])
            y = int(landmark[i + 5])
            img = cv2.circle(img, (x, y), 1, (0, 255, 0))
        img = cv2.putText(img, '{:.2f}'.format(score), (int(box[0]), int(box[1])),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    return img


parser = argparse.ArgumentParser(description='FaceID Demo')
parser.add_argument('--image_input', help='Input image')
parser.add_argument('--image_output', default='output.jpg', help='Output image')
parser.add_argument('--video', help='Video file')
parser.add_argument('--output', help='Output file')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.image_input:
        image_demo(args.image_input, args.image_output)
    else:
        video_demo(video=args.video, output=args.output)
