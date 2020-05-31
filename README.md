# MTCNN Implementation in Tensorflow 2

This is an implementation of [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf) in Tensorflow 2.

Weights have been imported from [Facenet MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align).

## Install dependencies

`pip3 install -r requirements.txt`

## Run demo

Webcam demo
```
python3 demo.py
```

Video demo
```
python3 demo.py --video /path/to/video.mp4
```

Image demo
```
python3 demo.py --image_input /path/to/image.jpg --image_output /path/to/output.jpg
```

## Examples

[Example](img/ex1.jpg)
