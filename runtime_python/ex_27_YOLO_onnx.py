#https://github.com/zhouzq-thu/YOLOv8_ONNX/blob/main/YOLOv8_ONNX/utils.py
from ultralytics import YOLO
import onnxruntime as ort
import numpy
import cv2
# ----------------------------------------------------------------------------------------------------------------------
import tools_draw_numpy
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './'
# ----------------------------------------------------------------------------------------------------------------------
colors80 = tools_draw_numpy.get_colors(80,colormap='tab10',shuffle=True)
# ----------------------------------------------------------------------------------------------------------------------
def nms(rects, confidences, threshold):
    if len(rects) == 0:
        return []

    rects = numpy.array(rects, dtype=float)
    confidences = numpy.array(confidences, dtype=float)

    pick = []
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = x1 + rects[:, 2]
    y2 = y1 + rects[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = numpy.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = numpy.maximum(x1[i], x1[idxs[:last]])
        yy1 = numpy.maximum(y1[i], y1[idxs[:last]])
        xx2 = numpy.minimum(x2[i], x2[idxs[:last]])
        yy2 = numpy.minimum(y2[i], y2[idxs[:last]])
        w = numpy.maximum(0, xx2 - xx1 + 1)
        h = numpy.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = numpy.delete(idxs, numpy.concatenate(([last], numpy.where(overlap > threshold)[0])))

    return pick
# ----------------------------------------------------------------------------------------------------------------------
def rescale_boxes(boxes, input_shape, image_shape):
    # input_shape = numpy.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    # boxes = numpy.divide(boxes, input_shape, dtype=numpy.float32)
    # boxes *= numpy.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
    boxes[:, 0] = boxes[:, 0] / (image_shape[1] / input_shape[1])
    boxes[:, 1] = boxes[:, 1] / (image_shape[0] / input_shape[0])
    boxes[:, 2] = boxes[:, 2] / (image_shape[1] / input_shape[1])
    boxes[:, 3] = boxes[:, 3] / (image_shape[0] / input_shape[0])
    return boxes
# ----------------------------------------------------------------------------------------------------------------------
def xywh2xyxy(x):
    y = numpy.copy(x)
    w,h = x[...,2], x[...,3]
    y[..., 0] = x[..., 0] - w/2
    y[..., 1] = x[..., 1] - h/2
    y[..., 2] = x[..., 0] + w/2
    y[..., 3] = x[..., 1] + h/2
    return y
# ----------------------------------------------------------------------------------------------------------------------
def extract_boxes(predictions,input_height, input_width,img_height, img_width):
    boxes = predictions[:, :4]
    boxes = rescale_boxes(boxes,(input_height, input_width),(img_height, img_width))
    boxes = xywh2xyxy(boxes)
    boxes[:, 0] = numpy.clip(boxes[:, 0], 0, img_width)
    boxes[:, 1] = numpy.clip(boxes[:, 1], 0, img_height)
    boxes[:, 2] = numpy.clip(boxes[:, 2], 0, img_width)
    boxes[:, 3] = numpy.clip(boxes[:, 3], 0, img_height)

    return boxes
# ----------------------------------------------------------------------------------------------------------------------
def process_output(output,input_height, input_width,img_height, img_width,conf_threshold = 0.3, iou_threshold = 0.5):

    predictions = numpy.squeeze(output[0]).T
    confidences = numpy.max(predictions[:, 4:], axis=1)
    predictions = predictions[confidences > conf_threshold, :]
    confidences = confidences[confidences > conf_threshold]

    if len(confidences) == 0:
        return [], [], []

    class_ids = numpy.argmax(predictions[:, 4:], axis=1)
    rects = extract_boxes(predictions,input_height, input_width,img_height, img_width)
    indices = nms(rects, confidences, iou_threshold)
    return rects[indices], confidences[indices], class_ids[indices]
# ----------------------------------------------------------------------------------------------------------------------
def ex_onnx_detect(filename_onnx,filename_image):

    session = ort.InferenceSession(filename_onnx)
    image = cv2.imread(filename_image)
    input_height, input_width = image.shape[:2]
    img_height, img_width,  = 640, 640

    image_preproc = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (img_width,img_height), interpolation=cv2.INTER_LINEAR)
    image_preproc = numpy.expand_dims(image_preproc, axis=0).astype('float32') / 255.
    output = session.run(None, {session.get_inputs()[0].name: numpy.transpose(image_preproc, [0, 3, 1, 2])})
    rects, confidences, class_ids = process_output(output,input_height, input_width,img_height, img_width)
    #dct_class_names = {0: 'cube', 1: 'pyramid'}

    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


    dct_class_names = {i: name for i, name in enumerate(class_names)}


    labels = [dct_class_names[class_id] + ' ' + '%.0f%%' % (100 * conf) for class_id, conf in zip(class_ids, confidences)]
    colors = [colors80[int(i)] for i in class_ids]
    image_res = tools_draw_numpy.draw_rects(image, rects.reshape((-1,2,2)), colors, labels=labels, w=2, alpha_transp=0.8)
    cv2.imwrite(folder_out + 'res_onnx.jpg', image_res)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_happy_path():
    model_detect = YOLO('yolov8n.pt')
    model_detect.predict('./data/ex_detector/dog_bicycle.jpg')

    model_detect = YOLO('yolov8.yaml')
    model_detect.predict('./data/ex_detector/dog_bicycle.jpg')

    #model_detect.train(data='coco128.yaml', epochs=1, imgsz=640)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_process_grayscaled():
    # model_detect = YOLO('yolov8n-custom.yaml')
    # model_detect.predict(cv2.imread('./images/dog_bicycle.jpg', 0),ch=1)
    # model_detect.predict('./images/dog_bicycle_gray.png',ch=1)
    # model_detect.train(data='coco128.yaml', epochs=1, imgsz=640,ch=1)

    model_detect_trained_epoch1 = YOLO('best.pt')
    model_detect_trained_epoch1.predict(cv2.imread('./images/dog_bicycle.jpg', 0), ch=1)
    model_detect_trained_epoch1.predict('./images/dog_bicycle_gray.png', ch=1)

    return
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    ex_onnx_detect('yolov8n.onnx', './dog_bicycle.jpg')
    #ex_happy_path()
    #ex_process_grayscaled()





