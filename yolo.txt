import cv2

net = cv2.dnn.readNet('dnn_model/yolov4-tiny.weights',
                      'dnn_model/yolov4-tiny.cfg')
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []

with open('dnn_model/classes.txt', 'r') as file_object:
    for class_name in file_object.readlines():
        classes.append(class_name.strip())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        object_name = classes[class_id]
        print(object_name, x, y, w, h)

        cv2.putText(frame, str(object_name), (x, y-10),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break