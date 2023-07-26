import cv2
import numpy as np

def load_yolo():
    """ Load the model """
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("yolov3.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def get_output_layers(net):
    """ Get the output layers of the model """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def detect_objects(image, net, output_layers):
    """ Detect objects in the image """
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width = image.shape[:2]
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return class_ids, confidences, boxes

def draw_predictions(image, class_ids, confidences, boxes, classes):
    for i in range(len(class_ids)):
        class_id = class_ids[i]
        confidence = confidences[i]
        box = boxes[i]

        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
        cv2.putText(image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    net, classes = load_yolo()

    # Replace 'your_image_path.jpg' with the path to your image file
    image = cv2.imread('your_image_path.jpg')
    class_ids, confidences, boxes = detect_objects(image, net, get_output_layers(net))
    draw_predictions(image, class_ids, confidences, boxes, classes)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()