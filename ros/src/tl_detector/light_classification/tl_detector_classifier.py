import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class Detector(object):

    def __init__(self):
        # path_to_frozen = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
        path_to_frozen = '../models/detection/frozen_inference_graph.pb'

        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Helper function to convert normalized box coordinates to pixels

    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    def run_inference_for_image(self, image):
        best_box, best_score = None, None
        expanded_image = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():

            boxes, scores, classes, num_detections = self.sess.run([
                self.boxes, self.scores, self.classes, self.num_detections
            ], feed_dict={self.image_tensor: expanded_image})

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            cls = classes.tolist()

            tl_idxs = [idx for idx, v in enumerate(cls) if int(v) == 10]
            if len(tl_idxs) > 0 and scores[tl_idxs[0]] >= 0.2:
                tl_idx = tl_idxs[0]
                dim = image.shape[0:2]
                box = self.box_normal_to_pixel(boxes[tl_idx], dim)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h / (box_w + 0.01)
                # print (tl_idx, scores[tl_idx], ratio, box_w, box_h)
                if box_w >= 20 and box_h >= 20 and ratio >= 1.5:
                    best_box = box
                    best_score = scores[tl_idx]

        return best_box, best_score


if __name__ == '__main__':
    detector = Detector()
    np_image = cv2.imread('images/1_yes.jpg')
    print ('image shape: ', np_image.shape, type(np_image))
    box, score = detector.run_inference_for_image(np_image)
    if box is not None and score is not None:
        print ('traffic light detected with score {}'.format(score))
    else:
        print ('No traffic light')
