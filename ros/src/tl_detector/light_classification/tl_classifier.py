from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

OBJECT_DETECTION_MODEL_PATH = '../models/detection/frozen_inference_graph.pb'

class TLClassifier(object):
    def __init__(self):
        self.detection_graph = None
        self.sess = None
        self.image_tensor = None
        self.boxes = None
        self.scores = None
        self.classes = None
        self.num_detections = None
        self.__init_object_detection()

    def __init_object_detection(self):
        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(OBJECT_DETECTION_MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def __box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    def detect_traffic_light(self, image):
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
                box = self.__box_normal_to_pixel(boxes[tl_idx], dim)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h / (box_w + 0.01)
                # print (tl_idx, scores[tl_idx], ratio, box_w, box_h)
                if box_w >= 20 and box_h >= 20 and ratio >= 1.5:
                    best_box = box
                    best_score = scores[tl_idx]

        return best_box, best_score

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN
