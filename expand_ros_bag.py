import numpy as np
import rosbag
import yaml
import cv2
from tqdm import tqdm
import os


class RosbagImageExtractor:

    def __init__(self, rosbag_file, topic: str, compressed_image: bool, output_dir: str):
        self.rosbag_file = rosbag_file
        self.topic = topic
        self.compressed_image = compressed_image # True if the image is compressed
        self.output_dir = output_dir
        print("Loading rosbag " + self.rosbag_file + "...")
        self.bag = rosbag.Bag(self.rosbag_file)
        print("...done.")

        # Print information and check rosbag -----
        self.num_samples = 0
        info_dict = yaml.load(self.bag._get_yaml_info(), yaml.Loader)
        print("Duration of the bag: " + str(info_dict["duration"]))
        for topic_messages in info_dict["topics"]:
            if topic_messages["topic"] == self.topic:
                self.num_samples = topic_messages["messages"]
        if self.num_samples > 0:
            print("Number of messages for topic " + self.topic + ": " + str(self.num_samples))
        else:
            raise Exception("Topic " + self.topic + " is not present in the given rosbag (" + self.rosbag_file + "). \n "
                            + "Available topics are: " + str([topic['topic'] for topic in info_dict["topics"]]))
        # -----------------------------------------

    def save_images(self) -> None:
        # if output dir does not exist create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print("Saving messages of topic " + self.topic + " to " + self.output_dir + "...")
        try:
            for index, (topic, msg, t) in tqdm(enumerate(self.bag.read_messages(topics=[self.topic]))):

                # Uncomment for debugging
                if index > 9999:
                    break
                if not self.compressed_image:
                    im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                    cv_image = cv2.cvtColor(im, cv2.IMREAD_COLOR)
                else:
                    im = np.fromstring(msg.data, np.uint8)
                    cv_image = cv2.imdecode(im, cv2.IMREAD_COLOR)

                timestr = "%.6f" % msg.header.stamp.to_sec()
                image_name = str(self.output_dir) + timestr[0:14] + ".png"
                cv2.imwrite(image_name, cv_image)

            self.bag.close()
        except Exception as e:
            # erase the output dir if something went wrong
            print("Something went wrong while extracting images from the rosbag. \n"
                  + "The output directory will be deleted.")
            print(e)
            os.rmdir(self.output_dir)
            raise e


if __name__ == "__main__" :

    bag_extractor = RosbagImageExtractor(
        rosbag_file="/home/rpellerito/trackformer/data/EXCAV_rosbag/packed/heap_camera_2022-04-22-14-48-12.bag",
        topic = "/camMainView/image_raw/compressed",  
        compressed_image = True,
        output_dir = "/home/rpellerito/trackformer/data/EXCAV/test/fake")

    bag_extractor.save_images()

    os.getcwd()
    collection = "/home/rpellerito/trackformer/data/EXCAV/test/"

    for i, filename in enumerate(sorted(os.listdir(collection))):
        print("\n", filename)
        
        os.rename("/home/rpellerito/trackformer/data/EXCAV/test/" + filename,
        "/home/rpellerito/trackformer/data/EXCAV/test_rename/" + (6-len(str(i)))*str(0)+ str(i) + ".png")
