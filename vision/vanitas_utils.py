"""
Dataset and config utilities module.
Author: Ryan-Rhys Griffiths 2021
"""

import json
import os

import numpy as np
import skimage

from mrcnn import utils
from mrcnn.config import Config


class VanitasConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vanitas"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + key + cross + winged_lion

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class VanitasDataset(utils.Dataset):

    def load_vanitas(self, dataset_dir, subset):
        """Load a subset of the vanitas dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Add classes.

        keys = ['Skull', 'Hourglass', 'Globe', 'Coins', 'Butterfly', 'Flowers', 'Watch', 'Dice', 'Fruit', 'Violin',
                'Lute', 'Flute', 'Candle', 'Inkstand', 'Music', 'Bubble', 'Lamp', 'Book', 'Glass', 'Goblet', 'Vase',
                'Crown', 'Bishop\'s mitre', 'Crab', 'Lobster', 'Seashells', 'Chicken']

        for i in range(len(keys)):
            i += 1
            self.add_class('vanitas', i, keys[i])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        directory_list = os.listdir(dataset_dir)
        annotations = []
        for file in directory_list:
            if file[-5:] == '.json':
                annotations.append(json.load(open(os.path.join(dataset_dir, file))))

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            filename = a['asset']['name']
            image_path = os.path.join(dataset_dir, filename)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            if type(a['regions']) is dict:
                polygons = [r['boundingBox'] for r in a['regions'].values()]
                tags = [r['tags'] for r in a['regions'].values()]
            else:
                polygons = [r['boundingBox'] for r in a['regions']]
                tags = [r['tags'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            self.add_image(
                "vanitas",
                image_id=filename,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, labels=tags)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a semiotics dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "vanitas":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.rectangle((int(p['top']), int(p['left'])), ((int(p['top'])+int(p['height'])), (int(p['left'])+int(p['width']))))
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance.
        class_ids = np.zeros([mask.shape[-1]], dtype=np.int32)
        for i in range(len(info["labels"][0])):
            if info["labels"][0][i] == 'Key':
                class_ids[i] = 1
            if info["labels"][0][i] == 'cross':
                class_ids[i] = 2
            if info["labels"][0][i] == 'winged_lion':
                class_ids[i] = 3
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "semiotics":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
