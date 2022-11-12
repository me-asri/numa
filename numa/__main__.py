from numa.dataset import hoda

import logging

logging.basicConfig(level=logging.INFO)

(training_images, training_labels), (testing_images,
                                     testing_labels) = hoda.load_dataset(30)
