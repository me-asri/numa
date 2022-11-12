"""
Wrapper for HODA Persian digit dataset

The code license does NOT cover the included HODA dataset files.
Dataset files may only be used for non-commercial purposes only.

See: http://farsiocr.ir/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%AF%D8%A7%D8%AF%D9%87/%D9%85%D8%AC%D9%85%D9%88%D8%B9%D9%87-%D8%A7%D8%B1%D9%82%D8%A7%D9%85-%D8%AF%D8%B3%D8%AA%D9%86%D9%88%DB%8C%D8%B3-%D9%87%D8%AF%DB%8C/
"""

from typing import Tuple
from enum import IntEnum

import os
import logging
import struct
import ctypes as ct

import cv2
import numpy as np

import numa.dataset

Dataset = Tuple[np.ndarray, np.ndarray]


class ImageType(IntEnum):
    BINARY = 0,
    GREYSCALE = 1


class HodaHeader(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ('year', ct.c_uint16),
        ('month', ct.c_uint8),
        ('day', ct.c_uint8),
        ('width', ct.c_uint8),
        ('height', ct.c_uint8),
        ('record_count', ct.c_uint32),
        ('letter_count', ct.c_uint32 * 128),
        ('image_type', ct.c_uint8),
        ('comment', ct.c_char * 256)
    ]


def _reshape_image(image: np.ndarray, binary: bool = False, size: int = 28) -> np.ndarray:
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if aspect_ratio > 1:  # horizontal image
        new_width = size
        new_height = np.round(new_width / aspect_ratio).astype(int)
        pad_vertical = (size - new_height) / 2
        pad_top, pad_bottom = np.floor(pad_vertical).astype(
            int), np.ceil(pad_vertical).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect_ratio < 1:  # vertical image
        new_height = size
        new_width = np.round(new_height * aspect_ratio).astype(int)
        pad_horizontal = (size - new_width) / 2
        pad_left, pad_right = np.floor(pad_horizontal).astype(
            int), np.ceil(pad_horizontal).astype(int)
        pad_top, pad_bottom = 0, 0
    else:
        new_height, new_width = size, size
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0

    scaled = cv2.resize(image, (new_width, new_height),
                        interpolation=cv2.INTER_LINEAR)
    scaled = cv2.copyMakeBorder(
        scaled, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0] * 3)

    if binary:
        scaled = np.where(scaled >= 127, 255, 0).astype('uint8')

    return scaled


def _read_dataset(dataset_path: str, size: int) -> Dataset:
    logging.info(f'Reading dataset {dataset_path}')
    with open(dataset_path, 'rb') as dataset:
        header_len = ct.sizeof(HodaHeader)
        header = HodaHeader.from_buffer_copy(dataset.read(header_len))

        # Whether samples have different sizes or not
        uniform = header.width > 0 and header.height > 0

        # Skip 1024 bytes (actual header size is 1024 bytes)
        dataset.seek(1024, os.SEEK_SET)

        images = np.zeros((header.record_count, size, size), 'uint8')
        labels = np.zeros((header.record_count), 'uint8')

        logging.info(
            f'Release date: {header.year}/{header.month}/{header.day}')
        logging.info(f'Records: {header.record_count}')

        for index in range(0, header.record_count):
            magic, label = struct.unpack('=BB', dataset.read(2))
            if magic != 0xFF:
                # First byte of each sample must be 0xFF
                logging.warn(f'Sample #{index} might be corrupted')

            if not uniform:
                sample_width, sample_height = struct.unpack(
                    '=BB', dataset.read(2))
            else:
                sample_width, sample_height = header.width, header.height

            # Not sure what this field is used for...
            byte_count, = struct.unpack('=H', dataset.read(2))

            image = np.zeros((sample_height, sample_width), np.uint8)
            if header.image_type == ImageType.BINARY:
                for y in range(0, sample_height):
                    counter = 0
                    background = True

                    while counter < sample_width:
                        pixel_count, = struct.unpack('=B', dataset.read(1))

                        for x in range(0, pixel_count):
                            if background:
                                image[y][x + counter] = 0  # Black
                            else:
                                image[y][x + counter] = 255  # White

                        background = not background
                        counter += pixel_count

                images[index] = _reshape_image(image, True, size)
            else:
                # ! Untested !
                # As far as I'm aware HODA dataset does NOT contain any greyscale samples
                for y in range(0, sample_height):
                    for x in range(0, sample_width):
                        image[y][x], = struct.unpack('=B', dataset.read(1))

                images[index] = _reshape_image(image, False, size)

            labels[index] = label

        return (images, labels)


def load_dataset(size: int = 28) -> Tuple[Dataset, Dataset]:
    """
    Load HODA dataset

    * Returns a tuple containing training dataset and testing dataset
    """

    dataset_dir = os.path.dirname(numa.dataset.__file__)

    training_path = os.path.join(dataset_dir, 'hoda', 'Train.cdb')
    testing_path = os.path.join(dataset_dir, 'hoda', 'Test.cdb')

    return (_read_dataset(training_path, size), _read_dataset(testing_path, size))
