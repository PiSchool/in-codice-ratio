import os
import struct
import tensorflow as tf
from six.moves import xrange # pylint: disable=redefined-builtin
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_small
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.image import OcrTest


@registry.register_problem
class OcrLatin(OcrTest):
    def preprocess_example(self, example, mode, _):
        # Resize from usual size ~1350x60 to 90x4 in this test.
        img = example["inputs"]
        example["inputs"] = tf.to_int64(
            tf.image.resize_images(img, [90, 4], tf.image.ResizeMethod.AREA))
        return example

    def generator(self, data_dir, tmp_dir, is_training):
        # In this test problem, we assume that the data is in tmp_dir/ocr/ in
        # files names 0.png, 0.txt, 1.png, 1.txt and so on until num_examples.
        character_vocab = text_encoder.ByteTextEncoder()
        ocr_dir = os.path.join(tmp_dir, "ocr/")
        num_examples = int(len(os.listdir(ocr_dir))/2)
        tf.logging.info("Looking for OCR data in %s." % ocr_dir)
        for i in xrange(num_examples):
            image_filepath = os.path.join(ocr_dir, "%d.png" % i)
            text_filepath = os.path.join(ocr_dir, "%d.txt" % i)
            with tf.gfile.Open(text_filepath, "r") as f:
                label = f.read()
            with tf.gfile.Open(image_filepath, "rb") as f:
                encoded_image_data = f.read()
            # In PNG files width and height are stored in these bytes.
            width, height = struct.unpack(">ii", encoded_image_data[16:24])
            encoded_label = character_vocab.encode(label.strip())
            yield {
              "image/encoded": [encoded_image_data],
              "image/format": ["png"],
              "image/class/label": encoded_label,
              "image/height": [height],
              "image/width": [width]
            }

# @registry.register_hparams
# def transformer_my_sketch():
#     """Modified transformer_small."""
#     hparams = transformer_small()
#     hparams.batch_size = 16
#     hparams.max_length = 784
#     hparams.clip_grad_norm = 5.
#     hparams.learning_rate_decay_scheme = "noam"
#     hparams.learning_rate = 0.1
#     hparams.initializer = "orthogonal"
#     hparams.sampling_method = "random"
#     hparams.learning_rate_warmup_steps = 10000
#     return hparams
