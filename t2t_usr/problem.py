import os
import struct
import tensorflow as tf
from six.moves import xrange # pylint: disable=redefined-builtin
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models import transformer_vae
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.image import OcrTest


@registry.register_model
class TransformerSketchOcr(transformer.Transformer):
    """Transformer with strided convolutions."""
    def encode(self, inputs, target_space, hparams):
        """Add two layers strided convolutions ontop of encode."""
        inputs = common_layers.conv_block(
            inputs,
            hparams.hidden_size, [((1, 1), (3, 3))],
            first_relu=False,
            padding="SAME",
            force2d=True,
            name="small_image_conv")

        hparams.num_compress_steps = 4
        compressed_inputs = transformer_vae.compress(
            inputs,
            c=None,
            is_2d=True,
            hparams=hparams,
            name="convolutions")

        return super(TransformerSketchOcr, self).encode(
            compressed_inputs, target_space, hparams)


@registry.register_problem
class OcrLatin(OcrTest):
    def example_reading_spec(self, label_repr=None):
        if label_repr is None:
            label_repr = ("image/class/label", tf.VarLenFeature(tf.int64) )

        data_fields = {
            "image/encoded": tf.FixedLenFeature((), tf.string),
            "image/format": tf.FixedLenFeature((), tf.string),
        }
        label_key, label_type = label_repr  # pylint: disable=unpacking-non-sequence
        data_fields[label_key] = label_type

        data_items_to_decoders = {
            "inputs":
                tf.contrib.slim.tfexample_decoder.Image(
                    image_key="image/encoded",
                    format_key="image/format",
                    channels=3),
            "targets":
                tf.contrib.slim.tfexample_decoder.Tensor(label_key),
        }
        return data_fields, data_items_to_decoders

    def preprocess_example(self, example, mode, _):
        # Resize from usual size ~1350x60 to 90x4 in this test.
        img = example["inputs"]
        example["inputs"] = tf.to_int64(
            tf.image.resize_images(img, [320, 16], tf.image.ResizeMethod.AREA))
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
