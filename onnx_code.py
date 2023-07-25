import cv2
import numpy as np
import os
from typing import Any

import onnx
import onnxruntime
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime import InferenceSession


def zero_padding(
    image: np.ndarray,
    input_size: tuple[int, int],
    stick_bottom: bool = False,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """We want to save aspect ratio of image, so we generate
    black empty image with requried aspect ratio (RAR), and then
    copy the original image to the new image. Position of image
    depends on specified flags.
    RAR is calculated from input_size.
    Function accepts both RGB and grayscale images.

    Parameters
    ----------
    image : np.ndarray
        Image to pad
    input_size : Tuple[int, int]
        Image will have specified size (W, H)
    stick_bottom : bool
        Is image should be placed at bottom instead of top
    center : bool
        Is image should be placed at center (overrides 'stick_bottom')

    Returns
    -------
    np.ndarray
        Resized and padded image
    np.ndarray
        Resized image
    Tuple[int, int]
        Shifts from left and top for postprocessing
    """
    W, H = input_size
    img_h, img_w, *ch = image.shape

    aim_ratio = H / W  # RAR
    current_ratio = img_h / img_w

    # Current image is taller than needed
    if current_ratio > aim_ratio:
        # scale width the same ratio, as height should be scaled:
        # new_w = w_i * (H / h_i) => H * (w_i / h_i) => H / (h_i / w_i) => H / current_ratio
        new_w = int(H / current_ratio)
        new_h = H
    # It is wider than needed
    else:
        # scale height the same ratio, as width should be scaled:
        # new_h = h_i * (W / w_i) => W * (h_i / w_i) => W * current_ratio
        new_w = W
        new_h = int(W * current_ratio)
    try:
        resized_image = cv2.resize(image, (new_w, new_h))
    except cv2.error:  # image or new size has zero in shape
        resized_image = np.zeros((1, 1), np.uint8)
    padded_image = np.zeros((H, W, *ch), np.uint8)

    shift_from_left = 0
    shift_from_top = 0
    if center:
        shift_from_left = int((W - new_w) / 2)
        shift_from_top = int((H - new_h) / 2)
        padded_image[
            shift_from_top : shift_from_top + new_h,
            shift_from_left : shift_from_left + new_w,
        ] = resized_image
    else:
        if stick_bottom:
            padded_image[-new_h:, 0:new_w] = resized_image
        else:
            padded_image[0:new_h, 0:new_w] = resized_image
    return padded_image, resized_image, (shift_from_left, shift_from_top)



def create_inference_session_from_model(
    model: ONNXModel,
) -> tuple[Any, list[str], list[str], list[tuple[int, int]]]:
    """Creates onnxruntime Inference Session from ONNX model

    Parameters
    ----------
    model : ONNXModel
        Model from which inference session will be created

    Returns
    -------
    ORTInferenceSession
        Created inference session
    list
        List of input names
    list
        List of output names
    list
        List of input sizes
    """
    tmp_path = ".tmp_model_file.onnx"
    onnx.save(model, tmp_path)
    session = onnxruntime.InferenceSession(tmp_path)
    os.remove(tmp_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    input_sizes = [(inp.shape[3], inp.shape[2]) for inp in session.get_inputs()]
    return session, input_names, output_names, input_sizes


class BrandModelClassifier:
    def __init__(self, model: ONNXModel):
        """
        Parameters
        ---------
        model : ONNXModel
            ONNX model
        """
        session_data = create_inference_session_from_model(model)
        self.input_image = None
        (
            self.session,
            self.input_layers,
            self.output_layers,
            self.input_size,
        ) = session_data
        self.input_size = self.input_size[0]  # only one input
        runner_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(runner_dir, "utils/brands.txt"), "r") as f:
            self.brands = [x for x in f.read().split("\n") if len(x) > 0]
        with open(os.path.join(runner_dir, "utils/brand_models.txt"), "r") as f:
            self.brand_models = [x for x in f.read().split("\n") if len(x) > 0]
        self.num_to_brand = {i: bm for i, bm in enumerate(self.brands)}
        self.num_to_brand_model = {i: bm for i, bm in enumerate(self.brand_models)}

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Input image

        Returns
        -------
        x : np.ndarray
            Preprocessed image
        """
        if len(x.shape) == 3:
            if x.shape[2] != 1:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x, resized_x, _ = zero_padding(x, self.input_size, center=True)
        self.input_image = x.copy()
        x = x.astype(np.float32) / 255.0  # from [0 - 255] to [0.0 - 1.0]
        x = x.reshape(
            (
                1,
                1,
            )
            + (x.shape)
        )  # from HW to NCHW
        return x

    def postprocess(
        self,
        brand_out: np.ndarray,
        brand_model_out: np.ndarray,
    ) -> tuple[tuple[str, float], tuple[str, float]]:
        """
        Parameters
        ----------
        brand_out : np.ndarray
            Probabilities for each brand
        brand_model_out : np.ndarray
            Probabilities for each brand-model pairs

        Returns
        -------
        Tuple[str, float]
            Brand and it's prob
        Tuple[str, float]
            Brand-model pair and it's prob
        """
        brand_out = brand_out.reshape(-1)
        # brands_prob = {self.num_to_brand[i]: prob for i, prob in enumerate(brand_out)}
        # print(brands_prob)
        brand_argmax = np.argmax(brand_out)
        brand_text = self.num_to_brand[brand_argmax]
        brand_prob = brand_out[brand_argmax]

        brand_model_out = brand_model_out.reshape(-1)
        brand_model_argmax = np.argmax(brand_model_out)
        brand_model_text = self.num_to_brand_model[brand_model_argmax]
        brand_model_prob = brand_model_out[brand_model_argmax]
        return ((brand_text, brand_prob), (brand_model_text, brand_model_prob))

    def predict(self, x: np.ndarray) -> tuple[tuple[str, float], tuple[str, float]]:
        """
        Parameters
        ----------
        x : np.ndarray
            Input image

        Returns
        -------
        Tuple[str, float]
            Brand and it's prob
        Tuple[str, float]
            Brand-model pair and it's prob
        """
        x_prepared = self.preprocess(x)
        inputs = {self.input_layers[0]: x_prepared}
        brand_out, brand_model_out = self.session.run(self.output_layers, inputs)
        return self.postprocess(brand_out[0], brand_model_out[0])
