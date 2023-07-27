import os
from typing import Dict

import cv2
import onnx
from time import sleep
from tqdm import tqdm

from onnx_code import BrandModelClassifier
from work_with_predict_and_real_data import compare_predict_and_marking_data, save_image_with_predict_mark, \
    make_no_match_dict, build_histogram


def create_model(model_path: str) -> BrandModelClassifier:
    onnx_model = onnx.load(model_path)
    classifier = BrandModelClassifier(onnx_model)
    return classifier


def check_datasets(list_of_datasets: list, output_path: str, model_path: str) -> Dict[str, int]:
    no_match_dict = dict()
    classifier = create_model(model_path)
    for dataset in tqdm(list_of_datasets, desc='Checked datasets', colour='green', ncols=100):
        sleep(1)
        check_images(dataset, output_path, no_match_dict, classifier)

    build_histogram(no_match_dict)
    return no_match_dict


def check_images(path: str, output_path: str, no_match_dict: Dict[str, int], classifier: BrandModelClassifier):
    abspath = os.path.abspath(path)
    for real_brand in tqdm(os.listdir(abspath), desc='Checked brands', colour='blue', ncols=100):
        sleep(1)
        brand_path = os.path.join(abspath, real_brand)
        real_models = os.listdir(brand_path)

        for real_model in tqdm(real_models, desc='Checked models', colour='yellow', ncols=100):
            sleep(1)
            model_path = os.path.join(brand_path, real_model)
            file_names = os.listdir(model_path)
            output_dir_path = os.path.join(output_path, os.path.basename(path), real_brand, real_model)

            for file_name in tqdm(file_names, desc='Checked files', ncols=100):
                sleep(1)
                file_path = os.path.join(model_path, file_name)
                img = cv2.imread(file_path)
                predictable_car = classifier.predict(img)

                if compare_predict_and_marking_data(predictable_car, real_brand, real_model) is False:
                    save_image_with_predict_mark(predictable_car, output_dir_path, file_path,
                                                 os.path.basename(file_path))
                    make_no_match_dict(predictable_car, real_brand, real_model, no_match_dict)
