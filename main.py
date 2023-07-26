import os
import shutil

import cv2
import matplotlib.pyplot as plt
import onnx
import seaborn as sns

from onnx_code import BrandModelClassifier


def create_model(model_path: str) -> BrandModelClassifier:
    onnx_model = onnx.load(model_path)
    classifier = BrandModelClassifier(onnx_model)
    return classifier


def check_datasets(list_of_datasets: list, output_path: str, model_path: str):
    no_match_dict = dict()
    classifier = create_model(model_path)
    for dataset in list_of_datasets:
        check_images(dataset, output_path, no_match_dict, classifier)

    build_histogram(no_match_dict)


def check_images(path: str, output_path: str, no_match_dict: dict[str, int], classifier: BrandModelClassifier):
    abspath = os.path.abspath(path)
    for dataset in os.listdir(abspath):
        dataset_path = os.path.join(abspath, dataset)
        real_brands = os.listdir(dataset_path)

        for real_brand in real_brands:
            brand_path = os.path.join(dataset_path, real_brand)
            real_models = os.listdir(brand_path)

            for real_model in real_models:
                model_path = os.path.join(brand_path, real_model)
                file_names = os.listdir(model_path)
                output_dir_path = os.path.join(output_path, os.path.basename(path), real_brand, real_model)

                for file_name in file_names:
                    file_path = os.path.join(model_path, file_name)
                    img = cv2.imread(file_path)
                    predictable_car = classifier.predict(img)

                    if compare_predict_and_marking_data(predictable_car, real_brand, real_model) is False:
                        save_image_with_predict_mark(predictable_car, output_dir_path, file_path,
                                                     os.path.basename(file_path))
                        make_no_match_dict(predictable_car, real_brand, real_model, no_match_dict)


def compare_predict_and_marking_data(
        predict_data: tuple[tuple[str, float], tuple[str, float]],
        real_brand: str,
        real_model: str
) -> bool:
    if predict_data[0][0] == real_brand:

        if predict_data[1][0] == real_model:
            return True
        else:
            return False
    else:
        return False


def save_image_with_predict_mark(predict_data: tuple[tuple[str, float], tuple[str, float]], output_path: str,
                                 file_path: str,
                                 file_name: str):
    os.makedirs(output_path, exist_ok=True)
    output_name = f'{predict_data[1][0]}_{file_name}'
    file_output_path = os.path.join(output_path, output_name)
    shutil.copy(file_path, file_output_path)


def make_no_match_dict(predict_data: tuple[tuple[str, float], tuple[str, float]], real_brand: str, real_model: str,
                       no_match_dict: dict[str, int]):
    real_brand_with_model = real_brand + ' ' + real_model

    str_for_no_match_dict = real_brand_with_model + ' - ' + predict_data[1][0]

    if str_for_no_match_dict in no_match_dict:
        no_match_dict[str_for_no_match_dict] += 1
    else:
        no_match_dict[str_for_no_match_dict] = 1


def build_histogram(no_match_dict: dict[str, int]):
    keys = list()
    values = list()
    for key, value in no_match_dict.items():
        keys.append(key)
        values.append(value)

    plt.figure(figsize=(17, 5))
    sns.barplot(x=keys, y=values)
    plt.show()


list_of_datasets = list()
list_of_datasets.append('/home/azhok/Dataset1')
list_of_datasets.append('/home/azhok/Dataset2')
check_datasets(list_of_datasets, '/home/azhok/Output', 'model.onnx')
