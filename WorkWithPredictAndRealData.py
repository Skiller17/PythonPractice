import os
import shutil

import matplotlib.pyplot as plt
import seaborn as sns


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