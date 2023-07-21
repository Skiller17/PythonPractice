import os


def find_duplicates(path: str) -> dict:
    duplicates = set()
    cars_dict = dict()

    abspath = os.path.abspath(path)

    for path_file in os.listdir(abspath):
        mark_path = os.path.join(abspath, path_file)
        models_cars_in_subdirectory = os.listdir(mark_path)

        for model_cars_in_subdirectory in models_cars_in_subdirectory:
            model_path = os.path.join(mark_path, model_cars_in_subdirectory)
            cars_in_subdirectory = os.listdir(model_path)

            for car_name_in_subdirectory in cars_in_subdirectory:
                car_path = os.path.join(model_path, car_name_in_subdirectory)

                if car_name_in_subdirectory not in cars_dict:
                    cars_dict[car_name_in_subdirectory] = [car_path]
                else:
                    cars_dict[car_name_in_subdirectory].append(car_path)
                    duplicates.add(car_name_in_subdirectory)

    duplicates_dict = {duplicates_key: cars_dict[duplicates_key] for duplicates_key in duplicates}
    return duplicates_dict
