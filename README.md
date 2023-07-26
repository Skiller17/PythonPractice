Вторая задача на практике <br/>
Условие: <br/>

Создать функцию, которая получает на вход список путей до папок датасетов марок и путь до output_path и выполняет следущее:
1) Пробегает по каждому изображению в датасетах, открывает его с помощью opencv
2) К открытому изображению применяет функцию  predict (с начала создай ее как функцию заглушку, которая принимает на вход np.ndarray а на выходе возвращает Tuple[str, str, float] (марка и модель, уверенность предсказания))
3) Полученные функцией predict марка и модель сравниваются с именами папок в которых лежит изображение, то есть происзодит сравнение разметки и предсказаний нейронной сети
4) При несовпадении, изображение сохраняется в новой папке output_path/brand/model и к названию изображения добавляется постфикс {predicted_brand}_{predicted_model}
5) Реализовать настоящую функцию predict через onnx
6) Построить гистограмму распределения наиболее частых несовпадений
7) Добавить прогресс бар, для них успользуется библиотека tqdm. Можно сделать несколько вложенных програссбаров, по датасетам, маркам, моделям и картинкам