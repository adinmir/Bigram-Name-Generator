# Bigram-Name-Generator
Для работы с этой программой, следуйте инструкциям ниже:

Загрузите файл names.txt из репозитория https://raw.githubusercontent.com/adinmir/Bigram-Name-Generator/main/path/to/names.txt и сохраните его на вашем компьютере.

Установите необходимые зависимости:

Убедитесь, что у вас установлен модуль random, который является частью стандартной библиотеки Python.
Установите модуль collections, который является частью стандартной библиотеки Python.
В программе есть функции, которые можно использовать:

read_names(file_path): Эта функция принимает путь к файлу names.txt и возвращает список имен, считанных из файла.
build_bigram_model(names): Эта функция принимает список имен и строит модель биграмм на основе этих имен. Она возвращает модель биграмм.
generate_name(bigram_model): Эта функция принимает модель биграмм и генерирует новое имя на основе этой модели. Она возвращает сгенерированное имя.
print_bigram_probabilities(bigram_model): Эта функция принимает модель биграмм и печатает вероятности для каждой биграммы.
Для использования программы:

Загрузите файл names.txt с помощью функции read_names(file_path). Укажите путь к файлу names.txt, который вы загрузили с репозитория, в качестве аргумента функции read_names(). Результатом будет список имен.

С помощью функции build_bigram_model(names) постройте модель биграмм на основе списка имен. Передайте список имен, полученный на предыдущем шаге, в качестве аргумента функции build_bigram_model(). Результатом будет модель биграмм.

Генерируйте новые имена с помощью функции generate_name(bigram_model). Передайте модель биграмм, полученную на предыдущем шаге, в качестве аргумента функции generate_name(). Результатом будет сгенерированное имя.

Печатайте вероятности для каждой биграммы с помощью функции print_bigram_probabilities(bigram_model). Передайте модель биграмм, полученную на предыдущем шаге, в качестве аргумента функции print_bigram_probabilities().
Обратите внимание, что перед использованием программы вам необходимо сохранить файл names.txt и установить необходимые зависимости.
