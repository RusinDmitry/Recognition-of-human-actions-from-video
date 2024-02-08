## Распознавание действий человека по видео
Реализован сервис определяющий по короткому видео (до 10 сек) какое действие делает человек

___
## Набор данных
База данных включает в себя 24 класса в каждом классе порядка 1000 видео

## Структура

- В папке front расположен простой веб-интерфес который позволяет загрузить видео и получить ответ от системы какое действие происходит на видео
запуск веб-интерфейса осуществляется командой


    uvicorn main:app --reload

- В папке Scripts размещен notebook в котором было реализовано обучение модели, данный ноутбук рекомендуется запускать в google colab
файл train.py был задействован для настройки модели на сервере с GPU
Для старта необходимо убедиться в правильных путях к датасету и запустить файл **train.py**;
- В видеофайлах представлена демка работы системы в целом


[![Demo](https://img.youtube.com/vi/zfEcPTuWeMA/0.jpg)](https://www.youtube.com/watch?v=zfEcPTuWeMA)

## Требования
Все необходимые пакеты перечислены в файле **requirements.txt** в папке Scripts