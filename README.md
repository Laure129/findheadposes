# findheadposes
Программа, предназначением которой является собрание фотографий человека с разных ракурсов.
Полезна для людей работающий как с 3D, так и с 2D графикой.
Интерфейс сделан на Django.
При вводе имени, программа с помощью Bing Image API скачивает изображения, после с помощью двух нейросетей(вычисление
положения головы в горизонтальной и вертикальной плоскости) и OpenCV определяет поворот и наклон головы(если удастся
обнаружить лицо), после загружает данные в бд. Сортировка сейчас идет по повороту головы в горизонтальной плоскости. Если лицо обнаружено не было, изображение будет находиться в конце альбома.
