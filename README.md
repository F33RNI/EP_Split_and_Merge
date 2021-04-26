# Инженерный проект за 5 (зимний) семестр
## Python. Полностью свободное использование
### Вариант 2.2 (Split &amp; Merge)
### Задание:

```
2. Сегментация на основе градаций серого
2.2. Метод разделения и объединения для символов текста Применить алгоритм Split&Merge для сегментации и применить классификатор MLP для определения надписей на изображениях. Задано: источник данных – наборы изображений Порядок обработки: 
• выполнить покадровое считывание методами OpenCV или AForgeNET и обесцвечивание для видеофайла 640х480
• применить операцию расщепления изображения за счет сужения дифференциации по яркости и контролем плотности и размера объектов (областей). 
• применить операцию слияния для смежных областей для определения кандидатов на распознавание. 
• обучить классификатор на набор печатных символов 
• реализовать вырезку кандидатов на распознавание (строк) и разбиения на фрагменты по пропорции символа (высота к ширине) 
• реализовать распознавание выдачу списка видов и положений знаков (с объединением в текст при построчном сканировании изображения)
```
