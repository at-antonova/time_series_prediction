## Обзор

Модуль time_series_prediction.predictor содержит классы:
- *TimeSeriesPredictor* - основной класс для прогнозирования временных рядов
- *NonPredModel* и его наследники - классы, обеспечивающие работу алгоритмов определения непрогнозируемых точек

Модуль time_series_prediction.experiment содержит функции для проведения некоторых вычислительных экспериментов

Модуль time_series_prediction.graphs содержит функции генерации основных графиков, которые предствлены в отчете


## Использование

Пример использования бибилиотеки приведен в файле example.py. Прогнозируется ряд Лоренца. Базовый алгоритм прогнозирования использует алгоритм определения непрогнозируемых точек ограничение на минимальный размер кластера с параметрами gamma=0.5, min_clusters=2, алгоритм вычисления единого прогнозного значения - центр наибольшего кластера DBSCAN. Алгоритм self-healing использует алгоритм странных паттернов с базовым алгоритмом ограничение на минимальный размер кластера с параметрами gamma=0.5, min_clusters=2; единое прогнозное значение - центр максимального кластера с весами, метод - фактор, factor=0.7.

------

## Overview

module time_series_prediction.predictor contains classes:
- *TimeSeriesPredictor* - main class for time series prediction workflow
- *NonPredModel* and its inherits - classes for algorithms of identifying non-predictable points

module time_series_prediction.experiment contains functions of perfomed numerical experiments

module time_series_prediction.graphs contains functions of main graphs generation that were provided in report


## Usage

Example of usage provided in example.py file. Predicting Lorenz time series. Base prediction uses algorithm of identifying non-predictable points of limit cluster size, gamma=0.5, min_clusters=2, algorithm of calculating unified prediction - mean of the largest cluster DBSCAN. Self-healing algorithm uses weird patterns with base non-predictable points algorithm of limit cluster size, gamma=0.5, min_clusters=2. Unified prediction - mean of largest weighted cluster DBSCAN, method - factor with factor=0.7.