import json

import numpy as np
import pandas as pd


class BitcoinDataProcessor:
    def __init__(self, file_path):
        """
        Инициализация обработчика данных Bitcoin.

        :param file_path: путь к файлу JSON с историческими данными о Bitcoin.
        """
        self.file_path = file_path
        self.data = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """
        Загружает и предобрабатывает данные Bitcoin из JSON файла.

        Входные данные должны быть в формате JSON, где каждая запись содержит:
        - date: дата и время в формате Unix timestamp (миллисекунды).
        - rate: стоимость Bitcoin в данный момент времени.
        - volume: объем торгов Bitcoin.
        - cap: рыночная капитализация Bitcoin.

        Возвращаемые данные - это pandas DataFrame с индексом по дате, содержащий:
        - rate: стоимость Bitcoin.
        - volume: объем торгов.
        - cap: рыночная капитализация.
        - rate_change: процентное изменение стоимости Bitcoin.
        - volume_change: процентное изменение объема торгов.
        - volatility: волатильность стоимости Bitcoin, рассчитанная на скользящем окне.

        :return: обработанный pandas DataFrame.
        """
        with open(self.file_path, 'r') as file:
            data = json.load(file)

        df = pd.DataFrame(data)

        # Преобразование Unix времени в читаемый формат даты и времени.
        df['date'] = pd.to_datetime(df['date'], unit='ms')


        # Установка даты в качестве индекса DataFrame.
        df.set_index('date', inplace=True)

        df.interpolate(method='linear', inplace=True)


        # Вычисление процентных изменений для ставок и объемов.
        df['rate_change'] = df['rate'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        # Расчет волатильности на основе процентного изменения ставок.
        df['volatility'] = df['rate_change'].rolling(window=12).std() * np.sqrt(12)

        # Удаление NaN значений, возникающих из-за первичных процентных изменений.
        df.dropna(inplace=True)

        return df

    def split_data(self, test_size=0.2):
        """
        Разделяет данные на обучающую и тестовую выборки.

        :param test_size: Процент данных, который будет использоваться для тестирования.
        :return: tuple (train_df, test_df)
        """
        test_len = int(len(self.data) * test_size)
        train_df = self.data[:-test_len]
        test_df = self.data[-test_len:]
        return train_df, test_df
