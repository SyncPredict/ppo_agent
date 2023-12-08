import gym
import numpy as np
import pandas as pd
from gym import spaces


class TradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, lookback_window_size=10, initial_balance=10000):
        super(TradingEnvironment, self).__init__()
        self.df = df
        self.max_steps = len(self.df) - 1
        self.current_step = lookback_window_size
        self.lookback_window_size = lookback_window_size

        # Действия: [0] - процент от баланса для покупки, [1] - процент от баланса для продажи
        self.action_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Наблюдения: стоимость Bitcoin, объем торгов, рыночная капитализация и другие показатели
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.df.columns) * lookback_window_size,),
            dtype=np.float32
        )
        self.balance = initial_balance
        self.capital = initial_balance
        self.transaction_fee_percent = 0.1
        self.btc_balance = 0.0
        self.purchased_prices = []
        self.dynamic_max_trade_percent = 0.1  # Изначальное значение
        self.risk_threshold = 0.05

        # Новые параметры для функции награды
        self.trend_lookback_window = 24  # Количество шагов для анализа тренда
        self.trend_reward_multiplier = 3  # Множитель для награды за правильное следование тренду

        self.state = self._next_observation()

    def _next_observation(self):
        """
        Получение наблюдения с окном исторических данных.
        """
        window_start = max(self.current_step - self.lookback_window_size, 0)
        window_end = self.current_step
        lookback_data = self.df.iloc[window_start:window_end]
        # Форматирование данных для нейронной сети
        obs = lookback_data.values.flatten()
        return obs

    def step(self, action):
        """
        Шаг среды на основе действия агента.
        Ограничение одновременной покупки и продажи.
        """
        self.current_step += 1
        current_price = self.df.iloc[self.current_step]['rate']
        action_value = action[0]

        sell_percent = -action_value if action_value < 0 else 0
        buy_percent = action_value if action_value > 0 else 0

        sell_amount = self.btc_balance * sell_percent
        buy_amount = self.balance * buy_percent / current_price

        # Обновление балансов
        reward, sell_amount, buy_amount = self.execute_trade(sell_amount, buy_amount, current_price)

        done = self.current_step >= self.max_steps
        self.state = self._next_observation()
        current_date = self.df.index[self.current_step]
        capital = self.balance + self.btc_balance * current_price
        self.capital = capital

        return self.state, reward, done, {
            'capital': self.capital,
            'current_price': current_price,
            'action': [sell_amount, buy_amount],
            'date': current_date
        }

    def calculate_trend_reward(self, current_price, sell_amount, buy_amount):
        """
        Расчет награды на основе действий агента и тренда цены.
        """
        if self.current_step < self.trend_lookback_window:
            return 0  # Недостаточно данных для анализа тренда

        trend_prices = self.df['rate'].iloc[self.current_step - self.trend_lookback_window:self.current_step]
        trend = np.polyfit(range(self.trend_lookback_window), trend_prices, 1)[0]

        # Положительный тренд
        if trend > 0:
            if sell_amount > 0:  # Награда за продажу при положительном тренде
                return trend * self.trend_reward_multiplier
            if buy_amount > 0:  # Штраф за покупку при положительном тренде
                return -trend * self.trend_reward_multiplier

        # Отрицательный тренд
        elif trend < 0:
            if sell_amount > 0:  # Штраф за продажу при отрицательном тренде
                return -abs(trend) * self.trend_reward_multiplier
            if buy_amount == 0:  # Награда за отсутствие покупок при отрицательном тренде
                return abs(trend) * self.trend_reward_multiplier

        return 0

    def execute_trade(self, sell_amount, buy_amount, current_price):
        """
        Выполнение торговых операций и обновление балансов.
        """
        # Ограничение максимального процента продажи и покупки

        recent_volatility = np.std(self.df['rate'].iloc[self.current_step - 10:self.current_step].pct_change())
        # Динамический процент торговли на основе волатильности
        dynamic_max_trade_percent = self.dynamic_max_trade_percent + (recent_volatility * 10)
        # Ограничиваем его максимальным значением
        dynamic_max_trade_percent = min(dynamic_max_trade_percent, 1.0)

        max_sell_amount = self.btc_balance * dynamic_max_trade_percent
        max_buy_amount = self.balance * dynamic_max_trade_percent / current_price

        sell_amount = min(sell_amount, max_sell_amount)
        buy_amount = min(buy_amount, max_buy_amount)

        reward = self.calculate_trend_reward(current_price, sell_amount, buy_amount)
        avg_purchase_price = np.mean(self.purchased_prices) if self.purchased_prices else 0

        # Продажа
        if sell_amount > 0:
            self.balance += sell_amount * current_price * (1 - self.transaction_fee_percent / 100)
            self.btc_balance -= sell_amount
            profit = (current_price - avg_purchase_price) * sell_amount
            reward += profit

        # Покупка
        if buy_amount > 0:
            self.balance -= buy_amount * current_price * (1 + self.transaction_fee_percent / 100)
            self.btc_balance += buy_amount
            self.purchased_prices.append(current_price)
            profit = (avg_purchase_price - current_price) * sell_amount
            reward += profit
        return reward, sell_amount, buy_amount

    def reset(self):
        """
        Сброс среды к начальному состоянию.
        """
        self.current_step = self.lookback_window_size
        self.balance = 10000
        self.btc_balance = 0.0
        self.purchased_prices = []
        self.state = self._next_observation()
        return self.state
