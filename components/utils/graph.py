import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_results(history):
    # Преобразование массива history в DataFrame
    df = pd.DataFrame(history)

    # Масштабирование данных методом Min-Max для капитала и цены
    min_max_scaler = MinMaxScaler()
    df[['capital', 'current_price']] = min_max_scaler.fit_transform(df[['capital', 'current_price']])

    df.ffill(inplace=True)

    # Создание графика
    plt.figure(figsize=(12, 6))

    # Добавление графиков капитала и текущей цены
    plt.plot(df['date'], df['capital'], label='Capital', linestyle='-.', linewidth=2)
    plt.plot(df['date'], df['current_price'], label='Current Price', linestyle='--', linewidth=2)

    # Добавление графика действий (покупки и продажи)
    for index, row in df.iterrows():
        action = row['action']
        if isinstance(action, list) and len(action) == 2:  # Проверка, что action является списком с двумя элементами
            sell_amount, buy_amount = action
            if sell_amount > 0:
                plt.plot(row['date'], df.at[index, 'current_price'], 'rv')  # Красная стрелка вниз
            if buy_amount > 0:
                plt.plot(row['date'], df.at[index, 'current_price'], 'g^')  # Зеленая стрелка вверх

    # Настройка меток осей и легенды
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.title('Dynamics of Capital and Current Price with Trade Actions (Normalized)')

    # Форматирование оси X для дат
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()
