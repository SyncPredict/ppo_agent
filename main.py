import torch
import matplotlib.pyplot as plt
from components.data.process_data import BitcoinDataProcessor
from components.trading_gym.gym_env import TradingEnvironment
from components.agent.ppo import ContinuousPPO
from components.utils.graph import plot_results

# Параметры
file_path = 'data.json'  # Укажите путь к файлу данных
lookback_window_size = 24
initial_balance = 10000

# Загрузка и предобработка данных
data_processor = BitcoinDataProcessor(file_path)
df_train, df_test = data_processor.split_data()

# Инициализация торговой среды
env = TradingEnvironment(df_train, lookback_window_size, initial_balance)
env_test = TradingEnvironment(df_test, lookback_window_size, initial_balance)

# Инициализация PPO агента
num_features = len(df_train.columns) * lookback_window_size
num_actions = env.action_space.shape[0]
ppo_agent = ContinuousPPO(num_features, num_actions)

# Параметры для обучения
num_episodes = 5  # Количество эпизодов для обучения
train_max_timesteps = (len(df_train) - 1) // 10
test_max_timesteps = (len(df_test) - 1) // 50

# Максимальное количество временных шагов в эпизоде

best_performance = -float('inf')


def print_progress(current_step, total_steps):
    progress = (current_step + 1) / total_steps
    bar_length = 40
    bar = '#' * int(progress * bar_length) + '-' * (bar_length - int(progress * bar_length))
    print(f'\r[{bar}] {current_step + 1}/{total_steps} ({progress * 100:.2f}%)', end='')


def percent_change(initial, final):
    if initial == 0:
        return float('inf')
    return (final - initial) / initial * 100


# Обучение агента

print(f'Max train timesteps = {train_max_timesteps} Max test timesteps = {test_max_timesteps}')

for episode in range(num_episodes):
    print(f"Episode {episode} Started.")
    history = []

    state = env.reset()
    episode_reward = 0

    states, actions, log_probs, rewards, dones = [], [], [], [], []

    for t in range(train_max_timesteps):
        action, log_prob = ppo_agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # Собираем данные роллаутов
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        dones.append(done)

        episode_reward += reward
        state = next_state

        print_progress(t, train_max_timesteps)

        if done:
            break

    values = []
    for state in states:
        _, _, value = ppo_agent.actor_critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        values.append(value)

    values = torch.tensor(values, dtype=torch.float32).squeeze()
    next_value = 0  # Или значение критика на последнем шаге, если эпизод не завершен
    returns = ppo_agent.compute_gae(next_value, rewards, dones, values)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = returns - values

    # Вызов функции update
    ppo_agent.update(states, actions, log_probs, returns, advantages)

    print(f"\nEpisode {episode} Score: {episode_reward}")

    state = env_test.reset()
    test_rewards = []

    print(f"Test Episode {episode} Started.")

    for t in range(test_max_timesteps):
        action, _ = ppo_agent.select_action(state, training=False)  # Добавьте параметр test в метод select_action
        state, reward, done, info = env_test.step(action)
        test_rewards.append(reward)

        print_progress(t, test_max_timesteps)

        if done:
            break

        history.append(info)

    initial_capital = initial_balance
    final_capital = history[-1]['capital']
    capital_change = percent_change(initial_capital, final_capital)

    initial_rate = history[0]['current_price']
    final_rate = history[-1]['current_price']
    rate_change = percent_change(initial_rate, final_rate)

    # Вычисляем производительность агента
    performance = capital_change - rate_change

    print(f"\nTest episode {episode} performance: {performance}. Capital profit: {final_capital - initial_capital}")

    # Сохраняем модель, если она показала лучшую производительность
    if performance > best_performance:
        best_performance = performance
        torch.save(ppo_agent.actor_critic.state_dict(), "best_ppo_model.pth")
        print(f"New best model saved with performance: {best_performance}")

        # Визуализация результатов
    plot_results(history)

torch.save(ppo_agent.actor_critic.state_dict(), "ppo_model_final.pth")
