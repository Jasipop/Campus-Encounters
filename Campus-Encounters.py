import numpy as np
import matplotlib.pyplot as plt

states_female = ['女生宿舍', '教学楼', '图书馆', '食堂', '操场', '体育馆', '澡堂', '超市', '快递站', '麦当劳']
transition_matrix_female = [
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
]
states_male = ['男生宿舍', '教学楼', '图书馆', '食堂', '操场', '体育馆', '澡堂', '超市', '快递站', '麦当劳']
transition_matrix_male = [
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
]
initial_state_female = '女生宿舍'
initial_state_male = '男生宿舍'

def next_state(current_state, sex):
    if sex == 'female':
        return np.random.choice(states_female, p=transition_matrix_female[states_female.index(current_state)])
    elif sex == 'male':
        return np.random.choice(states_male, p=transition_matrix_male[states_male.index(current_state)])

# 定义`simulate_encounters`函数，确保它返回的是相遇次数
def simulate_encounters(initial_state_female, initial_state_male, num_steps):
    current_state_female = initial_state_female
    current_state_male = initial_state_male
    encounter_count = 0
    for _ in range(num_steps):
        # 随机决定当前是男生还是女生行动
        gender = np.random.choice(['female', 'male'], p=[0.5, 0.5])
        if gender == 'female':
            current_state_female = next_state(current_state_female, 'female')
        else:
            current_state_male = next_state(current_state_male, 'male')
        # 检查是否相遇
        if current_state_female == current_state_male:
            encounter_count += 1
    # 返回相遇次数，而不是概率
    return encounter_count

num_simulations = 10  # 每个步数下的模拟次数
step_range = range(100, 10001, 100)  # 不同的步数范围

# 模拟过程
average_encounter_probabilities = []  # 存储每个步数下的平均相遇概率
encounter_probabilities = []  # 存储所有模拟的相遇概率

for num_steps in step_range:
    step_encounter_probabilities = []  # 存储当前步数下所有模拟的相遇概率
    for _ in range(num_simulations):
        encounters = simulate_encounters(initial_state_female, initial_state_male, num_steps)
        # 计算相遇概率（相遇次数 / 模拟步数）
        encounter_probability = encounters / num_steps
        step_encounter_probabilities.append(encounter_probability)
    encounter_probabilities.extend(step_encounter_probabilities)  # 添加到总列表
    average_probability = np.mean(step_encounter_probabilities)  # 计算当前步数下的平均概率
    average_encounter_probabilities.append(average_probability)

# 计算所有模拟结果的均值和方差
overall_mean = np.mean(encounter_probabilities)
overall_variance = np.var(encounter_probabilities)

# 打印结果
print(f"Overall mean encounter probability: {overall_mean}")
print(f"Overall variance of encounter probabilities: {overall_variance}")

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(step_range, average_encounter_probabilities, marker='o')
plt.xlabel('steps')
plt.ylabel('average probability of encounter')
plt.title('Average Probability of Male-Female Encounter on Campus Over Steps')
plt.grid(True)
plt.show()