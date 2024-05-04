from test_model import Model_center

model_center = Model_center()

user_question = input("请输入您的问题：")

result = model_center.question_handler(user_question)

print("RAG检索到的上下文：", result)
print("聊天机器人的回答：", result[0][1])

import pandas as pd
# 读取原始CSV文件
data = pd.read_csv('test_sample.csv', encoding='utf-8')

# 实例化Model_center对象
# model_center = Model_center()

# 新CSV文件的数据框
new_data = []

# 遍历每个问题
for index, row in data.iterrows():
    user_question = row['问题内容']
    
    # 调用question_handler方法处理问题
    result = model_center.question_handler(user_question)
    
    # 组装新数据
    new_row = {
        '问题类型': row['问题类型'],
        '问题内容': user_question,
        'RAG检索的上下文': result[1],
        '聊天机器人的回答': result[0][1][-1]
    }
    
    print(result[0][1][-1])
    
    new_data.append(new_row)

# 创建新的数据框
new_data_df = pd.DataFrame(new_data)

# 写入新的CSV文件，使用UTF-8编码
new_data_df.to_csv('test_output.csv', index=False, encoding='utf-8')

print("新的CSV文件已生成。")