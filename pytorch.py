import torch
import torch.nn as nn
import torch.optim as optim
import random

# Чтение данных из файла
file_path = "names.txt"
def read_names(file_path):
    with open(file_path, 'r') as file:
        names = [name.strip() for name in file.readlines()]
    return names

# Создание словаря символов и индексов
def create_char_dict(names):
    char_set = sorted(set(''.join(names)))
    char_dict = {char: i for i, char in enumerate(char_set)}
    return char_dict

# Создание обучающих данных
def create_training_data(names, char_dict):
    input_data = []
    target_data = []
    for name in names:
        name = '^' + name + '$'
        for i in range(len(name) - 1):
            input_data.append(char_dict[name[i]])
            target_data.append(char_dict[name[i+1]])
    return input_data, target_data

# Определение модели
class BigramLanguageModel(nn.Module):
    def _init_(self, input_size, hidden_size, output_size):
        super(BigramLanguageModel, self)._init_()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input):
        embedded = self.embedding(input)
        output = self.fc(embedded)
        return output

# Генерация имени
def generate_name(model, char_dict, start_char='^', max_length=20):
    with torch.no_grad():
        current_char = start_char
        name = ''
        while current_char != '$' and len(name) < max_length:
            input_tensor = torch.tensor([char_dict[current_char]], dtype=torch.long)
            output = model(input_tensor)
            _, predicted_index = torch.max(output, 1)
            predicted_char = list(char_dict.keys())[predicted_index.item()]
            name += predicted_char
            current_char = predicted_char
    return name

# Чтение данных из файла
names = read_names("names.txt")

# Создание словаря символов и индексов
char_dict = create_char_dict(names)

# Создание обучающих данных
input_data, target_data = create_training_data(names, char_dict)

# Определение параметров модели
input_size = len(char_dict)
hidden_size = 128
output_size = len(char_dict)

# Создание экземпляра модели
model = BigramLanguageModel(input_size, hidden_size, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Обучение модели
for epoch in range(100):
    # Преобразование данных в тензоры
    input_tensor = torch.tensor(input_data, dtype=torch.long)
    target_tensor = torch.tensor(target_data, dtype=torch.long)

    # Обнуление градиентов и выполнение прямого прохода
    optimizer.zero_grad()
    output = model(input_tensor)

    # Вычисление функции потерь и обратного прохода
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    # Вывод информации о процессе обучения
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}
