from data import train_data, test_data

# Создание словаря
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)

print('%d найдены уникальные слова' % vocab_size)  # найдено 18 уникальных слов
# Назначить индекс каждому слову
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

print(word_to_idx['good'])  # 16 (это может измениться)
print(idx_to_word[0])  # грустно (это может измениться)