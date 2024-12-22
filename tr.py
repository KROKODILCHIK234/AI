import re
import os
import json
from typing import List, Tuple
import nltk
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# Загружаем необходимые компоненты NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

print("Загрузка модели DeepPavlov/rubert-base-cased...")
tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')

# Переводим модель в режим оценки и на CPU
model.eval()
model.cpu()

# Инициализация глобальных переменных для базы знаний
questions_embeddings = None
answers = None
original_questions = None


# Функции предобработки и получения эмбеддингов
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_embedding(text: str) -> np.ndarray:
    text = preprocess_text(text)
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encoded)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embedding


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


# Функции для работы с базой знаний
def build_knowledge_base(questions: List[str], answers_list: List[str]):
    global questions_embeddings, answers, original_questions
    print("Создание базы знаний...")

    original_questions = questions
    embeddings = []

    for i, question in enumerate(questions):
        if i % 10 == 0:
            print(f"Обработано {i}/{len(questions)} вопросов")
        embedding = get_embedding(question)
        embeddings.append(embedding)

    questions_embeddings = np.stack(embeddings)
    answers = answers_list
    print("База знаний создана")


def find_answer(question: str, top_k: int = 1, threshold: float = 0.5) -> List[Tuple[str, float]]:
    if questions_embeddings is None or answers is None:
        raise ValueError("База знаний не создана")

    question_embedding = get_embedding(question)
    similarities = []

    for i in range(len(questions_embeddings)):
        similarity = calculate_similarity(question_embedding, questions_embeddings[i])
        similarities.append(similarity)

    similarities = np.array(similarities)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    results = []

    for idx in top_indices:
        if similarities[idx] >= threshold:
            results.append((answers[idx], float(similarities[idx])))
        print(f"Debug - Похожий вопрос: {original_questions[idx]}")
        print(f"Debug - Сходство: {similarities[idx]:.4f}")

    if not results:
        return [("Извините, я не уверен в ответе на этот вопрос. Попробуйте переформулировать вопрос.", 0.0)]

    return results


def save_model(path: str):
    torch.save({
        'questions_embeddings': questions_embeddings,
        'answers': answers,
        'original_questions': original_questions
    }, path)
    print(f"Модель сохранена в {path}")


def load_model(path: str):
    global questions_embeddings, answers, original_questions
    checkpoint = torch.load(path, map_location='cpu')
    questions_embeddings = checkpoint['questions_embeddings']
    answers = checkpoint['answers']
    original_questions = checkpoint['original_questions']
    print(f"Модель загружена из {path}")


# Функции для обработки данных
def load_data(json_path: str) -> Tuple[List[str], List[str]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        data = raw_data['data']

    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    return questions, answers


def prepare_datasets(json_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[
    List[str], List[str], List[str], List[str]]:
    questions, answers = load_data(json_path)
    return train_test_split(
        questions,
        answers,
        test_size=test_size,
        random_state=random_state
    )


# Функции для тренировки и запуска
def train():
    try:
        print("Загрузка данных...")
        train_questions, test_questions, train_answers, test_answers = prepare_datasets('data.json', test_size=0.1)
        build_knowledge_base(train_questions, train_answers)
        save_model('faq_bot.pth')

        test_questions = [
            "Сколько бюджетных мест на факультете?",
            "Как получить повышенную стипендию?",
            "Есть ли на факультете военная кафедра?",
            "Как организована практика студентов?",
            "Есть ли на факультете магистратура?"
        ]

        print("\nПримеры работы бота:")
        for question in test_questions:
            print(f"\nВопрос: {question}")
            answers = find_answer(question, top_k=1)
            for answer, confidence in answers:
                print(f"Уверенность: {confidence:.2%}")
                print(f"Ответ: {answer}")

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")


def run_interactive_bot():
    try:
        if not os.path.exists('faq_bot.pth'):
            print("Файл модели не найден. Запустите сначала setup.py для создания модели.")
            return

        print("Загрузка модели...")
        load_model('faq_bot.pth')
        print("Модель загружена успешно")

        print("\nЗадавайте вопросы (для выхода введите 'выход' или 'exit')")

        while True:
            question = input("\nВаш вопрос: ").strip()

            if question.lower() in ['выход', 'exit', 'quit', 'q']:
                print("До свидания!")
                break

            if not question:
                print("Вопрос не может быть пустым")
                continue

            try:
                answers = find_answer(question, top_k=1)

                for answer, confidence in answers:
                    print(f"\nОтвет (уверенность: {confidence:.2%}):")
                    print(answer)

                    if confidence < 0.5:
                        print("\nПримечание: уверенность в ответе низкая, возможно, стоит переформулировать вопрос")

            except Exception as e:
                print(f"Ошибка при получении ответа: {str(e)}")

    except Exception as e:
        print(f"Ошибка при запуске бота: {str(e)}")


if __name__ == "__main__":
    if not os.path.exists('faq_bot.pth'):
        train()
    run_interactive_bot()