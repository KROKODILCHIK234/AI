import re
import os
import json
from typing import List, Tuple
import nltk
import numpy as np
import torch
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# Глобальные переменные для модели и токенизатора
tokenizer = None
model = None

# Глобальные переменные для базы знаний
questions_embeddings = None
answers = None
original_questions = None


def initialize_model():
    """Инициализация модели и токенизатора"""
    global tokenizer, model
    if tokenizer is None or model is None:
        print("Загрузка модели DeepPavlov/rubert-base-cased...")
        tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
        model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
        model.eval()
        model.cpu()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_embedding(text: str) -> np.ndarray:
    global tokenizer, model
    if tokenizer is None or model is None:
        initialize_model()

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
    """
    Сохранение состояния модели с использованием numpy для числовых данных
    и обычного json для текстовых данных
    """
    # Сохраняем эмбеддинги отдельно через numpy
    embeddings_path = path + '.npz'
    np.savez_compressed(embeddings_path, embeddings=questions_embeddings)

    # Сохраняем текстовые данные через json
    text_data = {
        'answers': answers,
        'original_questions': original_questions
    }
    text_path = path + '.json'
    with open(text_path, 'w', encoding='utf-8') as f:
        json.dump(text_data, f, ensure_ascii=False, indent=2)

    print(f"Модель сохранена в {path}")


def load_model(path: str):
    """
    Загрузка состояния модели из разделённых файлов
    """
    global questions_embeddings, answers, original_questions

    try:
        # Загружаем эмбеддинги
        with np.load(path) as data:
            questions_embeddings = data['embeddings']

        # Загружаем текстовые данные
        text_path = path.replace('.npz', '.json')
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
            answers = text_data['answers']
            original_questions = text_data['original_questions']

        print(f"Модель успешно загружена из {path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        raise


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


def train():
    try:
        initialize_model()
        print("Загрузка данных...")
        train_questions, test_questions, train_answers, test_answers = prepare_datasets('data.json', test_size=0.1)
        build_knowledge_base(train_questions, train_answers)
        save_model('lo.pth')

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
        if not os.path.exists('lo.pth.npz'):
            print("Файл модели не найден. Запустите сначала setup.py для создания модели.")
            return

        print("Загрузка модели...")
        initialize_model()
        load_model('lo.pth.npz')
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


def main():
    # Добавляем поддержку многопроцессорности
    multiprocessing.freeze_support()

    if not os.path.exists('lo.pth.npz'):
        train()
    run_interactive_bot()


if __name__ == "__main__":
    main()