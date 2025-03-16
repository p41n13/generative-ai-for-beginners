{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Щоб запустити наступні записники, якщо ви ще цього не зробили, вам потрібно встановити ключ openai у файлі .env як `OPENAI_API_KEY`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: azure-ai-inference in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (1.0.0b7)\n",
      "Requirement already satisfied: azure-core>=1.30.0 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from azure-ai-inference) (1.32.0)\n",
      "Requirement already satisfied: isodate>=0.6.1 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from azure-ai-inference) (0.7.2)\n",
      "Requirement already satisfied: typing-extensions>=4.6.0 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from azure-ai-inference) (4.12.2)\n",
      "Requirement already satisfied: six>=1.11.0 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from azure-core>=1.30.0->azure-ai-inference) (1.17.0)\n",
      "Requirement already satisfied: requests>=2.21.0 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from azure-core>=1.30.0->azure-ai-inference) (2.32.3)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from requests>=2.21.0->azure-core>=1.30.0->azure-ai-inference) (2.3.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from requests>=2.21.0->azure-core>=1.30.0->azure-ai-inference) (3.10)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from requests>=2.21.0->azure-core>=1.30.0->azure-ai-inference) (3.4.1)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/docsa/CursorProjects/generative-ai-for-beginners/venv/lib/python3.10/site-packages (from requests>=2.21.0->azure-core>=1.30.0->azure-ai-inference) (2025.1.31)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "%pip install azure-ai-inference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import os\n",
    "import pandas as pd\n",
    "import requests\n",
    "import numpy as np\n",
    "from openai import OpenAI\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "load_dotenv()\n",
    "\n",
    "API_KEY = os.getenv(\"AIMLAPI_KEY\",\"\")\n",
    "assert API_KEY, \"ERROR: AIMLAPI Key is missing\"\n",
    "\n",
    "client = OpenAI(\n",
    "    api_key=API_KEY\n",
    "    )\n",
    "\n",
    "model = 'text-embedding-ada-002'\n",
    "\n",
    "SIMILARITIES_RESULTS_THRESHOLD = 0.75\n",
    "DATASET_NAME = \"../embedding_index_3m.json\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Далі ми завантажимо індекс вбудовувань у Pandas DataFrame. Індекс вбудовувань зберігається в JSON-файлі під назвою `embedding_index_3m.json`. Індекс вбудовувань містить вбудовування для кожного транскрипту YouTube до кінця жовтня 2023 року."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_dataset(source: str) -> pd.core.frame.DataFrame:\n",
    "    # Завантаження індексу відеосесій\n",
    "    pd_vectors = pd.read_json(source)\n",
    "    return pd_vectors.drop(columns=[\"text\"], errors=\"ignore\").fillna(\"\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Далі ми створимо функцію під назвою `get_videos`, яка шукатиме в індексі вбудовувань за запитом. Функція поверне 5 найрелевантніших відео, які найбільш схожі на запит. Функція працює наступним чином:\n",
    "\n",
    "1. Спочатку створюється копія індексу вбудовувань.\n",
    "2. Далі обчислюється вбудовування для запиту за допомогою API вбудовування OpenAI.\n",
    "3. Потім в індексі вбудовувань створюється новий стовпчик під назвою `similarity`. Стовпчик `similarity` містить косинусну подібність між вбудовуванням запиту та вбудовуванням для кожного сегмента відео.\n",
    "4. Далі індекс вбудовувань фільтрується за стовпчиком `similarity`. Індекс вбудовувань фільтрується, щоб включати тільки ті відео, які мають косинусну подібність більшу або рівну 0.75.\n",
    "5. Нарешті, індекс вбудовувань сортується за стовпчиком `similarity`, і повертаються 5 найкращих відео."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cosine_similarity(a, b):\n",
    "    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\n",
    "\n",
    "def get_videos(\n",
    "    query: str, dataset: pd.core.frame.DataFrame, rows: int\n",
    ") -> pd.core.frame.DataFrame:\n",
    "    # створення копії набору даних\n",
    "    video_vectors = dataset.copy()\n",
    "\n",
    "    # отримання вбудовувань для запиту за допомогою прямого виклику API\n",
    "    response = requests.post(\n",
    "        \"https://api.aimlapi.com/v1/embeddings\",\n",
    "        headers={\n",
    "            \"Authorization\": f\"Bearer {API_KEY}\",\n",
    "            \"Content-Type\": \"application/json\"\n",
    "        },\n",
    "        json={\n",
    "            \"model\": model,\n",
    "            \"input\": query,\n",
    "            \"encoding_format\": \"float\"\n",
    "        }\n",
    "    )\n",
    "    \n",
    "    data = response.json()\n",
    "    query_embeddings = data[\"data\"][0][\"embedding\"]\n",
    "    \n",
    "    # створення нового стовпчика з обчисленою подібністю для кожного рядка\n",
    "    video_vectors[\"similarity\"] = video_vectors[\"ada_v2\"].apply(\n",
    "        lambda x: cosine_similarity(np.array(query_embeddings), np.array(x))\n",
    "    )\n",
    "\n",
    "    # повернення верхніх рядків\n",
    "    return video_vectors.head(rows)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ця функція дуже проста, вона просто виводить результати пошукового запиту."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def display_results(videos: pd.core.frame.DataFrame, query: str):\n",
    "    def _gen_yt_url(video_id: str, seconds: int) -> str:\n",
    "        \"\"\"конвертація часу з формату 00:00:00 в секунди\"\"\"\n",
    "        return f\"https://youtu.be/{video_id}?t={seconds}\"\n",
    "\n",
    "    print(f\"\\nВідео, схожі на '{query}':\")\n",
    "    for _, row in videos.iterrows():\n",
    "        youtube_url = _gen_yt_url(row[\"videoId\"], row[\"seconds\"])\n",
    "        print(f\" - {row['title']}\")\n",
    "        print(f\"   Резюме: {' '.join(row['summary'].split()[:15])}...\")\n",
    "        print(f\"   YouTube: {youtube_url}\")\n",
    "        print(f\"   Схожість: {row['similarity']}\")\n",
    "        print(f\"   Доповідачі: {row['speaker']}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Спочатку індекс вбудовувань завантажується в DataFrame Pandas.\n",
    "2. Далі користувачу пропонується ввести запит.\n",
    "3. Потім викликається функція `get_videos` для пошуку в індексі вбудовувань за запитом.\n",
    "4. Нарешті, викликається функція `display_results` для відображення результатів користувачу.\n",
    "5. Потім користувачу пропонується ввести інший запит. Цей процес триває, доки користувач не введе `exit`.\n",
    "\n",
    "![](../images/notebook-search.png?WT.mc_id=academic-105485-koreyst)\n",
    "\n",
    "Вам буде запропоновано ввести запит. Введіть запит і натисніть Enter. Застосунок поверне список відео, які відповідають запиту. Застосунок також поверне посилання на місце у відео, де знаходиться відповідь на запитання.\n",
    "\n",
    "Ось кілька запитів, які можна спробувати:\n",
    "\n",
    "- Що таке Azure Machine Learning?\n",
    "- Як працюють згорткові нейронні мережі?\n",
    "- Що таке нейронна мережа?\n",
    "- Чи можу я використовувати Jupyter Notebooks з Azure Machine Learning?\n",
    "- Що таке ONNX?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Videos similar to 'What is ONNX?':\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: Join Seth Juarez as he discusses ethical concerns with AI in this episode of the...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=0\n",
      "   Similarity: 0.7106293845107627\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: In this video, the speaker discusses the challenges of building technology that is both responsible...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=187\n",
      "   Similarity: 0.723749443235198\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: The video discusses the limitations of generalizability in machine learning models and the importance of...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=373\n",
      "   Similarity: 0.7083360166792736\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: The video discusses the importance of considering various factors when developing AI technologies. It emphasizes...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=561\n",
      "   Similarity: 0.7193984747336011\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: The video discusses the importance of understanding the context and needs of users when developing...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=744\n",
      "   Similarity: 0.7061095018054818\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      "\n",
      "Videos similar to 'What is ONNX?':\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: Join Seth Juarez as he discusses ethical concerns with AI in this episode of the...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=0\n",
      "   Similarity: 0.7106293845107627\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: In this video, the speaker discusses the challenges of building technology that is both responsible...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=187\n",
      "   Similarity: 0.723749443235198\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: The video discusses the limitations of generalizability in machine learning models and the importance of...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=373\n",
      "   Similarity: 0.7083360166792736\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: The video discusses the importance of considering various factors when developing AI technologies. It emphasizes...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=561\n",
      "   Similarity: 0.7193984747336011\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n",
      " - You're Not Solving the Problem You Think You're Solving\n",
      "   Summary: The video discusses the importance of understanding the context and needs of users when developing...\n",
      "   YouTube: https://youtu.be/-tJQm4mSh1s?t=744\n",
      "   Similarity: 0.7061095018054818\n",
      "   Speakers: Seth Juarez, Josh Lovejoy, Sarah Bird\n"
     ]
    }
   ],
   "source": [
    "pd_vectors = load_dataset(DATASET_NAME)\n",
    "\n",
    "# отримання запиту користувача з вводу\n",
    "while True:\n",
    "    query = input(\"Введіть запит: \")\n",
    "    if query == \"exit\":\n",
    "        break\n",
    "    videos = get_videos(query, pd_vectors, 5)\n",
    "    display_results(videos, query)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}