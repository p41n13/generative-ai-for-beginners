# Розділ 7: Створення чат-додатків
## Швидкий старт з API моделей Github

Цей ноутбук адаптовано з [репозиторію прикладів Azure OpenAI](https://github.com/Azure/azure-openai-samples?WT.mc_id=academic-105485-koreyst), який включає ноутбуки для доступу до сервісів [Azure OpenAI](notebook-azure-openai.ipynb).

# Огляд  
"Великі мовні моделі - це функції, які перетворюють текст на текст. Отримавши вхідний рядок тексту, велика мовна модель намагається передбачити текст, який ітиме далі"(1). Цей "швидкий старт" познайомить користувачів з концепціями LLM високого рівня, основними вимогами пакетів для початку роботи з AML, м'яким введенням у дизайн промптів та кількома короткими прикладами різних випадків використання.

## Зміст  

[Огляд](#огляд)  
[Як використовувати сервіс OpenAI](#як-використовувати-сервіс-openai)  
[1. Створення вашого сервісу OpenAI](#1.-створення-вашого-сервісу-openai)  
[2. Встановлення](#2.-встановлення)    
[3. Облікові дані](#3.-облікові-дані)  

[Випадки використання](#випадки-використання)    
[1. Підсумовування тексту](#1.-підсумовування-тексту)  
[2. Класифікація тексту](#2.-класифікація-тексту)  
[3. Генерація нових назв продуктів](#3.-генерація-нових-назв-продуктів)  
[4. Тонке налаштування класифікатора](#4.тонке-налаштування-класифікатора)  

[Посилання](#посилання)

### Побудуйте свій перший промпт  
Ця коротка вправа надасть базове введення для подання промптів моделі в Github Models для простого завдання "підсумовування".


**Кроки**:  
1. Встановіть бібліотеку `azure-ai-inference` у своєму середовищі Python, якщо ви ще цього не зробили.  
2. Завантажте стандартні допоміжні бібліотеки та налаштуйте облікові дані для Github Models.  
3. Виберіть модель для вашого завдання  
4. Створіть простий промпт для моделі  
5. Надішліть запит до API моделі!

### 1. Встановіть `azure-ai-inference`

```python
%pip install azure-ai-inference
```

### 2. Імпортуйте допоміжні бібліотеки та створіть облікові дані

```python
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)
```

### 3. Пошук правильної моделі  
Моделі GPT-3.5-turbo або GPT-4 можуть розуміти та генерувати природну мову.

```python
# Виберіть модель загального призначення curie для тексту
model_name = "gpt-4o"
```

## 4. Дизайн промпту  

"Магія великих мовних моделей полягає в тому, що, навчаючись мінімізувати цю похибку передбачення на великих обсягах тексту, моделі в результаті вивчають концепції, корисні для цих передбачень. Наприклад, вони вивчають такі концепції як"(1):

* як правильно писати
* як працює граматика
* як перефразовувати
* як відповідати на запитання
* як вести розмову
* як писати багатьма мовами
* як кодувати
* тощо.

#### Як керувати великою мовною моделлю  
"З усіх входів до великої мовної моделі, безумовно, найбільш впливовим є текстовий промпт"(1).

Великі мовні моделі можна спонукати до створення виводу кількома способами:

Інструкція: Скажіть моделі, що ви хочете
Завершення: Спонукайте модель завершити початок того, що ви хочете
Демонстрація: Покажіть моделі, що ви хочете, за допомогою:
Кількох прикладів у промпті
Багатьох сотень або тисяч прикладів у навчальному наборі даних для тонкого налаштування"


#### Існують три основні рекомендації щодо створення промптів:

**Показуйте та розповідайте**. Чітко вказуйте, що ви хочете, через інструкції, приклади або їх комбінацію. Якщо ви хочете, щоб модель розташувала список елементів в алфавітному порядку або класифікувала абзац за настроєм, покажіть їй, що саме ви хочете.

**Надавайте якісні дані**. Якщо ви намагаєтеся створити класифікатор або змусити модель слідувати певній схемі, переконайтеся, що є достатньо прикладів. Обов'язково перевірте свої приклади — модель зазвичай достатньо розумна, щоб зрозуміти основні орфографічні помилки та дати вам відповідь, але вона також може припустити, що це навмисно, і це може вплинути на відповідь.

**Перевіряйте налаштування**. Параметри temperature та top_p контролюють, наскільки детермінованою є модель при генерації відповіді. Якщо ви просите відповідь, де є лише одна правильна відповідь, то вам варто встановити їх нижче. Якщо ви шукаєте більш різноманітні відповіді, то можливо вам захочеться встановити їх вище. Найпоширенішою помилкою, яку люди роблять з цими налаштуваннями, є припущення, що вони є елементами керування "розумністю" або "креативністю".


Джерело: https://github.com/Azure/OpenAI/blob/main/How%20to/Completions.md

### 5. Надсилаємо!

```python
# Створіть свій перший промпт
text_prompt = "Should oxford commas always be used?"

response = client.complete(
  model=model_name,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":text_prompt},])

response.choices[0].message.content
```

### Повторіть той самий виклик, як порівнюються результати?

```python
response = client.complete(
  model=model_name,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":text_prompt},])

response.choices[0].message.content
```

## Підсумовування тексту  
#### Завдання  
Підсумуйте текст, додавши 'tl;dr:' в кінці текстового уривку. Зверніть увагу, як модель розуміє, як виконувати низку завдань без додаткових інструкцій. Ви можете експериментувати з більш описовими промптами, ніж tl;dr, щоб модифікувати поведінку моделі та налаштувати підсумок, який ви отримуєте(3).  

Останні роботи продемонстрували значні успіхи в багатьох завданнях та тестах NLP шляхом попереднього навчання на великому корпусі тексту з подальшим тонким налаштуванням на конкретне завдання. Хоча зазвичай архітектура не залежить від завдання, цей метод все ще вимагає наборів даних для тонкого налаштування, специфічних для завдання, які містять тисячі або десятки тисяч прикладів. На відміну від цього, люди зазвичай можуть виконувати нове мовне завдання лише з кількох прикладів або з простих інструкцій - щось, з чим поточні системи NLP все ще значною мірою борються. Тут ми показуємо, що масштабування мовних моделей значно покращує ефективність, не залежну від завдання, з кількома прикладами, іноді навіть досягаючи конкурентоспроможності з попередніми найсучаснішими підходами тонкого налаштування.

Tl;dr

# Вправи для кількох випадків використання  
1. Підсумовування тексту  
2. Класифікація тексту  
3. Генерація нових назв продуктів

```python
prompt = "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something that current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.\n\nTl;dr"
```

```python
#Встановлення кількох додаткових типових параметрів під час виклику API

response = client.complete(
  model=model_name,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":prompt},])

response.choices[0].message.content
```

## Класифікація тексту  
#### Завдання  
Класифікуйте елементи на категорії, надані під час виведення. У наступному прикладі ми надаємо як категорії, так і текст для класифікації у промпті(*playground_reference). 

Запит клієнта: Привіт, одна з клавіш на клавіатурі мого ноутбука нещодавно зламалася, і мені потрібна заміна:

Класифікована категорія:

```python
prompt = "Classify the following inquiry into one of the following: categories: [Pricing, Hardware Support, Software Support]\n\ninquiry: Hello, one of the keys on my laptop keyboard broke recently and I'll need a replacement:\n\nClassified category:"
print(prompt)
```

```python
#Встановлення кількох додаткових типових параметрів під час виклику API

response = client.complete(
  model=model_name,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":prompt},])

response.choices[0].message.content
```

## Генерація нових назв продуктів
#### Завдання
Створіть назви продуктів з прикладів слів. Тут ми включаємо в промпт інформацію про продукт, для якого ми збираємося генерувати назви. Ми також надаємо схожий приклад, щоб показати схему, яку ми хочемо отримати. Ми також встановили високе значення температури, щоб збільшити випадковість та отримати більш інноваційні відповіді.

Опис продукту: Домашній міксер для молочних коктейлів
Ключові слова: швидкий, здоровий, компактний.
Назви продуктів: HomeShaker, Fit Shaker, QuickShake, Shake Maker

Опис продукту: Пара взуття, яка може підходити для будь-якого розміру стопи.
Ключові слова: адаптивний, що підходить, omni-fit.

```python
prompt = "Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: A pair of shoes that can fit any foot size.\nSeed words: adaptable, fit, omni-fit."

print(prompt)
```

```python
#Встановлення кількох додаткових типових параметрів під час виклику API

response = client.complete(
  model=model_name,
  messages = [{"role":"system", "content":"You are a helpful assistant."},
               {"role":"user","content":prompt}])

response.choices[0].message.content
```

# Посилання  
- [Openai Cookbook](https://github.com/openai/openai-cookbook?WT.mc_id=academic-105485-koreyst)  
- [OpenAI Studio Examples](https://oai.azure.com/portal?WT.mc_id=academic-105485-koreyst)  
- [Best practices for fine-tuning GPT-3 to classify text](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit#?WT.mc_id=academic-105485-koreyst)

# Для отримання додаткової допомоги  
[OpenAI Commercialization Team](AzureOpenAITeam@microsoft.com) 

# Учасники
* [Chew-Yean Yam](https://www.linkedin.com/in/cyyam/)