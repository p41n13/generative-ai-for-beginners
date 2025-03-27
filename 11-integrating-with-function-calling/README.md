# Інтеграція з викликом функцій

[![Інтеграція з викликом функцій](./images/11-lesson-banner.png?WT.mc_id=academic-105485-koreyst)](https://aka.ms/gen-ai-lesson11-gh?WT.mc_id=academic-105485-koreyst)

Ви вже чимало дізналися з попередніх уроків. Однак ми можемо покращити знання ще більше. Деякі речі, які ми можемо покращити, це як отримати більш узгоджений формат відповіді, щоб полегшити роботу з відповіддю на подальших етапах. Також ми можемо додати дані з інших джерел для подальшого збагачення нашого додатку.

Вищезгадані проблеми і є тим, що цей розділ має на меті вирішити.

## Вступ

Цей урок охопить:

- Пояснення, що таке виклик функцій та випадки його використання.
- Створення виклику функції за допомогою Azure OpenAI.
- Як інтегрувати виклик функції в додаток.

## Цілі навчання

До кінця цього уроку ви зможете:

- Пояснити мету використання виклику функцій.
- Налаштувати виклик функцій за допомогою служби Azure OpenAI.
- Розробити ефективні виклики функцій для конкретного випадку використання вашого додатку.

## Сценарій: вдосконалення нашого чат-бота за допомогою функцій

Для цього уроку ми хочемо створити функцію для нашого освітнього стартапу, яка дозволить користувачам використовувати чат-бот для пошуку технічних курсів. Ми будемо рекомендувати курси, які відповідають їхньому рівню навичок, поточній ролі та технологіям, які їх цікавлять.

Для реалізації цього сценарію ми використаємо комбінацію:

- `Azure OpenAI` для створення зручного інтерфейсу чату для користувача.
- `Microsoft Learn Catalog API` для допомоги користувачам у пошуку курсів на основі їхніх запитів.
- `Виклик функцій` для того, щоб взяти запит користувача та надіслати його функції, яка здійснить API-запит.

Для початку, давайте розглянемо, чому ми взагалі хочемо використовувати виклик функцій:

## Навіщо потрібен виклик функцій

До виклику функцій відповіді від LLM були неструктурованими і непослідовними. Розробники були змушені писати складний код для перевірки, щоб переконатися, що вони можуть обробити кожен варіант відповіді. Користувачі не могли отримати відповіді на такі запитання, як "Яка зараз погода в Стокгольмі?". Це пов'язано з тим, що моделі були обмежені часом, на якому дані були навчені.

Виклик функцій - це функція сервісу Azure OpenAI, що дозволяє подолати такі обмеження:

- **Узгоджений формат відповіді**. Якщо ми можемо краще контролювати формат відповіді, ми можемо легше інтегрувати відповідь з іншими системами.
- **Зовнішні дані**. Можливість використовувати дані з інших джерел додатку в контексті чату.

## Ілюстрація проблеми через сценарій

> Ми рекомендуємо вам використовувати [вбудований блокнот](./python/aoai-assignment.ipynb?WT.mc_id=academic-105485-koreyst), якщо ви хочете запустити наведений нижче сценарій. Ви також можете просто читати далі, оскільки ми намагаємося проілюструвати проблему, де функції можуть допомогти її вирішити.

Давайте розглянемо приклад, який ілюструє проблему формату відповіді:

Припустимо, ми хочемо створити базу даних студентів, щоб запропонувати їм відповідний курс. Нижче ми маємо два описи студентів, які дуже схожі за даними, які вони містять.

1. Створюємо з'єднання з нашим ресурсом Azure OpenAI:

   ```python
   import os
   import json
   from openai import AzureOpenAI
   from dotenv import load_dotenv
   load_dotenv()

   client = AzureOpenAI(
   api_key=os.environ['AZURE_OPENAI_API_KEY'],  # це також значення за замовчуванням, його можна опустити
   api_version = "2023-07-01-preview"
   )

   deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']
   ```

   Нижче наведено код Python для налаштування нашого підключення до Azure OpenAI, де ми встановлюємо `api_type`, `api_base`, `api_version` та `api_key`.

1. Створення двох описів студентів за допомогою змінних `student_1_description` та `student_2_description`.

   ```python
   student_1_description="Emily Johnson is a sophomore majoring in computer science at Duke University. She has a 3.7 GPA. Emily is an active member of the university's Chess Club and Debate Team. She hopes to pursue a career in software engineering after graduating."

   student_2_description = "Michael Lee is a sophomore majoring in computer science at Stanford University. He has a 3.8 GPA. Michael is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after finishing his studies."
   ```

   Ми хочемо надіслати наведені вище описи студентів до LLM для аналізу даних. Ці дані пізніше можуть бути використані в нашому додатку і надіслані до API або збережені в базі даних.

1. Давайте створимо два ідентичні запити, в яких ми проінструктуємо LLM, яка інформація нас цікавить:

   ```python
   prompt1 = f'''
   Please extract the following information from the given text and return it as a JSON object:

   name
   major
   school
   grades
   club

   This is the body of text to extract the information from:
   {student_1_description}
   '''

   prompt2 = f'''
   Please extract the following information from the given text and return it as a JSON object:

   name
   major
   school
   grades
   club

   This is the body of text to extract the information from:
   {student_2_description}
   '''
   ```

   Наведені вище запити інструктують LLM витягти інформацію та повернути відповідь у форматі JSON.

1. Після налаштування запитів та підключення до Azure OpenAI, тепер ми надішлемо запити до LLM за допомогою `openai.ChatCompletion`. Ми зберігаємо запит у змінній `messages` і присвоюємо роль `user`. Це імітує повідомлення від користувача, написане до чат-бота.

   ```python
   # відповідь на перший запит
   openai_response1 = client.chat.completions.create(
   model=deployment,
   messages = [{'role': 'user', 'content': prompt1}]
   )
   openai_response1.choices[0].message.content

   # відповідь на другий запит
   openai_response2 = client.chat.completions.create(
   model=deployment,
   messages = [{'role': 'user', 'content': prompt2}]
   )
   openai_response2.choices[0].message.content
   ```

Тепер ми можемо надіслати обидва запити до LLM і проаналізувати отриману відповідь, знайшовши її як `openai_response1['choices'][0]['message']['content']`.

1. Нарешті, ми можемо перетворити відповідь у формат JSON, викликавши `json.loads`:

   ```python
   # Завантаження відповіді як об'єкта JSON
   json_response1 = json.loads(openai_response1.choices[0].message.content)
   json_response1
   ```

   Відповідь 1:

   ```json
   {
     "name": "Emily Johnson",
     "major": "computer science",
     "school": "Duke University",
     "grades": "3.7",
     "club": "Chess Club"
   }
   ```

   Відповідь 2:

   ```json
   {
     "name": "Michael Lee",
     "major": "computer science",
     "school": "Stanford University",
     "grades": "3.8 GPA",
     "club": "Robotics Club"
   }
   ```

   Незважаючи на те, що запити однакові, а описи схожі, ми бачимо, що значення властивості `Grades` відформатовані по-різному, оскільки іноді можемо отримати формат `3.7` або `3.7 GPA`, наприклад.

   Цей результат пов'язаний з тим, що LLM приймає неструктуровані дані у вигляді письмового запиту і також повертає неструктуровані дані. Нам потрібно мати структурований формат, щоб знати, чого очікувати при зберіганні або використанні цих даних.

Отже, як нам вирішити проблему форматування? За допомогою виклику функцій можна переконатися, що ми отримуємо структуровані дані. При використанні виклику функцій LLM фактично не викликає і не запускає жодних функцій. Натомість ми створюємо структуру, якої LLM має дотримуватися для своїх відповідей. Потім ми використовуємо ці структуровані відповіді, щоб знати, яку функцію запускати в наших додатках.

![функціональний потік](./images/Function-Flow.png?WT.mc_id=academic-105485-koreyst)

Потім ми можемо взяти те, що повертається з функції, і надіслати це назад до LLM. LLM відповість за допомогою природної мови на запит користувача.

## Випадки використання викликів функцій

Існує багато різних випадків використання, коли виклики функцій можуть покращити ваш додаток, наприклад:

- **Виклик зовнішніх інструментів**. Чат-боти чудово відповідають на запитання користувачів. Використовуючи виклик функцій, чат-боти можуть використовувати повідомлення від користувачів для виконання певних завдань. Наприклад, студент може попросити чат-бота "Надіслати електронного листа моєму інструктору з повідомленням, що мені потрібна додаткова допомога з цього предмету". Це може викликати функцію `send_email(to: string, body: string)`

- **Створення запитів до API або бази даних**. Користувачі можуть знаходити інформацію за допомогою природної мови, яка перетворюється на відформатований запит або запит до API. Прикладом цього може бути викладач, який запитує "Хто з студентів виконав останнє завдання", що може викликати функцію з назвою `get_completed(student_name: string, assignment: int, current_status: string)`

- **Створення структурованих даних**. Користувачі можуть взяти блок тексту або CSV і використовувати LLM для вилучення важливої інформації з нього. Наприклад, студент може перетворити статтю з Вікіпедії про мирні угоди, щоб створити флеш-картки з ШІ. Це можна зробити за допомогою функції з назвою `get_important_facts(agreement_name: string, date_signed: string, parties_involved: list)`

## Створення вашого першого виклику функції

Процес створення виклику функції включає 3 основні кроки:

1. **Виклик** API чат-завершень зі списком ваших функцій та повідомленням користувача.
2. **Читання** відповіді моделі для виконання дії, тобто виконання функції або виклику API.
3. **Створення** ще одного виклику до API чат-завершень з відповіддю від вашої функції, щоб використати цю інформацію для створення відповіді користувачеві.

![Потік LLM](./images/LLM-Flow.png?WT.mc_id=academic-105485-koreyst)

### Крок 1 - створення повідомлень

Перший крок - створити повідомлення користувача. Це можна динамічно призначити, взявши значення текстового вводу, або можна призначити значення тут. Якщо це ваш перший досвід роботи з API чат-завершень, нам потрібно визначити `role` та `content` повідомлення.

Роль `role` може бути або `system` (створення правил), `assistant` (модель) або `user` (кінцевий користувач). Для виклику функцій ми призначимо це як `user` і приклад запитання.

```python
messages= [ {"role": "user", "content": "Find me a good course for a beginner student to learn Azure."} ]
```

Присвоюючи різні ролі, для LLM стає зрозуміло, чи це система щось каже, чи користувач, що допомагає створити історію розмови, на яку LLM може спиратися.

### Крок 2 - створення функцій

Далі, ми визначимо функцію та параметри цієї функції. Тут ми використаємо лише одну функцію під назвою `search_courses`, але ви можете створити кілька функцій.

> **Важливо**: Функції включаються в системне повідомлення до LLM і будуть враховуватися в кількості доступних токенів, які ви маєте.

Нижче ми створюємо функції як масив елементів. Кожен елемент є функцією і має властивості `name`, `description` та `parameters`:

```python
functions = [
   {
      "name":"search_courses",
      "description":"Retrieves courses from the search index based on the parameters provided",
      "parameters":{
         "type":"object",
         "properties":{
            "role":{
               "type":"string",
               "description":"The role of the learner (i.e. developer, data scientist, student, etc.)"
            },
            "product":{
               "type":"string",
               "description":"The product that the lesson is covering (i.e. Azure, Power BI, etc.)"
            },
            "level":{
               "type":"string",
               "description":"The level of experience the learner has prior to taking the course (i.e. beginner, intermediate, advanced)"
            }
         },
         "required":[
            "role"
         ]
      }
   }
]
```

Давайте опишемо кожен екземпляр функції більш детально нижче:

- `name` - Ім'я функції, яку ми хочемо викликати.
- `description` - Це опис того, як працює функція. Тут важливо бути конкретним і чітким.
- `parameters` - Список значень і формат, які ви хочете, щоб модель виробляла у своїй відповіді. Масив параметрів складається з елементів, де кожен елемент має такі властивості:
  1.  `type` - Тип даних, в яких будуть зберігатися властивості.
  1.  `properties` - Список конкретних значень, які модель використовуватиме для своєї відповіді
      1. `name` - Ключ - це назва властивості, яку модель використовуватиме у своїй відформатованій відповіді, наприклад, `product`.
      1. `type` - Тип даних цієї властивості, наприклад, `string`.
      1. `description` - Опис конкретної властивості.

Також є необов'язкова властивість `required` - необхідна властивість для завершення виклику функції.

### Крок 3 - Виконання виклику функції

Після визначення функції, нам потрібно включити її в виклик API чат-завершень. Ми робимо це, додаючи `functions` до запиту. У цьому випадку `functions=functions`.

Також є можливість встановити `function_call` в значення `auto`. Це означає, що ми дозволимо LLM вирішити, яку функцію слід викликати на основі повідомлення користувача, а не призначати її самостійно.

Ось приклад коду нижче, де ми викликаємо `ChatCompletion.create`, зауважте, як ми встановлюємо `functions=functions` і `function_call="auto"`, тим самим даючи LLM вибір, коли викликати функції, які ми йому надаємо:

```python
response = client.chat.completions.create(model=deployment,
                                        messages=messages,
                                        functions=functions,
                                        function_call="auto")

print(response.choices[0].message)
```

Відповідь, яка тепер повертається, виглядає так:

```json
{
  "role": "assistant",
  "function_call": {
    "name": "search_courses",
    "arguments": "{\n  \"role\": \"student\",\n  \"product\": \"Azure\",\n  \"level\": \"beginner\"\n}"
  }
}
```

Тут ми бачимо, як була викликана функція `search_courses` і з якими аргументами, як зазначено у властивості `arguments` у відповіді JSON.

Висновок: LLM зміг знайти дані, що відповідають аргументам функції, оскільки він витягнув їх зі значення, наданого параметру `messages` у виклику завершення чату. Нижче нагадування про значення `messages`:

```python
messages= [ {"role": "user", "content": "Find me a good course for a beginner student to learn Azure."} ]
```

Як бачите, `student`, `Azure` та `beginner` були витягнуті з `messages` і встановлені як вхідні дані для функції. Використання функцій таким чином є чудовим способом вилучення інформації з підказки, а також забезпечення структури для LLM і наявності багаторазового функціоналу.

Далі нам потрібно побачити, як ми можемо використовувати це в нашому додатку.

## Інтегрування викликів функцій до додатка

Після того, як ми протестували відформатовану відповідь від LLM, тепер ми можемо інтегрувати це в додаток.

### Керування потоком

Щоб інтегрувати це в наш додаток, виконаємо такі кроки:

1. Спочатку зробимо виклик до сервісів Open AI та збережемо повідомлення у змінній під назвою `response_message`.

   ```python
   response_message = response.choices[0].message
   ```

1. Тепер визначимо функцію, яка викликатиме API Microsoft Learn для отримання списку курсів:

   ```python
   import requests

   def search_courses(role, product, level):
     url = "https://learn.microsoft.com/api/catalog/"
     params = {
        "role": role,
        "product": product,
        "level": level
     }
     response = requests.get(url, params=params)
     modules = response.json()["modules"]
     results = []
     for module in modules[:5]:
        title = module["title"]
        url = module["url"]
        results.append({"title": title, "url": url})
     return str(results)
   ```

   Зверніть увагу, що тепер ми створюємо реальну функцію Python, яка відповідає іменам функцій, представленим у змінній `functions`. Ми також робимо реальні зовнішні API-виклики для отримання потрібних даних. У цьому випадку ми звертаємось до API Microsoft Learn для пошуку навчальних модулів.

Гаразд, ми створили змінні `functions` і відповідну функцію Python, як сказати LLM, як зіставити ці два елементи, щоб наша функція Python була викликана?

1. Щоб перевірити, чи потрібно викликати функцію Python, нам потрібно подивитися у відповідь LLM і перевірити, чи є там частина `function_call`, і викликати вказану функцію. Ось як можна зробити цю перевірку:

   ```python
   # Перевіряємо, чи модель хоче викликати функцію
   if response_message.function_call.name:
    print("Recommended Function call:")
    print(response_message.function_call.name)
    print()

    # Викликаємо функцію.
    function_name = response_message.function_call.name

    available_functions = {
            "search_courses": search_courses,
    }
    function_to_call = available_functions[function_name]

    function_args = json.loads(response_message.function_call.arguments)
    function_response = function_to_call(**function_args)

    print("Output of function call:")
    print(function_response)
    print(type(function_response))


    # Додаємо відповідь асистента та відповідь функції до повідомлень
    messages.append( # додаємо відповідь асистента до повідомлень
        {
            "role": response_message.role,
            "function_call": {
                "name": function_name,
                "arguments": response_message.function_call.arguments,
            },
            "content": None
        }
    )
    messages.append( # додаємо відповідь функції до повідомлень
        {
            "role": "function",
            "name": function_name,
            "content":function_response,
        }
    )
   ```

   Ці три рядки забезпечують вилучення імені функції, аргументів і здійснення виклику:

   ```python
   function_to_call = available_functions[function_name]

   function_args = json.loads(response_message.function_call.arguments)
   function_response = function_to_call(**function_args)
   ```

   Нижче наведено результат виконання нашого коду:

   **Вивід**

   ```Recommended Function call:
   {
     "name": "search_courses",
     "arguments": "{\n  \"role\": \"student\",\n  \"product\": \"Azure\",\n  \"level\": \"beginner\"\n}"
   }

   Output of function call:
   [{'title': 'Describe concepts of cryptography', 'url': 'https://learn.microsoft.com/training/modules/describe-concepts-of-cryptography/?
   WT.mc_id=api_CatalogApi'}, {'title': 'Introduction to audio classification with TensorFlow', 'url': 'https://learn.microsoft.com/en-
   us/training/modules/intro-audio-classification-tensorflow/?WT.mc_id=api_CatalogApi'}, {'title': 'Design a Performant Data Model in Azure SQL
   Database with Azure Data Studio', 'url': 'https://learn.microsoft.com/training/modules/design-a-data-model-with-ads/?
   WT.mc_id=api_CatalogApi'}, {'title': 'Getting started with the Microsoft Cloud Adoption Framework for Azure', 'url':
   'https://learn.microsoft.com/training/modules/cloud-adoption-framework-getting-started/?WT.mc_id=api_CatalogApi'}, {'title': 'Set up the
   Rust development environment', 'url': 'https://learn.microsoft.com/training/modules/rust-set-up-environment/?WT.mc_id=api_CatalogApi'}]
   <class 'str'>
   ```

1. Тепер ми надішлемо оновлене повідомлення `messages` до LLM, щоб отримати відповідь природною мовою замість відповіді у форматі JSON API.

   ```python
   print("Messages in next request:")
   print(messages)
   print()

   second_response = client.chat.completions.create(
      messages=messages,
      model=deployment,
      function_call="auto",
      functions=functions,
      temperature=0
         )  # отримуємо нову відповідь від GPT, де він може бачити відповідь функції


   print(second_response.choices[0].message)
   ```

   **Вивід**

   ```python
   {
     "role": "assistant",
     "content": "I found some good courses for beginner students to learn Azure:\n\n1. [Describe concepts of cryptography] (https://learn.microsoft.com/training/modules/describe-concepts-of-cryptography/?WT.mc_id=api_CatalogApi)\n2. [Introduction to audio classification with TensorFlow](https://learn.microsoft.com/training/modules/intro-audio-classification-tensorflow/?WT.mc_id=api_CatalogApi)\n3. [Design a Performant Data Model in Azure SQL Database with Azure Data Studio](https://learn.microsoft.com/training/modules/design-a-data-model-with-ads/?WT.mc_id=api_CatalogApi)\n4. [Getting started with the Microsoft Cloud Adoption Framework for Azure](https://learn.microsoft.com/training/modules/cloud-adoption-framework-getting-started/?WT.mc_id=api_CatalogApi)\n5. [Set up the Rust development environment](https://learn.microsoft.com/training/modules/rust-set-up-environment/?WT.mc_id=api_CatalogApi)\n\nYou can click on the links to access the courses."
   }

   ```

   ## Завдання

Щоб продовжити вивчення виклику функцій Azure OpenAI, ви можете створити:

- Більше параметрів функції, які можуть допомогти учням знайти більше курсів.
- Створити інший виклик функції, який збирає більше інформації від учня, наприклад, його рідну мову.
- Створити обробку помилок, коли виклик функції та/або виклик API не повертає жодних відповідних курсів.

Підказка: Дотримуйтесь [документації з довідника API Learn](https://learn.microsoft.com/training/support/catalog-api-developer-reference?WT.mc_id=academic-105485-koreyst), щоб побачити, як і де ці дані доступні.

## Чудова робота! Продовжуйте подорож

Після завершення цього уроку ознайомтеся з нашою [колекцією навчальних матеріалів з генеративного ШІ](https://aka.ms/genai-collection?WT.mc_id=academic-105485-koreyst), щоб продовжити підвищувати свої знання з генеративного ШІ!
