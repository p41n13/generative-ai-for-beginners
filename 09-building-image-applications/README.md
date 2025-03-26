# Створення застосунків для генерації зображень

[![Створення застосунків для генерації зображень](./images/09-lesson-banner.png?WT.mc_id=academic-105485-koreyst)](https://aka.ms/gen-ai-lesson9-gh?WT.mc_id=academic-105485-koreyst)

Великі мовні моделі (LLM) здатні на більше, ніж просто генерація тексту. Можливо також генерувати зображення з текстових описів. Використання зображень як модальності може бути дуже корисним у ряді галузей: медичні технології, архітектура, туризм, розробка ігор та інші. У цьому розділі ми розглянемо дві найпопулярніші моделі генерації зображень, DALL-E та Midjourney.

## Вступ

У цьому уроці ми розглянемо:

- Генерацію зображень та її користь.
- DALL-E та Midjourney: що це таке і як вони працюють.
- Як створити застосунок для генерації зображень.

## Цілі навчання

Після завершення цього уроку ви зможете:

- Створювати застосунки для генерації зображень.
- Визначати межі для вашого застосунку за допомогою мета-промптів.
- Працювати з DALL-E та Midjourney.

## Навіщо створювати застосунок для генерації зображень?

Застосунки для генерації зображень - це чудовий спосіб дослідити можливості генеративного ШІ. Вони можуть використовуватися, наприклад, для:

- **Редагування та синтезу зображень**. Ви можете генерувати зображення для різних випадків використання, таких як редагування та синтез зображень.

- **Застосування в різних галузях**. Їх також можна використовувати для генерації зображень для різних галузей, таких як медичні технології, туризм, розробка ігор та інші.

## Сценарій: Edu4All

У рамках цього уроку ми продовжимо працювати з нашим стартапом Edu4All. Студенти створюватимуть зображення для своїх завдань. Які саме зображення – вирішуватимуть самі студенти, але це можуть бути ілюстрації для власної казки, створення нового персонажа для розповіді або візуалізація своїх ідей та концепцій.

Ось приклад того, що могли б згенерувати студенти Edu4All, якщо вони працюють на уроці над пам'ятками:

![Стартап Edu4All, урок про пам'ятки, Ейфелева вежа](./images/startup.png?WT.mc_id=academic-105485-koreyst)

використовуючи промпт на кшталт

> "Собака біля Ейфелевої вежі в ранньому ранковому світлі"

## Що таке DALL-E та Midjourney?

[DALL-E](https://openai.com/dall-e-2?WT.mc_id=academic-105485-koreyst) та [Midjourney](https://www.midjourney.com/?WT.mc_id=academic-105485-koreyst) - це дві з найпопулярніших моделей генерації зображень, які дозволяють використовувати промпти для створення зображень.

### DALL-E

Почнемо з DALL-E, який є моделлю генеративного ШІ, що створює зображення з текстових описів.

> [DALL-E - це комбінація двох моделей, CLIP та дифузної уваги](https://towardsdatascience.com/openais-dall-e-and-clip-101-a-brief-introduction-3a4367280d4e?WT.mc_id=academic-105485-koreyst).

- **CLIP** - це модель, яка генерує вбудовування (embeddings), які є числовими представленнями даних, із зображень та тексту.

- **Дифузна увага** - це модель, яка генерує зображення з вбудовувань. DALL-E навчається на наборі даних зображень та тексту і може використовуватися для генерації зображень з текстових описів. Наприклад, DALL-E можна використовувати для генерації зображень кота в капелюсі або собаки з ірокезом.

### Midjourney

Midjourney працює аналогічно DALL-E, він генерує зображення з текстових промптів. Midjourney також можна використовувати для генерації зображень за допомогою промптів на кшталт "кіт у капелюсі" або "собака з ірокезом".

![Зображення, згенероване Midjourney, механічний голуб](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Rupert_Breheny_mechanical_dove_eca144e7-476d-4976-821d-a49c408e4f36.png/440px-Rupert_Breheny_mechanical_dove_eca144e7-476d-4976-821d-a49c408e4f36.png?WT.mc_id=academic-105485-koreyst)
_Авторство зображення - Wikipedia, зображення згенероване Midjourney_

## Як працюють DALL-E та Midjourney

Спочатку [DALL-E](https://arxiv.org/pdf/2102.12092.pdf?WT.mc_id=academic-105485-koreyst). DALL-E - це модель генеративного ШІ, заснована на архітектурі трансформера з _авторегресивним трансформером_.

_Авторегресивний трансформер_ визначає, як модель генерує зображення з текстових описів, він генерує один піксель за раз, а потім використовує згенеровані пікселі для генерації наступного пікселя. Проходячи через кілька шарів у нейронній мережі, поки зображення не буде завершено.

За допомогою цього процесу DALL-E контролює атрибути, об'єкти, характеристики та інше в згенерованому зображенні. Проте DALL-E 2 та 3 мають більший контроль над згенерованим зображенням.

## Створення вашого першого застосунку для генерації зображень

Отже, що потрібно для створення застосунку для генерації зображень? Вам потрібні наступні бібліотеки:

- **python-dotenv**, настійно рекомендується використовувати цю бібліотеку для зберігання ваших секретів у файлі _.env_ окремо від коду.
- **openai**, ця бібліотека використовується для взаємодії з API OpenAI.
- **pillow**, для роботи із зображеннями в Python.
- **requests**, для допомоги з HTTP-запитами.

1. Створіть файл _.env_ з наступним вмістом:

   ```text
   AZURE_OPENAI_ENDPOINT=<ваша кінцева точка>
   AZURE_OPENAI_API_KEY=<ваш ключ>
   ```

   Знайдіть цю інформацію в Azure Portal для вашого ресурсу в розділі "Keys and Endpoint".

1. Зберіть вищезгадані бібліотеки у файлі _requirements.txt_ таким чином:

   ```text
   python-dotenv
   openai
   pillow
   requests
   ```

1. Далі створіть віртуальне середовище та встановіть бібліотеки:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   Для Windows використовуйте наступні команди для створення та активації віртуального середовища:

   ```bash
   python3 -m venv venv
   venv\Scripts\activate.bat
   ```

1. Додайте наступний код у файл _app.py_:

   ```python
   import openai
   import os
   import requests
   from PIL import Image
   import dotenv

   # import dotenv
   dotenv.load_dotenv()

   # Get endpoint and key from environment variables
   openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
   openai.api_key = os.environ['AZURE_OPENAI_API_KEY']

   # Assign the API version (DALL-E is currently supported for the 2023-06-01-preview API version only)
   openai.api_version = '2023-06-01-preview'
   openai.api_type = 'azure'


   try:
       # Create an image by using the image generation API
       generation_response = openai.Image.create(
           prompt='Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils',    # Enter your prompt text here
           size='1024x1024',
           n=2,
           temperature=0,
       )
       # Set the directory for the stored image
       image_dir = os.path.join(os.curdir, 'images')

       # If the directory doesn't exist, create it
       if not os.path.isdir(image_dir):
           os.mkdir(image_dir)

       # Initialize the image path (note the filetype should be png)
       image_path = os.path.join(image_dir, 'generated-image.png')

       # Retrieve the generated image
       image_url = generation_response["data"][0]["url"]  # extract image URL from response
       generated_image = requests.get(image_url).content  # download the image
       with open(image_path, "wb") as image_file:
           image_file.write(generated_image)

       # Display the image in the default image viewer
       image = Image.open(image_path)
       image.show()

   # catch exceptions
   except openai.InvalidRequestError as err:
       print(err)

   ```

Давайте пояснимо цей код:

- Спочатку ми імпортуємо необхідні бібліотеки, включаючи бібліотеку OpenAI, бібліотеку dotenv, бібліотеку requests та бібліотеку Pillow.

  ```python
  import openai
  import os
  import requests
  from PIL import Image
  import dotenv
  ```

- Далі ми завантажуємо змінні середовища з файлу _.env_.

  ```python
  # import dotenv
  dotenv.load_dotenv()
  ```

- Після цього ми встановлюємо кінцеву точку, ключ для API OpenAI, версію та тип.

  ```python
  # Get endpoint and key from environment variables
  openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
  openai.api_key = os.environ['AZURE_OPENAI_API_KEY']

  # add version and type, Azure specific
  openai.api_version = '2023-06-01-preview'
  openai.api_type = 'azure'
  ```

- Далі ми генеруємо зображення:

  ```python
  # Create an image by using the image generation API
  generation_response = openai.Image.create(
      prompt='Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils',    # Enter your prompt text here
      size='1024x1024',
      n=2,
      temperature=0,
  )
  ```

  Вищенаведений код відповідає JSON-об'єктом, який містить URL згенерованого зображення. Ми можемо використовувати URL для завантаження зображення та збереження його у файл.

- Нарешті, ми відкриваємо зображення та використовуємо стандартний переглядач зображень для його відображення:

  ```python
  image = Image.open(image_path)
  image.show()
  ```

### Більше деталей про генерацію зображення

Розглянемо код, що генерує зображення, більш детально:

```python
generation_response = openai.Image.create(
        prompt='Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils',    # Enter your prompt text here
        size='1024x1024',
        n=2,
        temperature=0,
    )
```

- **prompt** - це текстовий промпт, який використовується для генерації зображення. У цьому випадку ми використовуємо промпт "Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils".
- **size** - це розмір згенерованого зображення. У цьому випадку ми генеруємо зображення розміром 1024x1024 пікселів.
- **n** - це кількість згенерованих зображень. У цьому випадку ми генеруємо два зображення.
- **temperature** - це параметр, який контролює випадковість виводу моделі генеративного ШІ. Температура - це значення між 0 та 1, де 0 означає, що вивід є детермінованим, а 1 означає, що вивід є випадковим. Значення за замовчуванням - 0.7.

Існує більше можливостей для роботи із зображеннями, які ми розглянемо в наступному розділі.

## Додаткові можливості генерації зображень

Ви вже бачили, як ми змогли згенерувати зображення за допомогою кількох рядків Python. Проте є й інші речі, які можна робити із зображеннями.

Ви також можете:

- **Виконувати редагування**. Надаючи існуюче зображення, маску та промпт, ви можете змінити зображення. Наприклад, ви можете додати щось до частини зображення. Уявіть, що у нас є зображення кролика, ви можете додати капелюх до кролика. Для цього вам потрібно надати зображення, маску (яка визначає частину області для зміни) та текстовий промпт, який вказує, що слід зробити.

  ```python
  response = openai.Image.create_edit(
    image=open("base_image.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="An image of a rabbit with a hat on its head.",
    n=1,
    size="1024x1024"
  )
  image_url = response['data'][0]['url']
  ```

  Базове зображення містило б лише кролика, але на кінцевому зображенні був би капелюх на кролику.

- **Створювати варіації**. Ідея полягає в тому, що ви берете існуюче зображення і просите створити його варіації. Щоб створити варіацію, ви надаєте зображення та текстовий промпт і пишете код, як-от:

  ```python
  response = openai.Image.create_variation(
    image=open("bunny-lollipop.png", "rb"),
    n=1,
    size="1024x1024"
  )
  image_url = response['data'][0]['url']
  ```

  > Примітка: це підтримується лише на OpenAI

## Температура

Температура - це параметр, який контролює випадковість виводу моделі генеративного ШІ. Температура - це значення між 0 та 1, де 0 означає, що вивід є детермінованим, а 1 означає, що вивід є випадковим. Значення за замовчуванням - 0.7.

Давайте розглянемо приклад того, як працює температура, запустивши цей промпт двічі:

> Промпт: "Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils"

![Кролик на коні, що тримає льодяник, версія 1](./images/v1-generated-image.png?WT.mc_id=academic-105485-koreyst)

Тепер запустимо той самий промпт ще раз, щоб побачити, що ми не отримаємо однакове зображення двічі:

![Згенероване зображення кролика на коні](./images/v2-generated-image.png?WT.mc_id=academic-105485-koreyst)

Як ви бачите, зображення схожі, але не однакові. Давайте спробуємо змінити значення температури на 0.1 і подивимося, що станеться:

```python
 generation_response = openai.Image.create(
        prompt='Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils',    # Enter your prompt text here
        size='1024x1024',
        n=2
    )
```

### Зміна температури

Отже, давайте спробуємо зробити відповідь більш детермінованою. З двох зображень, які ми згенерували, ми могли спостерігати, що на першому зображенні є кролик, а на другому – кінь, тобто зображення значно відрізняються.

Тому давайте змінимо наш код і встановимо температуру на 0:

```python
generation_response = openai.Image.create(
        prompt='Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils',    # Enter your prompt text here
        size='1024x1024',
        n=2,
        temperature=0
    )
```

Тепер, коли ви запустите цей код, ви отримаєте ці два зображення:

- ![Температура 0, v1](./images/v1-temp-generated-image.png?WT.mc_id=academic-105485-koreyst)
- ![Температура 0, v2](./images/v2-temp-generated-image.png?WT.mc_id=academic-105485-koreyst)

Тут ви можете чітко бачити, як зображення більше схожі одне на одне.

## Як визначити межі для вашого застосунку за допомогою метапромптів

З нашою демонстрацією ми вже можемо генерувати зображення для наших клієнтів. Однак нам потрібно створити деякі межі для нашого застосунку.

Наприклад, ми не хочемо генерувати зображення, які не придатні для роботи або не підходять для дітей.

Ми можемо зробити це за допомогою _метапромптів_. Метапромпти - це текстові підказки, які використовуються для контролю виводу моделі генеративного ШІ. Наприклад, ми можемо використовувати метапромпти для контролю виводу і забезпечення того, щоб згенеровані зображення були придатні для роботи або підходили для дітей.

### Як це працює?

Отже, як працюють метапромпти?

Метапромпти - це текстові підказки, які використовуються для контролю виводу моделі генеративного ШІ, вони розміщуються перед текстовим промптом і використовуються для контролю виводу моделі та вбудовуються в застосунки для контролю виводу моделі. Вони об'єднують введений промпт та метапромпт в єдиний текстовий промпт.

Ось приклад метапромпту:

```text
Ви асистент-дизайнер, який створює зображення для дітей.

Зображення повинно бути придатне для роботи та підходити для дітей.

Зображення повинно бути кольоровим.

Зображення повинно бути в альбомній орієнтації.

Зображення повинно бути в співвідношенні сторін 16:9.

Не враховуйте жодного введення з наступного, яке не придатне для роботи або не підходить для дітей.

(Введення)

```

Тепер давайте подивимося, як ми можемо використовувати метапромпти в нашій демонстрації.

```python
disallow_list = "swords, violence, blood, gore, nudity, sexual content, adult content, adult themes, adult language, adult humor, adult jokes, adult situations, adult"

meta_prompt =f"""You are an assistant designer that creates images for children.

The image needs to be safe for work and appropriate for children.

The image needs to be in color.

The image needs to be in landscape orientation.

The image needs to be in a 16:9 aspect ratio.

Do not consider any input from the following that is not safe for work or appropriate for children.
{disallow_list}
"""

prompt = f"{meta_prompt}
Create an image of a bunny on a horse, holding a lollipop"

# TODO add request to generate image
```

З вищенаведеного промпту ви можете бачити, як усі створені зображення враховують метапромпт.

## Завдання - дозволимо студентам

Ми представили Edu4All на початку цього уроку. Тепер час дозволити студентам генерувати зображення для їхніх завдань.

Студенти створюватимуть зображення для своїх завдань, що містять пам'ятки. Які саме пам'ятки - вирішувати самим студентам. Студентам пропонується використовувати свою креативність у цьому завданні, щоб розміщувати ці пам'ятки в різних контекстах.

## Рішення

Ось одне можливе рішення:

```python
import openai
import os
import requests
from PIL import Image
import dotenv

# import dotenv
dotenv.load_dotenv()

# Get endpoint and key from environment variables
openai.api_base = "<replace with endpoint>"
openai.api_key = "<replace with api key>"

# Assign the API version (DALL-E is currently supported for the 2023-06-01-preview API version only)
openai.api_version = '2023-06-01-preview'
openai.api_type = 'azure'

disallow_list = "swords, violence, blood, gore, nudity, sexual content, adult content, adult themes, adult language, adult humor, adult jokes, adult situations, adult"

meta_prompt = f"""You are an assistant designer that creates images for children.

The image needs to be safe for work and appropriate for children.

The image needs to be in color.

The image needs to be in landscape orientation.

The image needs to be in a 16:9 aspect ratio.

Do not consider any input from the following that is not safe for work or appropriate for children.
{disallow_list}"""

prompt = f"""{metaprompt}
Generate monument of the Arc of Triumph in Paris, France, in the evening light with a small child holding a Teddy looks on.
""""

try:
    # Create an image by using the image generation API
    generation_response = openai.Image.create(
        prompt=prompt,    # Enter your prompt text here
        size='1024x1024',
        n=2,
        temperature=0,
    )
    # Set the directory for the stored image
    image_dir = os.path.join(os.curdir, 'images')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    image_path = os.path.join(image_dir, 'generated-image.png')

    # Retrieve the generated image
    image_url = generation_response["data"][0]["url"]  # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    # Display the image in the default image viewer
    image = Image.open(image_path)
    image.show()

# catch exceptions
except openai.InvalidRequestError as err:
    print(err)
```

## Чудова робота! Продовжуйте навчання

Після завершення цього уроку перегляньте нашу [колекцію навчальних матеріалів з генеративного ШІ](https://aka.ms/genai-collection?WT.mc_id=academic-105485-koreyst), щоб продовжити підвищувати рівень ваших знань з генеративного ШІ!

Переходьте до Уроку 10, де ми розглянемо, як [створювати ШІ-застосунки з низьким кодом](../10-building-low-code-ai-applications/README.md?WT.mc_id=academic-105485-koreyst)