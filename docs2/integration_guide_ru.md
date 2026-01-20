# Руководство по интеграции новых агентов и инструментов

## Введение

Это руководство описывает процесс интеграции новых агентов и инструментов в систему DeepAgent Orchestrator. Следуя этим инструкциям, вы сможете расширить функциональность системы, добавив специализированные компоненты для решения конкретных задач.

## Создание новых агентов

### Формат определения агента

Агенты определяются в YAML-файлах и автоматически загружаются системой. Каждый агент должен содержать следующие поля:

```yaml
name: уникальное_имя_агента
description: "Человекочитаемое описание агента"
system_prompt: |
  Подробные инструкции для агента о том, как он должен себя вести,
  какие задачи выполнять и как взаимодействовать с пользователем.
tools:
  - список_доступных_инструментов
model: "опционально_модель_для_агента"
```

### Пример простого агента

```yaml
# agents/translator_agent.yaml
name: translator_agent
description: "Агент для перевода текстов между языками"
system_prompt: |
  Вы профессиональный переводчик с многолетним опытом.
  Ваша задача - точно и естественно переводить тексты между языками.
  
  При переводе следуйте этим правилам:
  1. Сохраняйте оригинальный смысл и стиль текста
  2. Учитывайте культурные особенности целевой аудитории
  3. Не добавляйте пояснения или комментарии к переводу
  4. Если встречаете непереводимую игру слов, укажите это
  
  Всегда спрашивайте уточняющие вопросы, если текст неясен.
tools:
  - web_search
  - read_file
  - write_file
```

### Пример сложного агента с кастомной моделью

```yaml
# agents/legal_expert.yaml
name: legal_expert
description: "Юридический эксперт для анализа контрактов"
system_prompt: |
  Вы признанный юрист-эксперт в области коммерческих контрактов.
  Ваша специализация - анализ, интерпретация и рекомендации по контрактам.
  
  При анализе контрактов обращайте внимание на:
  1. Рисковые формулировки и неопределенности
  2. Финансовые обязательства и сроки исполнения
  3. Условия расторжения и ответственность сторон
  4. Соответствие законодательству
  
  Всегда предоставляйте конкретные ссылки на статьи контракта.
tools:
  - read_file
  - write_file
  - extract_entities
  - analyze_text_full
model: "openai:gpt-4-legal"  # Специализированная модель
```

### Рекомендации по созданию агентов

#### 1. Четкая специализация
Каждый агент должен иметь четко определенную область компетенции. Избегайте создания универсальных агентов, которые пытаются делать всё сразу.

#### 2. Подробные инструкции
System prompt должен содержать достаточно деталей о том, как агент должен подходить к задачам, какие методы использовать и какие ошибки избегать.

#### 3. Подходящий набор инструментов
Выбирайте инструменты, которые действительно необходимы для выполнения задач агента. Избыточный набор инструментов усложняет работу агента.

#### 4. Тестирование
Перед добавлением агента в production обязательно протестируйте его на различных сценариях использования.

## Создание новых инструментов

### Структура инструмента

Инструменты создаются как Python функции с декоратором `@tool` и регистрируются в системе через `ToolRegistry`.

```python
# src/tools/my_custom_tool.py
from langchain.tools import tool

@tool
def my_custom_tool(input_parameter: str, optional_param: int = 10) -> str:
    """
    Человекочитаемое описание инструмента.
    
    Args:
        input_parameter: Описание обязательного параметра
        optional_param: Описание опционального параметра
        
    Returns:
        Результат выполнения инструмента
    """
    # Реализация инструмента
    result = process_input(input_parameter, optional_param)
    return result
```

### Пример простого инструмента

```python
# src/tools/calculator.py
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Выполняет математические вычисления.
    
    Поддерживает базовые операции: +, -, *, /, **, ()
    
    Args:
        expression: Математическое выражение для вычисления
        
    Returns:
        Результат вычисления в виде строки
        
    Example:
        calculator("2 + 2 * 3") -> "8"
        calculator("(10 + 5) / 3") -> "5.0"
    """
    try:
        # Безопасное вычисление выражения
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Ошибка: выражение содержит недопустимые символы"
        
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"
```

### Пример сложного инструмента с внешним API

```python
# src/tools/weather_tool.py
import os
from langchain.tools import tool
import requests

@tool
def get_weather(city: str) -> str:
    """
    Получает текущую погоду для указанного города.
    
    Использует OpenWeatherMap API для получения актуальной информации.
    
    Args:
        city: Название города на английском языке
        
    Returns:
        Информация о погоде в формате JSON
        
    Note:
        Требуется API ключ OpenWeatherMap в переменной окружения OPENWEATHER_API_KEY
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "Ошибка: не установлен API ключ OpenWeatherMap"
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "ru"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        weather_info = {
            "город": data["name"],
            "температура": f"{data['main']['temp']}°C",
            "ощущается_как": f"{data['main']['feels_like']}°C",
            "влажность": f"{data['main']['humidity']}%",
            "описание": data["weather"][0]["description"],
            "давление": f"{data['main']['pressure']} гПа"
        }
        
        return str(weather_info)
    except requests.RequestException as e:
        return f"Ошибка получения данных о погоде: {str(e)}"
    except KeyError as e:
        return f"Ошибка обработки данных: отсутствует поле {str(e)}"
```

### Регистрация инструмента

После создания инструмента его нужно зарегистрировать в соответствующем модуле:

```python
# src/tools/custom_tools.py
from .registry import ToolRegistry
from .calculator import calculator
from .weather_tool import get_weather

def get_custom_tools(registry: ToolRegistry | None = None) -> ToolRegistry:
    """Регистрирует кастомные инструменты."""
    if registry is None:
        registry = ToolRegistry()
    
    # Регистрация инструментов
    registry.register(
        name="calculator",
        func=calculator,
        description="Выполняет математические вычисления",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Математическое выражение для вычисления"
                }
            },
            "required": ["expression"]
        }
    )
    
    registry.register(
        name="get_weather",
        func=get_weather,
        description="Получает текущую погоду для указанного города",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города на английском языке"
                }
            },
            "required": ["city"]
        }
    )
    
    return registry
```

### Интеграция инструмента в систему

Добавьте регистрацию вашего инструмента в основной файл инструментов:

```python
# src/tools/__init__.py
from .registry import ToolRegistry, get_default_tool_registry
from .search import get_search_tools
from .analysis import get_analysis_tools
from .pattern_tools import get_pattern_tools
from .dependency_tools import get_dependency_tools
from .custom_tools import get_custom_tools  # Добавьте эту строку

def get_all_tools() -> ToolRegistry:
    """Получает все инструменты системы."""
    registry = get_default_tool_registry()
    registry = get_search_tools(registry)
    registry = get_analysis_tools(registry)
    registry = get_pattern_tools(registry)
    registry = get_dependency_tools(registry)
    registry = get_custom_tools(registry)  # Добавьте эту строку
    return registry
```

## Интеграция с внешними системами

### Работа с базами данных

#### Пример интеграции с PostgreSQL

```python
# src/tools/postgres_tool.py
import psycopg2
from langchain.tools import tool

@tool
def execute_postgres_query(query: str, connection_string: str) -> str:
    """
    Выполняет SQL запрос к базе данных PostgreSQL.
    
    Args:
        query: SQL запрос для выполнения
        connection_string: Строка подключения к базе данных
        
    Returns:
        Результаты запроса в формате таблицы
    """
    try:
        conn = psycopg2.connect(connection_string)
        cur = conn.cursor()
        cur.execute(query)
        
        if query.strip().upper().startswith("SELECT"):
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            result = {"columns": columns, "rows": rows}
        else:
            conn.commit()
            result = {"affected_rows": cur.rowcount}
            
        cur.close()
        conn.close()
        return str(result)
    except Exception as e:
        return f"Ошибка выполнения запроса: {str(e)}"
```

### Работа с облачными сервисами

#### Пример интеграции с AWS S3

```python
# src/tools/aws_s3_tool.py
import boto3
from langchain.tools import tool

@tool
def upload_to_s3(file_path: str, bucket: str, key: str) -> str:
    """
    Загружает файл в Amazon S3.
    
    Args:
        file_path: Путь к локальному файлу
        bucket: Имя S3 бакета
        key: Ключ объекта в S3
        
    Returns:
        Результат загрузки
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(file_path, bucket, key)
        return f"Файл успешно загружен в s3://{bucket}/{key}"
    except Exception as e:
        return f"Ошибка загрузки в S3: {str(e)}"
```

## Тестирование новых компонентов

### Unit тесты для инструментов

```python
# tests/tools/test_calculator.py
import pytest
from src.tools.calculator import calculator

def test_calculator_simple_operations():
    """Тест простых математических операций."""
    assert calculator("2 + 2") == "4"
    assert calculator("10 - 5") == "5"
    assert calculator("3 * 4") == "12"
    assert calculator("15 / 3") == "5.0"

def test_calculator_complex_expressions():
    """Тест сложных выражений."""
    assert calculator("2 + 2 * 3") == "8"
    assert calculator("(2 + 2) * 3") == "12"
    assert calculator("2 ** 3") == "8"

def test_calculator_error_handling():
    """Тест обработки ошибок."""
    result = calculator("2 / 0")
    assert "Ошибка" in result
    
    result = calculator("import os")
    assert "недопустимые символы" in result
```

### Интеграционные тесты для агентов

```python
# tests/agents/test_translator_agent.py
import pytest
import yaml
from src.agent_registry import SubAgentRegistry
from src.orchestrator import DeepAgentOrchestrator

@pytest.fixture
def translator_agent():
    """Фикстура с агентом переводчика."""
    agent_yaml = """
    name: translator_agent
    description: "Агент для перевода текстов"
    system_prompt: "Вы профессиональный переводчик"
    tools:
      - web_search
    """
    
    agent_spec = yaml.safe_load(agent_yaml)
    return agent_spec

def test_translator_agent_creation(translator_agent):
    """Тест создания агента переводчика."""
    assert translator_agent["name"] == "translator_agent"
    assert "переводчик" in translator_agent["description"]
    assert len(translator_agent["tools"]) > 0

@pytest.mark.asyncio
async def test_translator_agent_execution(translator_agent):
    """Тест выполнения задачи агентом."""
    orchestrator = DeepAgentOrchestrator()
    # Добавьте тестовую логику здесь
```

## Лучшие практики

### 1. Безопасность

#### Валидация входных данных
```python
@tool
def safe_file_reader(file_path: str) -> str:
    """
    Безопасно читает файл с проверкой пути.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Содержимое файла или сообщение об ошибке
    """
    import os
    from pathlib import Path
    
    # Проверка на относительные пути
    if ".." in file_path:
        return "Ошибка: запрещено использовать относительные пути"
    
    # Ограничение рабочей директории
    workspace = Path("/tmp/agent-workspace")
    full_path = (workspace / file_path).resolve()
    
    if not str(full_path).startswith(str(workspace)):
        return "Ошибка: доступ ограничен рабочей директорией"
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Ошибка чтения файла: {str(e)}"
```

### 2. Обработка ошибок

#### Грациозная деградация
```python
@tool
def resilient_web_search(query: str) -> str:
    """
    Поиск в интернете с обработкой ошибок.
    
    Args:
        query: Поисковый запрос
        
    Returns:
        Результаты поиска или сообщение об ошибке
    """
    import os
    import time
    
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Поиск недоступен: не установлен API ключ"
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        results = client.search(query, max_results=5)
        return str(results)
    except ImportError:
        return "Поиск недоступен: требуется установка tavily-python"
    except Exception as e:
        # Повторная попытка через 1 секунду
        time.sleep(1)
        try:
            results = client.search(query, max_results=3)
            return f"Результаты (после повторной попытки): {results}"
        except Exception as e2:
            return f"Ошибка поиска: {str(e)}. Повторная попытка также не удалась: {str(e2)}"
```

### 3. Логирование

#### Подробное логирование для отладки
```python
import logging
from langchain.tools import tool

logger = logging.getLogger(__name__)

@tool
def debuggable_tool(input_data: str) -> str:
    """
    Инструмент с подробным логированием.
    
    Args:
        input_data: Входные данные для обработки
        
    Returns:
        Результат обработки
    """
    logger.info(f"Начало выполнения инструмента с данными: {input_data[:50]}...")
    
    try:
        # Симуляция обработки
        result = process_data(input_data)
        logger.debug(f"Промежуточный результат: {result}")
        
        final_result = format_result(result)
        logger.info(f"Инструмент выполнен успешно. Размер результата: {len(final_result)}")
        return final_result
    except Exception as e:
        logger.error(f"Ошибка выполнения инструмента: {str(e)}", exc_info=True)
        return f"Ошибка: {str(e)}"
```

## Отладка и диагностика

### Использование событийной системы

```python
from src.events import EventType, create_event

@tool
def instrumented_tool(data: str) -> str:
    """
    Инструмент с интеграцией событийной системы.
    
    Args:
        data: Входные данные
        
    Returns:
        Результат обработки
    """
    # Генерация события начала обработки
    event_bus.publish(create_event(
        EventType.TASK_STARTED,
        payload={"tool": "instrumented_tool", "data_size": len(data)}
    ))
    
    try:
        result = process_data(data)
        
        # Генерация события успешного завершения
        event_bus.publish(create_event(
            EventType.TASK_COMPLETED,
            payload={"tool": "instrumented_tool", "result_size": len(result)}
        ))
        
        return result
    except Exception as e:
        # Генерация события ошибки
        event_bus.publish(create_event(
            EventType.TASK_FAILED,
            payload={"tool": "instrumented_tool", "error": str(e)}
        ))
        raise
```

## Производительность

### Кэширование результатов

```python
from functools import lru_cache
import time

@tool
def cached_computation(complex_input: str) -> str:
    """
    Инструмент с кэшированием результатов.
    
    Args:
        complex_input: Сложные входные данные
        
    Returns:
        Результат вычислений
    """
    return _expensive_computation(complex_input)

@lru_cache(maxsize=128)
def _expensive_computation(input_data: str) -> str:
    """Кэшируемая функция для дорогих вычислений."""
    # Симуляция дорогих вычислений
    time.sleep(2)
    return f"Результат для {input_data}"
```

## Заключение

Интеграция новых агентов и инструментов в DeepAgent Orchestrator открывает широкие возможности для расширения функциональности системы. Следуя этим рекомендациям, вы сможете создавать качественные, безопасные и эффективные компоненты, которые будут полезны как для автоматических процессов, так и для взаимодействия с пользователем.

Помните о важности тестирования, документирования и соблюдения стандартов кодирования при создании новых компонентов. Это обеспечит стабильность и поддерживаемость всей системы в долгосрочной перспективе.