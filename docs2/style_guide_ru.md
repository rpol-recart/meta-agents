# Стилевое руководство и правила проекта DeepAgent Orchestrator

## Введение

Этот документ определяет стандарты кодирования, правила проекта и лучшие практики для разработчиков, работающих над DeepAgent Orchestrator. Следование этим правилам обеспечивает согласованность, читаемость и поддерживаемость кода во всем проекте.

## Стилевые соглашения

### Именование

#### Файлы и директории
- Используйте `snake_case` для имен файлов и директорий
- Пример: `agent_registry.py`, `orchestrator_factory.py`

#### Классы
- Используйте `PascalCase` для имен классов
- Пример: `DeepAgentOrchestrator`, `OrchestratorConfig`

#### Функции и переменные
- Используйте `snake_case` для имен функций и переменных
- Пример: `create_agent`, `model_service`, `thread_id`

#### Константы
- Используйте `SCREAMING_SNAKE_CASE` для констант
- Пример: `DEFAULT_MODEL_FALLBACK`, `MAX_RETRY_ATTEMPTS`

#### Булевы переменные
- Используйте префиксы `is_`, `has_`, `should_` для булевых переменных
- Пример: `is_enabled`, `has_memory`, `should_reload`

### Типизация

#### Аннотации типов
- Все публичные функции и методы должны иметь аннотации типов
- Используйте современный синтаксис: `str | None` вместо `Optional[str]`
- Используйте `list[T]` и `dict[K, V]` вместо `List[T]` и `Dict[K, V]`

#### Примеры аннотаций:
```python
# Хорошо
def create_agent(
    model: Any,
    system_prompt: str | None = None,
    tools: list[str] | None = None
) -> CompiledAgent:
    pass

# Плохо
def create_agent(model, system_prompt=None, tools=None):
    pass
```

### Импорты

#### Порядок импортов
1. Стандартная библиотека
2. Сторонние библиотеки
3. Локальные модули

#### Пример:
```python
# stdlib
import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# third-party
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langgraph.prebuilt.tool_node import ToolRuntime

# local
from .interfaces import ModelInterface, AgentInterface, TaskInterface
from .services import ModelService, AgentService, TaskService
```

#### Рекомендации:
- Используйте относительные импорты для внутренних модулей
- Избегайте wildcard импортов (`from module import *`)
- Группируйте связанные импорты

### Форматирование

#### Отступы и пробелы
- Используйте 4 пробела для отступов (не табуляцию)
- Добавляйте один пробел после запятых
- Добавляйте пробелы вокруг операторов сравнения и присваивания

#### Длина строк
- Максимальная длина строки: 100 символов
- Для длинных строк используйте скобки для переноса:

```python
# Хорошо
long_string = (
    "Это очень длинная строка, которая должна быть "
    "разделена на несколько строк для лучшей читаемости"
)

# Плохо
long_string = "Это очень длинная строка, которая должна быть разделена на несколько строк для лучшей читаемости, но она слишком длинная и нечитаемая"
```

#### Кавычки
- Используйте двойные кавычки для строк
- Используйте одинарные кавычки для строк внутри f-строк

```python
# Хорошо
message = "Привет, мир!"
query = f"SELECT * FROM users WHERE name = '{user_name}'"

# Плохо
message = 'Привет, мир!'
```

#### Пустые строки
- Две пустые строки между классами верхнего уровня
- Одна пустая строка между методами класса
- Одна пустая строка между группами импортов

## Документация

### Docstrings

#### Формат
Используйте Google-style docstrings:

```python
def create_agent(
    model: Any,
    system_prompt: str | None = None,
    tools: list[str] | None = None
) -> CompiledAgent:
    """
    Создает нового агента с указанными параметрами.

    Args:
        model: Языковая модель для агента
        system_prompt: Системный промпт для агента
        tools: Список доступных инструментов

    Returns:
        Скомпилированный агент

    Raises:
        ValueError: Если модель не предоставлена
        ConfigurationError: Если конфигурация некорректна

    Example:
        >>> model = init_chat_model("anthropic:claude-sonnet-4-20250514")
        >>> agent = create_agent(model, "Вы эксперт по анализу данных")
    """
```

#### Обязательные элементы
1. Краткое описание функции/класса
2. Описание параметров (Args)
3. Описание возвращаемого значения (Returns)
4. Описание исключений (Raises) если есть
5. Примеры использования (Example) если необходимо

### Комментарии

#### Общие правила
- Пишите комментарии на русском языке
- Комментарии должны объяснять "почему", а не "что"
- Избегайте очевидных комментариев

#### Хорошие примеры:
```python
# Хорошо - объясняет почему
# Используем временную директорию для избежания конфликтов с другими процессами
temp_dir = Path("/tmp/agent-workspace")

# Хорошо - объясняет сложную логику
# Алгоритм Бойера-Мура для быстрого поиска подстроки
def boyer_moore_search(text: str, pattern: str) -> int:
    pass
```

#### Плохие примеры:
```python
# Плохо - очевидный комментарий
x = x + 1  # Увеличиваем x на 1

# Плохо - комментарий на английском в русскоязычном проекте
# Initialize the model service
model_service = ModelService()
```

## Обработка ошибок

### Исключения

#### Создание пользовательских исключений
```python
class OrchestratorError(Exception):
    """Базовое исключение для оркестратора."""
    pass

class ConfigurationError(OrchestratorError):
    """Ошибка конфигурации."""
    
    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(message)
        self.config_key = config_key
```

#### Обработка исключений
```python
# Хорошо - конкретная обработка
try:
    result = await self.agent.ainvoke(inputs, config=config)
    logger.info(f"Задача выполнена для потока: {thread_id}")
    return result
except ValidationError as e:
    logger.error(f"Ошибка валидации конфигурации: {e}")
    raise ConfigurationError(f"Некорректная конфигурация: {e}") from e
except Exception as e:
    logger.error(f"Неожиданная ошибка выполнения задачи: {e}")
    raise OrchestratorError(f"Ошибка выполнения задачи: {e}") from e

# Плохо - общая обработка без логирования
try:
    result = await self.agent.ainvoke(inputs, config=config)
    return result
except Exception:
    return None
```

### Логирование

#### Уровни логирования
- `DEBUG` - Подробная отладочная информация
- `INFO` - Общая информация о ходе выполнения
- `WARNING` - Предупреждения о потенциально проблемных ситуациях
- `ERROR` - Ошибки, препятствующие нормальному выполнению
- `CRITICAL` - Критические ошибки, требующие немедленного внимания

#### Примеры использования:
```python
import logging

logger = logging.getLogger(__name__)

# INFO - нормальные события
logger.info("Оркестратор инициализирован с моделью: %s", self.config.model_name)

# DEBUG - подробная информация для отладки
logger.debug("Создан агент с параметрами: %s", agent_params)

# WARNING - потенциально проблемные ситуации
logger.warning("Используется модель по умолчанию, рекомендуется явно указать модель")

# ERROR - ошибки выполнения
logger.error("Не удалось инициализировать модель: %s", e)

# CRITICAL - критические ошибки
logger.critical("Критическая ошибка инициализации системы: %s", e)
```

## Асинхронное программирование

### Паттерны использования

#### Асинхронные функции
```python
# Хорошо - async def для I/O операций
async def run_task(self, task: str, thread_id: str | None = None) -> dict[str, Any]:
    """Выполняет задачу асинхронно."""
    try:
        result = await self._task_service.execute(
            task=task,
            agent=self.agent,
            thread_id=thread_id
        )
        return result
    except Exception as e:
        logger.error("Ошибка выполнения задачи: %s", e)
        raise
```

#### Асинхронные генераторы
```python
async def stream(
    self, task: str, thread_id: str | None = None
) -> AsyncGenerator[dict[str, Any], None]:
    """Стриминг выполнения задачи."""
    try:
        async for chunk in self._task_service.stream(
            task=task,
            agent=self.agent,
            thread_id=thread_id
        ):
            yield chunk
    except Exception as e:
        logger.error("Ошибка стриминга: %s", e)
        raise
```

#### Параллельное выполнение
```python
import asyncio

# Хорошо - параллельное выполнение задач
async def run_parallel_tasks(self, tasks: list[str]) -> list[dict]:
    """Выполняет несколько задач параллельно."""
    coroutines = [
        self.run(task, thread_id=f"parallel-{i}")
        for i, task in enumerate(tasks)
    ]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    return results
```

## Тестирование

### Структура тестов

#### Организация файлов
```
tests/
├── conftest.py              # Общие фикстуры
├── test_orchestrator.py     # Тесты оркестратора
├── test_config.py           # Тесты конфигурации
├── test_services/           # Тесты сервисов
│   ├── test_model_service.py
│   ├── test_agent_service.py
│   └── test_task_service.py
└── test_tools/              # Тесты инструментов
    ├── test_search_tools.py
    └── test_analysis_tools.py
```

#### Фикстуры
```python
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.fixture
def mock_model():
    """Мock объект модели для тестирования."""
    return Mock()

@pytest.fixture
def mock_agent_service():
    """Mock сервис агентов."""
    service = Mock()
    service.create_agent = Mock(return_value=Mock())
    return service

@pytest.fixture
async def orchestrator_with_mocks(mock_model, mock_agent_service):
    """Оркестратор с mock сервисами."""
    from src.orchestrator import DeepAgentOrchestrator
    
    orchestrator = DeepAgentOrchestrator(
        model_name="test-model",
        model_service=Mock(),
        agent_service=mock_agent_service,
        task_service=Mock()
    )
    return orchestrator
```

### Написание тестов

#### AAA паттерн (Arrange, Act, Assert)
```python
@pytest.mark.asyncio
async def test_orchestrator_run_success(orchestrator_with_mocks):
    """Тест успешного выполнения задачи."""
    # Arrange
    orchestrator = orchestrator_with_mocks
    test_task = "Проанализируйте этот текст"
    expected_result = {"messages": [{"content": "Анализ завершен"}]}
    
    orchestrator._task_service.execute = AsyncMock(return_value=expected_result)
    
    # Act
    result = await orchestrator.run(test_task)
    
    # Assert
    assert result == expected_result
    orchestrator._task_service.execute.assert_called_once_with(
        task=test_task,
        agent=orchestrator.agent,
        thread_id=None
    )
```

#### Тестирование исключений
```python
@pytest.mark.asyncio
async def test_orchestrator_run_failure(orchestrator_with_mocks):
    """Тест обработки ошибок при выполнении задачи."""
    # Arrange
    orchestrator = orchestrator_with_mocks
    test_task = "Проанализируйте этот текст"
    error_message = "Ошибка модели"
    
    orchestrator._task_service.execute = AsyncMock(
        side_effect=Exception(error_message)
    )
    
    # Act & Assert
    with pytest.raises(Exception, match=error_message):
        await orchestrator.run(test_task)
```

## Git и управление версиями

### Commit сообщения

#### Формат conventional commits
```
<type>(<scope>): <description>

[body]

[footer]
```

#### Типы коммитов:
- `feat`: Новая функциональность
- `fix`: Исправление ошибок
- `docs`: Изменения в документации
- `style`: Изменения форматирования
- `refactor`: Рефакторинг кода
- `test`: Добавление или изменение тестов
- `chore`: Технические изменения

#### Примеры:
```
feat(orchestrator): добавить поддержку горячей перезагрузки агентов

Добавлена возможность автоматической перезагрузки агентов
при изменении их конфигурационных файлов.

Closes #123
```

```
fix(config): исправить загрузку переменных окружения

Исправлена ошибка при загрузке переменных окружения,
которая приводила к игнорированию API ключей.

Fixes #456
```

### Ветвление

#### Стратегия ветвления
- `main` - основная ветка с production кодом
- `develop` - ветка разработки
- `feature/*` - ветки для новых функций
- `bugfix/*` - ветки для исправления ошибок
- `release/*` - ветки для подготовки релизов

#### Примеры именования:
```
feature/add-s3-integration
bugfix/fix-memory-leak
release/v1.2.0
```

## Безопасность

### Обработка секретов

#### Переменные окружения
```python
# Хорошо - использование переменных окружения
import os

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ConfigurationError("Требуется ANTHROPIC_API_KEY")

# Плохо - хардкодинг секретов
api_key = "sk-ant-1234567890abcdef"  # Никогда так не делайте!
```

#### Валидация входных данных
```python
from pydantic import BaseModel, validator

class AgentConfig(BaseModel):
    name: str
    model: str
    tools: list[str]
    
    @validator("name")
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Имя агента не может быть пустым")
        if len(v) > 50:
            raise ValueError("Имя агента не может превышать 50 символов")
        return v.strip()
```

## Производительность

### Оптимизация памяти

#### Избегайте утечек памяти
```python
# Хорошо - явная очистка ресурсов
class ResourceManager:
    def __init__(self):
        self.resources = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        for resource in self.resources:
            if hasattr(resource, 'close'):
                resource.close()
        self.resources.clear()
```

#### Использование генераторов
```python
# Хорошо - использование генераторов для больших данных
def process_large_dataset(data_source):
    """Обрабатывает большой набор данных без загрузки всего в память."""
    for item in data_source:
        yield process_item(item)

# Плохо - загрузка всего в память
def process_large_dataset_bad(data_source):
    """Плохой пример - загружает все данные в память."""
    all_data = list(data_source)  # Может вызвать OutOfMemoryError
    return [process_item(item) for item in all_data]
```

## Совместимость

### Поддержка версий Python

#### Требования
- Минимальная версия: Python 3.10
- Рекомендуемая версия: Python 3.11+
- Использование современных возможностей языка

#### Примеры совместимости:
```python
# Хорошо - использование современного синтаксиса
from typing import Any  # Python 3.10+

def process_data(data: dict[str, Any]) -> list[str]:  # Union syntax
    return [item for item in data.values() if isinstance(item, str)]

# Использование match-case (Python 3.10+)
match status:
    case "success":
        handle_success()
    case "error":
        handle_error()
    case _:
        handle_unknown()
```

## Заключение

Следование этим правилам и соглашениям помогает поддерживать высокое качество кода, упрощает совместную разработку и обеспечивает долгосрочную поддерживаемость проекта. Все участники проекта должны ознакомиться с этими правилами и применять их в своей работе.

Периодически эти правила могут обновляться для соответствия лучшим практикам отрасли и специфике проекта. Все изменения в правилах должны быть задокументированы и согласованы с командой разработчиков.