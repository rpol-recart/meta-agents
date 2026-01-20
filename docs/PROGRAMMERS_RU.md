# DeepAgent Orchestrator - Документация для разработчиков

## Содержание

1. [Обзор архитектуры](#обзор-архитектуры)
2. [Структура проекта](#структура-проекта)
3. [Основные компоненты](#основные-компоненты)
4. [Dependency Injection](#dependency-injection)
5. [Конфигурация](#конфигурация)
6. [Сервисы](#сервисы)
7. [Событийная система](#событийная-система)
8. [Мониторинг и observability](#мониторинг-и-observability)
9. [Тестирование](#тестирование)
10. [Разработка и внесение изменений](#разработка-и-внесение-изменений)

---

## Обзор архитектуры

DeepAgent Orchestrator — это система управления multi-agent на основе LangChain, которая предоставляет:

- Параллельное выполнение под-агентов
- Горячую перезагрузку конфигураций агентов
- CLI и REST API интерфейсы
- Встроенное планирование и файловые операции
- Human-in-the-loop (HITL) workflow

### Ключевые принципы архитектуры

1. **Модульность**: Система разделена на независимые модули с чёткими границами
2. **Dependency Injection**: Компоненты получают зависимости через конструктор
3. **Событийная коммуникация**: Loose coupling через publish-subscribe модель
4. **Конфигурация как код**: Все настройки определяются в YAML файлах
5. **Observability**: Встроенный мониторинг, метрики и health checks

---

## Структура проекта

```
src/
├── __init__.py              # Главный модуль пакета
├── orchestrator.py          # Основной оркестратор (DeepAgentOrchestrator)
├── cli.py                   # CLI интерфейс (Click)
├── api.py                   # REST API (FastAPI)
├── config.py                # Централизованная конфигурация
├── agent_registry.py        # Реестр агентов с hot-reload
├── exceptions.py            # Иерархия исключений
│
├── interfaces/              # Абстрактные интерфейсы
│   ├── __init__.py
│   ├── model_interface.py   # Интерфейс модели
│   ├── agent_interface.py   # Интерфейс агента
│   ├── task_interface.py    # Интерфейс задачи
│   ├── backend_interface.py # Интерфейс бэкенда
│   └── storage_interface.py # Интерфейс хранилища
│
├── services/                # Реализации сервисов
│   ├── __init__.py
│   ├── model_service.py     # Инициализация модели
│   ├── agent_service.py     # Создание агентов
│   ├── task_service.py      # Выполнение задач
│   └── agent_loader.py      # Загрузка агентов из YAML
│
├── factories/               # Фабрики для создания компонентов
│   ├── __init__.py
│   └── orchestrator_factory.py
│
├── events/                  # Событийная система
│   └── __init__.py          # EventBus, Event, EventEmitter
│
├── monitoring/              # Мониторинг и observability
│   └── __init__.py          # Metrics, HealthChecker
│
└── tools/                   # Инструменты
    ├── __init__.py
    ├── registry.py          # Реестр инструментов
    └── search.py            # Инструменты поиска
```

---

## Основные компоненты

### DeepAgentOrchestrator

Основной класс оркестратора, который управляет всем жизненным циклом агентов.

```python
from src.orchestrator import DeepAgentOrchestrator

# Создание с использованием сервисов по умолчанию
orchestrator = DeepAgentOrchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    system_prompt="Ты полезный ассистент",
    subagents=[],  # Список под-агентов
    enable_memory=False,
)

# Dependency Injection для тестирования
from src.services import ModelService, AgentService, TaskService

orchestrator = DeepAgentOrchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    model_service=ModelService(),
    agent_service=AgentService(),
    task_service=TaskService(),
)

# Выполнение задачи
result = await orchestrator.run("Проанализируй этот текст")
```

### OrchestratorConfig

Конфигурация оркестратора с поддержкой dataclass.

```python
from src.orchestrator import OrchestratorConfig

config = OrchestratorConfig(
    model_name="anthropic:claude-sonnet-4-20250514",
    system_prompt="Custom prompt",
    subagents=[{"name": "researcher", "description": "..."}],
    enable_memory=False,
    workspace_dir="/tmp/agent-workspace",
)

# Создание из централизованных настроек
from src.config import load_settings

settings = load_settings("config.yaml")
config = OrchestratorConfig.from_settings(settings)
```

---

## Dependency Injection

### Доступные интерфейсы

```python
from src.interfaces import (
    ModelInterface,    # Инициализация модели
    AgentInterface,    # Создание агентов
    TaskInterface,     # Выполнение задач
    BackendInterface,  # Бэкенд операции
    StorageInterface,  # Операции хранилища
)
```

### Использование сервисов

```python
from src.services import ModelService, AgentService, TaskService
from src.orchestrator import DeepAgentOrchestrator

# Создание сервисов
model_service = ModelService()
agent_service = AgentService()
task_service = TaskService()

# Инициализация модели напрямую
model = model_service.initialize(
    model_name="anthropic:claude-sonnet-4-20250514",
    api_key="your-api-key",
)

# Создание агента
agent = agent_service.create_agent(
    model=model,
    system_prompt="Ты эксперт по анализу",
    subagents=[],
)

# Выполнение задачи
result = await task_service.execute(
    task="Проанализируй данные",
    agent=agent,
)
```

### OrchestratorFactory

Фабрика для создания оркестраторов с DI.

```python
from src.factories import OrchestratorFactory, create_orchestrator

# Использование фабрики
factory = OrchestratorFactory()
orchestrator = factory.create(
    model_name="anthropic:claude-sonnet-4-20250514",
    agents_dir="agents/",
    enable_memory=False,
)

# Удобная функция
orchestrator = create_orchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    agents_dir="agents/",
)

# DI кастомных сервисов
custom_model_service = CustomModelService()
orchestrator = factory.with_model_service(custom_model_service).create(
    model_name="custom-model",
)
```

---

## Конфигурация

### Settings (Pydantic)

```python
from src.config import Settings, ModelConfig, BackendConfig, HITLConfig

# Конфигурация по умолчанию
settings = Settings()

# Кастомная конфигурация
settings = Settings(
    model=ModelConfig(
        provider="anthropic",
        name="anthropic:claude-sonnet-4-20250514",
        api_key="your-api-key",
        base_url=None,
        temperature=0.1,
    ),
    backend=BackendConfig(
        root_dir="/tmp/agent-workspace",
        store_namespace="memories",
    ),
    hitl=HITLConfig(
        enabled=False,
        tools={},
    ),
)

# Сохранение конфигурации
settings.save("config.yaml")

# Загрузка конфигурации
settings = Settings.load("config.yaml")
```

### Переменные окружения

```python
# Модель по умолчанию
export DEFAULT_MODEL="anthropic:claude-sonnet-4-20250514"

# API ключ
export OPENAI_API_KEY="your-api-key"

# Базовый URL для совместимых API
export OPENAI_BASE_URL="https://api.example.com/v1"
```

---

## Сервисы

### ModelService

Сервис для инициализации LLM моделей.

```python
from src.services import ModelService

service = ModelService()

# Валидация имени модели
is_valid = service.validate_model_name("anthropic:claude-sonnet-4-20250514")

# Получение доступных моделей
models = service.get_available_models()
# {
#     "anthropic:claude-sonnet-4-20250514": {
#         "provider": "anthropic",
#         "name": "Claude Sonnet 4",
#     },
# }

# Инициализация модели
model = service.initialize(
    model_name="anthropic:claude-sonnet-4-20250514",
    api_key="your-api-key",
    base_url=None,
    temperature=0.1,
)
```

### AgentService

Сервис для создания агентов.

```python
from src.services import AgentService

service = AgentService()

# Создание агента
agent = service.create_agent(
    model=model,
    system_prompt="Ты эксперт по анализу данных",
    subagents=[],  # Список под-агентов
    custom_tools=[],
    interrupt_on={"write_file": {"allowed_decisions": ["approve", "reject"]}},
    backend=backend_factory,
)

# Управление под-агентами
agent = service.add_subagent(agent, {"name": "researcher", ...})
agent = service.remove_subagent(agent, "researcher")
subagents = service.get_subagents(agent)
```

### TaskService

Сервис для выполнения задач.

```python
from src.services import TaskService

service = TaskService()

# Синхронное выполнение
result = await service.execute(
    task="Проанализируй текст",
    agent=agent,
    thread_id="optional-thread-id",
)

# Стриминг результатов
async for chunk in service.stream(
    task="Проанализируй текст",
    agent=agent,
):
    print(chunk)

# Стриминг событий
async for event in service.stream_events(
    task="Проанализируй текст",
    agent=agent,
):
    print(event)

# Управление состоянием
state = await service.get_state(agent, "thread-id")
cleared = await service.clear_state(agent, "thread-id")
```

### AgentLoader

Загрузчик агентов из YAML файлов.

```python
from src.services import AgentLoader, load_agents_from_dir

# Создание загрузчика
loader = AgentLoader(agents_dir="agents/")

# Регистрация кастомных инструментов
loader.register_tool("my_tool", my_tool_function, "Description")

# Загрузка агентов из директории
agents = loader.load_from_directory("agents/")

# Загрузка одного агента
agent = loader.load_from_file("agents/researcher.yaml")

# Валидация агента
errors = loader.validate_agent(agent_data)
# [] если валиден, иначе список ошибок
```

---

## Событийная система

### EventBus

Центральная шина событий для loose coupling.

```python
from src.events import EventBus, Event, EventType, create_event

# Создание шины событий
event_bus = EventBus()

# Подписка на тип событий
async def on_task_completed(event: Event):
    print(f"Task completed: {event.payload}")

event_bus.subscribe(EventType.TASK_COMPLETED, on_task_completed)

# Подписка на все события
async def on_all_events(event: Event):
    print(f"Event: {event.type.value}")

event_bus.subscribe_all(on_all_events)

# Публикация события
event = create_event(
    EventType.TASK_COMPLETED,
    payload={"task": "analysis", "result": "success"},
)
await event_bus.publish(event)

# Отписка
event_bus.unsubscribe(EventType.TASK_COMPLETED, on_task_completed)
```

### EventEmitter

Миксин для классов, которые генерируют события.

```python
from src.events import EventEmitter, Event, EventType

class MyComponent(EventEmitter):
    async def do_something(self):
        await self.emit(Event(
            type=EventType.TASK_STARTED,
            payload={"component": "my_component"},
        ))

# Использование
component = MyComponent()
component.set_event_bus(event_bus)
await component.do_something()
```

### Типы событий

```python
from src.events import EventType

EventType.TASK_STARTED        # Задача начата
EventType.TASK_COMPLETED      # Задача завершена
EventType.TASK_FAILED         # Задача не выполнена
EventType.AGENT_CREATED       # Агент создан
EventType.SUBAGENT_ADDED      # Под-агент добавлен
EventType.SUBAGENT_REMOVED    # Под-агент удалён
EventType.HITL_INTERRUPT      # HITL прерывание
EventType.STATE_CLEARED       # Состояние очищено
EventType.ERROR_OCCURRED      # Ошибка
EventType.MODEL_INITIALIZED   # Модель инициализирована
EventType.CONFIG_CHANGED      # Конфигурация изменена
```

---

## Мониторинг и observability

### MetricsCollector

Сборщик метрик производительности.

```python
from src.monitoring import MetricsCollector

collector = MetricsCollector()

# Запись событий
collector.record_task_start("thread-123")
collector.record_task_complete("thread-123", 1500.0)  # мс
collector.record_task_failure("thread-123", 500.0)

# Получение метрик
metrics = collector.get_metrics()
print(f"Task count: {metrics.task_count}")
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg duration: {metrics.average_task_duration_ms:.2f}ms")
```

### HealthChecker

Система health checks.

```python
from src.monitoring import HealthChecker, HealthStatus

checker = HealthChecker()

# Регистрация компонентов
async def check_model():
    from src.monitoring import HealthCheck
    return HealthCheck(
        status=HealthStatus.HEALTHY,
        component="model",
        message="Model loaded",
    )

checker.register_component("model", check_model)

# Проверка всех компонентов
checks = await checker.check_all()
for check in checks:
    print(f"{check.component}: {check.status.value}")

# Общий статус
status = checker.get_overall_status(checks)
```

### ObservabilityManager

Центральный менеджер для всех функций observability.

```python
from src.monitoring import ObservabilityManager

obs = ObservabilityManager()
obs.enable()

# Запись метрик
obs.record_task_start("thread-123")
obs.record_task_complete("thread-123", 1500.0)

# Получение метрик
metrics = obs.get_metrics()

# Health checks
checks = await obs.check_health()
```

---

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest tests/ -v

# Конкретный файл
pytest tests/test_orchestrator.py -v

# Конкретный тест
pytest tests/test_orchestrator.py::TestOrchestratorConfig::test_defaults -v

# С покрытием
pytest --cov=src --cov-report=term-missing
```

### Структура тестов

```
tests/
├── conftest.py              # Фикстуры pytest
├── test_orchestrator.py     # Тесты оркестратора
├── test_config.py           # Тесты конфигурации
├── test_api.py              # Тесты API
├── test_cli.py              # Тесты CLI
├── test_agent_registry.py   # Тесты реестра агентов
├── test_tools.py            # Тесты инструментов
└── test_orchestrator_factory.py  # Тесты фабрики
```

### Пример теста с DI

```python
import pytest
from src.orchestrator import DeepAgentOrchestrator
from src.services import ModelService, AgentService, TaskService

@pytest.fixture
def mock_services():
    return ModelService(), AgentService(), TaskService()

@pytest.mark.asyncio
async def test_orchestrator_with_custom_services(mock_services):
    model_service, agent_service, task_service = mock_services

    orchestrator = DeepAgentOrchestrator(
        model_name="test-model",
        model_service=model_service,
        agent_service=agent_service,
        task_service=task_service,
    )

    assert orchestrator.model_service is model_service
    assert orchestrator.agent_service is agent_service
    assert orchestrator.task_service is task_service
```

---

## Разработка и внесение изменений

### Соглашения по коду

1. **Именование**:
   - Файлы: `snake_case.py`
   - Классы: `PascalCase`
   - Функции/переменные: `snake_case`
   - Константы: `SCREAMING_SNAKE_CASE`
   - Булевы: `is_*`, `has_*`, `should_*`

2. **Типизация**:
   - Использовать `X | None` вместо `Optional[X]`
   - Использовать `list[T]` и `dict[K, V]` из typing
   - Все публичные функции должны иметь аннотации типов

3. **Импорты**:
   ```python
   # stdlib
   import asyncio
   import logging
   from pathlib import Path

   # third-party
   from langchain.chat_models import init_chat_model
   from pydantic import BaseModel

   # local
   from .orchestrator import DeepAgentOrchestrator
   from .interfaces import ModelInterface
   ```

4. **Форматирование**:
   - 4 пробела для отступа
   - Максимальная длина строки: 100 символов
   - Двойные кавычки для строк
   - Одна пустая строка между объявлениями верхнего уровня

### Создание нового сервиса

1. Определить интерфейс в `src/interfaces/`:

```python
# src/interfaces/custom_interface.py
from abc import ABC, abstractmethod
from typing import Any

class CustomInterface(ABC):
    @abstractmethod
    def do_something(self, value: str) -> Any:
        pass
```

2. Реализовать в `src/services/`:

```python
# src/services/custom_service.py
from ..interfaces import CustomInterface
from ..exceptions import CustomError

class CustomService(CustomInterface):
    def do_something(self, value: str) -> Any:
        try:
            result = self._process(value)
            return result
        except Exception as e:
            raise CustomError(message=f"Failed: {e}") from e
```

3. Экспортировать в `src/services/__init__.py`:

```python
from .custom_service import CustomService

__all__ = [..., "CustomService"]
```

### Создание нового инструмента

1. Создать файл в `src/tools/`:

```python
# src/tools/my_tool.py
from langchain.tools import tool

@tool
def my_useful_tool(input_text: str) -> str:
    """Полезное описание инструмента."""
    # Реализация
    return result
```

2. Зарегистрировать в `src/tools/registry.py`:

```python
from .my_tool import my_useful_tool

DEFAULT_TOOLS = [
    my_useful_tool,
    # ...
]
```

### Линтинг и форматирование

```bash
# Форматирование
black src/ tests/

# Проверка форматирования
black src/ tests/ --check

# Линтинг
ruff check src/ tests/

# Автоисправление линтинга
ruff check --fix src/ tests/

# Проверка типов
mypy src/ tests/
```

### Git commit

Используйте conventional commits:

```
feat: добавить новый инструмент поиска
fix: исправить ошибку в TaskService
docs: обновить документацию
refactor: рефакторинг сервисов
test: добавить тесты для компонента
```

---

## Частые задачи

### Добавление нового под-агента

```python
orchestrator.add_subagent(
    name="researcher",
    description="Агент для исследования",
    system_prompt="Ты эксперт по поиску информации...",
    tools=["search", "read_file"],
    model="anthropic:claude-sonnet-4-20250514",
)
```

### Настройка HITL

```python
orchestrator.configure_hitl(
    tool_name="write_file",
    allowed_decisions=["approve", "edit", "reject"],
)
```

### Работа с состоянием

```python
# Получение состояния
state = orchestrator.get_state("thread-123")

# Асинхронное получение
state = await orchestrator.get_state_async("thread-123")

# Очистка состояния
orchestrator.clear_state("thread-123")
```

### Стриминг

```python
async for chunk in orchestrator.stream("Проанализируй текст"):
    print(chunk)

async for event in orchestrator.astream_events("Проанализируй текст"):
    print(event)
```

---

## Полезные ссылки

- [LangChain Documentation](https://python.langchain.com)
- [LangGraph Documentation](https://python.langgraph.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Pydantic Documentation](https://docs.pydantic.dev)
