# Руководство программиста DeepAgent Orchestrator

## Введение

Это руководство предназначено для разработчиков, которые хотят внести вклад в развитие DeepAgent Orchestrator или создавать собственные агенты и инструменты для системы. В документе описаны архитектурные принципы, структуру кода, стандарты разработки и лучшие практики.

## Структура проекта

```
agent-orchestrator/
├── src/                    # Основной исходный код
│   ├── __init__.py
│   ├── orchestrator.py     # Главный оркестратор
│   ├── agent_registry.py   # Реестр агентов
│   ├── cli.py              # Командная строка
│   ├── api.py              # REST API
│   ├── config.py           # Конфигурация
│   ├── events/             # Событийная система
│   ├── interfaces/         # Интерфейсы
│   ├── services/           # Сервисы
│   ├── factories/          # Фабрики
│   ├── monitoring/         # Мониторинг
│   ├── tools/              # Инструменты
│   │   ├── __init__.py
│   │   ├── registry.py     # Реестр инструментов
│   │   ├── search.py       # Поисковые инструменты
│   │   ├── analysis.py     # Инструменты анализа
│   │   ├── pattern_tools.py # Инструменты распознавания паттернов
│   │   ├── dependency_tools.py # Инструменты анализа зависимостей
│   │   ├── s3_tools.py     # Инструменты S3
│   │   └── neo4j_tools.py  # Инструменты Neo4j
│   └── orchestration/      # Слой оркестрации
│       ├── __init__.py
│       └── coordinator.py  # Координатор агентов
├── agents/                 # Определения агентов в YAML
├── tests/                  # Тесты
├── docs/                   # Документация
├── config/                 # Файлы конфигурации
├── pyproject.toml          # Конфигурация проекта
└── README.md               # Основная документация
```

## Основные компоненты

### 1. Оркестратор (DeepAgentOrchestrator)

Центральный компонент системы, отвечающий за управление агентами и выполнение задач.

#### Использование:

```python
from src.orchestrator import DeepAgentOrchestrator

# Создание оркестратора с настройками по умолчанию
orchestrator = DeepAgentOrchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    system_prompt="Вы полезный ассистент"
)

# Выполнение задачи
result = await orchestrator.run("Проанализируйте этот текст")

# Потоковое выполнение
async for chunk in orchestrator.stream("Проанализируйте этот текст"):
    print(chunk)
```

#### Dependency Injection:

```python
from src.services import ModelService, AgentService, TaskService

# Создание сервисов для тестирования
model_service = ModelService()
agent_service = AgentService()
task_service = TaskService()

# Использование DI
orchestrator = DeepAgentOrchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    model_service=model_service,
    agent_service=agent_service,
    task_service=task_service
)
```

### 2. Конфигурация (OrchestratorConfig)

Класс для управления конфигурацией оркестратора.

```python
from src.orchestrator import OrchestratorConfig
from src.config import load_settings, apply_env_overrides

# Создание конфигурации напрямую
config = OrchestratorConfig(
    model_name="anthropic:claude-sonnet-4-20250514",
    system_prompt="Пользовательский промпт",
    subagents=[{"name": "researcher", "description": "..."}],
    enable_memory=False,
    workspace_dir="/tmp/agent-workspace"
)

# Создание из централизованных настроек
settings = load_settings("config.yaml")
settings = apply_env_overrides(settings)
config = OrchestratorConfig.from_settings(settings)
```

### 3. Сервисы

#### ModelService

Сервис для инициализации языковых моделей.

```python
from src.services import ModelService

service = ModelService()

# Валидация имени модели
is_valid = service.validate_model_name("anthropic:claude-sonnet-4-20250514")

# Получение доступных моделей
models = service.get_available_models()

# Инициализация модели
model = service.initialize(
    model_name="anthropic:claude-sonnet-4-20250514",
    api_key="your-api-key",
    base_url=None,
    temperature=0.1
)
```

#### AgentService

Сервис для создания и управления агентами.

```python
from src.services import AgentService

service = AgentService()

# Создание агента
agent = service.create_agent(
    model=model,
    system_prompt="Вы эксперт по анализу данных",
    subagents=[],
    custom_tools=[],
    interrupt_on={"write_file": {"allowed_decisions": ["approve", "reject"]}},
    backend=backend_factory
)

# Управление подагентами
agent = service.add_subagent(agent, {"name": "researcher", ...})
agent = service.remove_subagent(agent, "researcher")
subagents = service.get_subagents(agent)
```

#### TaskService

Сервис для выполнения задач.

```python
from src.services import TaskService

service = TaskService()

# Синхронное выполнение
result = await service.execute(
    task="Проанализируйте текст",
    agent=agent,
    thread_id="optional-thread-id"
)

# Стриминг результатов
async for chunk in service.stream(
    task="Проанализируйте текст",
    agent=agent
):
    print(chunk)

# Стриминг событий
async for event in service.stream_events(
    task="Проанализируйте текст",
    agent=agent
):
    print(event)
```

### 4. Интерфейсы

Все основные компоненты реализуют интерфейсы для обеспечения гибкости и тестируемости.

```python
from src.interfaces import (
    ModelInterface,    # Инициализация модели
    AgentInterface,    # Создание агентов
    TaskInterface,     # Выполнение задач
    BackendInterface,  # Бэкенд операции
    StorageInterface   # Операции хранилища
)
```

### 5. Фабрики

Фабрики обеспечивают удобное создание компонентов с Dependency Injection.

```python
from src.factories import OrchestratorFactory, create_orchestrator

# Использование фабрики
factory = OrchestratorFactory()
orchestrator = factory.create(
    model_name="anthropic:claude-sonnet-4-20250514",
    agents_dir="agents/",
    enable_memory=False
)

# Удобная функция
orchestrator = create_orchestrator(
    model_name="anthropic:claude-sonnet-4-20250514",
    agents_dir="agents/"
)

# DI кастомных сервисов
custom_model_service = CustomModelService()
orchestrator = factory.with_model_service(custom_model_service).create(
    model_name="custom-model"
)
```

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
    payload={"task": "analysis", "result": "success"}
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
            payload={"component": "my_component"}
        ))

# Использование
component = MyComponent()
component.set_event_bus(event_bus)
await component.do_something()
```

## Мониторинг и наблюдаемость

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
        message="Model loaded"
    )

checker.register_component("model", check_model)

# Проверка всех компонентов
checks = await checker.check_all()
for check in checks:
    print(f"{check.component}: {check.status.value}")

# Общий статус
status = checker.get_overall_status(checks)
```

## Создание новых компонентов

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
        task_service=task_service
    )

    assert orchestrator.model_service is model_service
    assert orchestrator.agent_service is agent_service
    assert orchestrator.task_service is task_service
```

## Стандарты кодирования

### Именование

1. **Файлы**: `snake_case.py`
2. **Классы**: `PascalCase`
3. **Функции/переменные**: `snake_case`
4. **Константы**: `SCREAMING_SNAKE_CASE`
5. **Булевы**: `is_*`, `has_*`, `should_*`

### Типизация

1. Использовать `X | None` вместо `Optional[X]`
2. Использовать `list[T]` и `dict[K, V]` из typing
3. Все публичные функции должны иметь аннотации типов

### Импорты

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

### Форматирование

1. 4 пробела для отступа
2. Максимальная длина строки: 100 символов
3. Двойные кавычки для строк
4. Одна пустая строка между объявлениями верхнего уровня

### Документация

1. Модуль-level docstrings для всех публичных модулей
2. Class docstrings описывающие атрибуты и использование
3. Function docstrings с секциями Args и Returns
4. Использование Google-style docstrings

```python
class DeepAgentOrchestrator:
    """
    Multi-agent orchestrator using LangChain DeepAgent.

    Attributes:
        config: OrchestratorConfig instance
        agent: Compiled DeepAgent graph
    """

    async def run(self, task: str, thread_id: str | None = None) -> dict[str, Any]:
        """
        Execute a task through the orchestrator.

        Args:
            task: The task to execute
            thread_id: Optional thread ID for state persistence

        Returns:
            Dict containing the execution result with messages
        """
```

## Линтинг и форматирование

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

## Git commit conventions

Используйте conventional commits:

```
feat: добавить новый инструмент поиска
fix: исправить ошибку в TaskService
docs: обновить документацию
refactor: рефакторинг сервисов
test: добавить тесты для компонента
```

## Частые задачи

### Добавление нового подагента

```python
orchestrator.add_subagent(
    name="researcher",
    description="Агент для исследования",
    system_prompt="Вы эксперт по поиску информации...",
    tools=["search", "read_file"],
    model="anthropic:claude-sonnet-4-20250514"
)
```

### Настройка HITL

```python
orchestrator.configure_hitl(
    tool_name="write_file",
    allowed_decisions=["approve", "edit", "reject"]
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
async for chunk in orchestrator.stream("Проанализируйте текст"):
    print(chunk)

async for event in orchestrator.astream_events("Проанализируйте текст"):
    print(event)
```

## Полезные ссылки

- [LangChain Documentation](https://python.langchain.com)
- [LangGraph Documentation](https://python.langgraph.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Pydantic Documentation](https://docs.pydantic.dev)