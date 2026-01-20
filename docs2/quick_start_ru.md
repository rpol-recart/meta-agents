# Быстрый старт с DeepAgent Orchestrator

## Введение

Это руководство поможет вам быстро установить и начать использовать DeepAgent Orchestrator - мощную систему оркестрации AI-агентов. Вы узнаете, как установить систему, настроить окружение и выполнить первые задачи.

## Системные требования

### Минимальные требования
- Python 3.10 или выше
- 4 ГБ свободной оперативной памяти
- 2 ГБ свободного места на диске
- Доступ в интернет для загрузки зависимостей

### Рекомендуемые требования
- Python 3.11 или выше
- 8 ГБ оперативной памяти
- 4 ГБ свободного места на диске
- Современный процессор с несколькими ядрами

## Установка

### 1. Клонирование репозитория

```bash
# Клонирование репозитория
git clone https://github.com/your-org/deepagent-orchestrator.git
cd deepagent-orchestrator

# Или если у вас уже есть архив
tar -xzf deepagent-orchestrator.tar.gz
cd deepagent-orchestrator
```

### 2. Создание виртуального окружения

```bash
# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
# Установка основных зависимостей
pip install -e .

# Установка зависимостей для разработки (опционально)
pip install -e ".[dev]"
```

### 4. Настройка конфигурации

```bash
# Копирование примера конфигурации
cp .env.example .env

# Редактирование конфигурации
nano .env
```

## Конфигурация

### Переменные окружения

Основные переменные окружения, которые необходимо настроить:

```bash
# .env файл
# Модель по умолчанию
DEFAULT_MODEL=anthropic:claude-sonnet-4-20250514

# API ключи
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key

# Опционально: базовый URL для совместимых API
OPENAI_BASE_URL=http://localhost:11434/v1

# Поисковые сервисы
TAVILY_API_KEY=your-tavily-api-key

# Хранилища данных (опционально)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
S3_BUCKET_NAME=your-bucket-name
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### Файл конфигурации YAML

Альтернативно, можно использовать файл `config/settings.yaml`:

```yaml
# config/settings.yaml
model:
  provider: "anthropic"
  name: "claude-sonnet-4-20250514"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.1
  max_tokens: 4096

backend:
  root_dir: "/tmp/agent-workspace"
  store_namespace: "memories"

subagents:
  directory: "agents/"
  auto_load: true
  hot_reload: true

hitl:
  enabled: true
  tools:
    run_command:
      allowed_decisions: ["approve", "edit", "reject"]
    delete_file:
      allowed_decisions: ["approve", "reject"]

api:
  host: "0.0.0.0"
  port: 8000
  reload: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Первый запуск

### Через командную строку

```bash
# Проверка установки
orchestrate --help

# Выполнение простой задачи
orchestrate run "Напиши краткое описание AI-агентов" --agents-dir agents/

# Выполнение задачи с подробным выводом
orchestrate run "Исследуй последние тренды в AI" --agents-dir agents/ -v

# Запуск с определенной моделью
orchestrate run "Проанализируй этот текст" --model "openai:gpt-4" --agents-dir agents/
```

### Через Python API

```python
# simple_example.py
import asyncio
from src.orchestrator import DeepAgentOrchestrator

async def main():
    # Создание оркестратора
    orchestrator = DeepAgentOrchestrator(
        model_name="anthropic:claude-sonnet-4-20250514",
        system_prompt="Вы полезный ассистент"
    )
    
    # Выполнение задачи
    result = await orchestrator.run("Напиши краткое описание AI-агентов")
    
    # Вывод результата
    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

Запуск скрипта:
```bash
python simple_example.py
```

## Примеры использования

### 1. Простой анализ текста

```bash
# Через CLI
orchestrate run "Проанализируй преимущества использования AI-агентов в бизнесе" --agents-dir agents/

# Через Python
import asyncio
from src.orchestrator import DeepAgentOrchestrator

async def text_analysis():
    orchestrator = DeepAgentOrchestrator()
    result = await orchestrator.run(
        "Проанализируй преимущества использования AI-агентов в бизнесе"
    )
    print(result["messages"][-1].content)

asyncio.run(text_analysis())
```

### 2. Исследование с несколькими агентами

```bash
# Создание задачи с использованием нескольких агентов
orchestrate run "Исследуй тему машинного обучения и создай отчет" --agents-dir agents/
```

### 3. Генерация кода

```bash
# Генерация Python скрипта
orchestrate run "Создай Python скрипт для чтения CSV файла и вычисления среднего значения" --agents-dir agents/
```

### 4. Работа с файлами

```bash
# Создание и редактирование файлов
orchestrate run "Создай файл README.md с описанием нашего проекта" --agents-dir agents/
```

## Использование REST API

### Запуск сервера

```bash
# Запуск API сервера
orchestrate api --port 8000

# Запуск с указанием хоста
orchestrate api --host 0.0.0.0 --port 8000
```

### Примеры API запросов

#### Выполнение задачи

```bash
# POST запрос для выполнения задачи
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Исследуй последние достижения в области нейронных сетей",
    "model": "anthropic:claude-sonnet-4-20250514",
    "stream": false
  }'
```

#### Стриминг результатов

```bash
# Получение результатов через WebSocket
# Подключение к ws://localhost:8000/ws/run/{thread_id}
```

#### Управление агентами

```bash
# Получение списка агентов
curl -X GET http://localhost:8000/subagents

# Создание нового агента
curl -X POST http://localhost:8000/subagents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_researcher",
    "description": "Специализированный исследователь",
    "system_prompt": "Вы эксперт в области технологий",
    "tools": ["web_search", "read_file"]
  }'
```

## Создание собственных агентов

### Быстрое создание агента

```bash
# Создание нового агента через CLI
orchestrate create-agent research_specialist "Специалист по научным исследованиям" "Вы эксперт в проведении научных исследований" --output agents/research_specialist.yaml
```

### Ручное создание YAML файла

```yaml
# agents/my_custom_agent.yaml
name: my_custom_agent
description: "Мой кастомный агент"
system_prompt: |
  Вы специализированный ассистент для выполнения конкретных задач.
  Следуйте этим правилам:
  1. Всегда проверяйте точность информации
  2. Задавайте уточняющие вопросы при необходимости
  3. Предоставляйте структурированные ответы
tools:
  - web_search
  - read_file
  - write_file
```

## Тестирование установки

### Проверка базовой функциональности

```bash
# Запуск тестов
pytest tests/ -v

# Запуск конкретных тестов
pytest tests/test_orchestrator.py::TestOrchestratorBasics::test_initialization -v

# Проверка линтинга
ruff check src/

# Проверка типов
mypy src/
```

### Пример комплексного теста

```python
# test_installation.py
import asyncio
import pytest
from src.orchestrator import DeepAgentOrchestrator

@pytest.mark.asyncio
async def test_basic_functionality():
    """Тест базовой функциональности системы."""
    # Создание оркестратора
    orchestrator = DeepAgentOrchestrator(
        model_name="test-model",  # Используем тестовую модель
        enable_memory=False
    )
    
    # Проверка наличия встроенных инструментов
    assert hasattr(orchestrator, 'agent')
    assert orchestrator.agent is not None
    
    print("✅ Базовая установка работает корректно")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
```

## Распространенные проблемы и решения

### 1. Ошибка импорта модулей

**Проблема:**
```
ModuleNotFoundError: No module named 'src'
```

**Решение:**
```bash
# Убедитесь, что вы в правильной директории
pwd
# Должно показать директорию проекта

# Переустановите пакет в режиме разработки
pip install -e .
```

### 2. Ошибка API ключа

**Проблема:**
```
AuthenticationError: Invalid API key
```

**Решение:**
```bash
# Проверьте переменные окружения
echo $ANTHROPIC_API_KEY

# Убедитесь, что ключ установлен правильно
export ANTHROPIC_API_KEY=your-correct-api-key
```

### 3. Проблемы с зависимостями

**Проблема:**
```
ImportError: cannot import name 'DeepAgent' from 'deepagents'
```

**Решение:**
```bash
# Обновите зависимости
pip install --upgrade deepagents langchain langgraph

# Или переустановите все зависимости
pip uninstall -y deepagent-orchestrator
pip install -e .
```

## Полезные команды

### Управление агентами

```bash
# Список всех агентов
orchestrate list-agents agents/

# Создание нового агента
orchestrate create-agent new_agent "Новый агент" "Описание агента"

# Проверка состояния API
curl http://localhost:8000/health
```

### Разработка и отладка

```bash
# Форматирование кода
black src/ tests/

# Линтинг
ruff check src/ tests/

# Автоисправление ошибок линтинга
ruff check --fix src/ tests/

# Проверка типов
mypy src/
```

## Следующие шаги

После успешной установки и выполнения первых задач вы можете:

1. **Изучить документацию** - Ознакомьтесь с полной документацией в директории `docs/`
2. **Создать собственных агентов** - Разработайте агентов для своих специфических задач
3. **Интегрировать внешние сервисы** - Подключите базы данных, облачные хранилища и другие системы
4. **Настроить мониторинг** - Внедрите систему отслеживания производительности и ошибок
5. **Развернуть в production** - Настройте систему для использования в рабочей среде

## Поддержка и сообщество

Если у вас возникли проблемы или вопросы:

1. **Документация** - Проверьте полную документацию в `docs/`
2. **GitHub Issues** - Создайте issue в репозитории проекта
3. **Discord/Slack** - Присоединяйтесь к сообществу разработчиков
4. **Stack Overflow** - Используйте теги `deepagent-orchestrator`

## Заключение

Поздравляем! Вы успешно установили и протестировали DeepAgent Orchestrator. Система готова к использованию для автоматизации сложных задач с помощью AI-агентов. 

Помните, что это лишь начало - система обладает мощными возможностями для расширения и настройки под ваши конкретные нужды. Экспериментируйте, создавайте новых агентов и инструменты, и делитесь своим опытом с сообществом.