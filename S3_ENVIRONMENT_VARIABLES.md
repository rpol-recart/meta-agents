# S3 Environment Variables Documentation

## Required Environment Variables

Для работы с S3 инструментами необходимо настроить следующие переменные окружения:

### AWS Credentials (обязательно)

```bash
# AWS Access Key ID
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE

# AWS Secret Access Key  
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

### AWS Region (опционально)

```bash
# AWS Region (по умолчанию: us-east-1)
export AWS_REGION=us-east-1
```

### S3-Compatible Endpoint (опционально)

Для использования с S3-совместимыми сервисами (MinIO, LocalStack, DigitalOcean Spaces, etc.):

```bash
# Custom S3-compatible endpoint URL
export AWS_ENDPOINT_URL=https://s3.example.com

# Для LocalStack:
export AWS_ENDPOINT_URL=http://localhost:4566

# Для MinIO:
export AWS_ENDPOINT_URL=http://localhost:9000
```

### Default Bucket (опционально)

```bash
# Имя бакета по умолчанию
export S3_BUCKET_NAME=my-data-bucket
```

## Complete Configuration Example

### AWS Production

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_REGION=us-east-1
```

### Local Development with LocalStack

```bash
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://localhost:4566
export S3_BUCKET_NAME=local-bucket
```

### MinIO

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://localhost:9000
export S3_BUCKET_NAME=my-bucket
```

## IAM Permissions

Учетные данные AWS должны иметь следующие разрешения:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name",
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```

## Verification

Проверьте конфигурацию:

```python
from new_tool import s3_config_status

print(s3_config_status())
```

Или из командной строки:

```bash
python new_tool.py
```

## Installation

```bash
pip install boto3
```

## Usage Examples

### List files in bucket

```python
from new_tool import list_s3_files

result = list_s3_files(
    bucket="my-bucket",
    prefix="data/",
    max_keys=100
)
print(result)
```

### Get file information

```python
from new_tool import get_s3_file_info

result = get_s3_file_info(
    bucket="my-bucket",
    key="data/file.csv"
)
print(result)
```

### Search files

```python
from new_tool import search_s3_files

result = search_s3_files(
    bucket="my-bucket",
    prefix="data/",
    pattern="*.csv",
    case_sensitive=False
)
print(result)
```

## Error Handling

| Ошибка | Причина | Решение |
|--------|---------|---------|
| Missing credentials | Не установлены AWS_ACCESS_KEY_ID и/или AWS_SECRET_ACCESS_KEY | Добавьте переменные окружения |
| Access denied | Недостаточно прав | Проверьте IAM разрешения |
| NoSuchBucket | Бакет не существует | Проверьте имя бакета |
| Connection Error | Не удается подключиться к API | Проверьте AWS_ENDPOINT_URL |

## Troubleshooting

### Проверка переменных окружения

```bash
# Linux/Mac
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
echo $AWS_REGION

# Windows (PowerShell)
echo $env:AWS_ACCESS_KEY_ID
```

### Проверка AWS CLI

```bash
aws configure list
aws s3 ls
```
