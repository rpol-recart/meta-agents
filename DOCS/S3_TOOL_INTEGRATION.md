# S3 Tool Integration Guide

## Overview

This document explains how to integrate the S3 tool into the Aluminum Technology Assistant agent framework.

## Tool Integration

The S3 tool has been integrated as two separate LangChain tools:

1. `S3ListFilesTool` - Lists files in an S3 bucket
2. `S3GetBucketInfoTool` - Gets information about an S3 bucket

## Adding the Tools to an Agent

To add the S3 tools to an agent, import and instantiate them:

```python
from tools.s3_agent_tool import S3ListFilesTool, S3GetBucketInfoTool

# Create the tools
s3_list_tool = S3ListFilesTool()
s3_info_tool = S3GetBucketInfoTool()

# Add to your agent's tools list
tools = [
    # ... other tools
    s3_list_tool,
    s3_info_tool,
]
```

## Required Environment Variables

Before using the S3 tools, you must set the following environment variables:

- `AWS_ACCESS_KEY_ID` - Your AWS access key ID
- `AWS_SECRET_ACCESS_KEY` - Your AWS secret access key
- `AWS_DEFAULT_REGION` - AWS region (optional, defaults to us-east-1)

See `DOCS/S3_TOOL_ENV_VARS.md` for detailed instructions on setting these variables.

## Usage Examples

### Listing Files in a Bucket

```python
# List files in a bucket
result = s3_list_tool._run("my-bucket-name", "documents/", 100)
if result["success"]:
    for file in result["files"]:
        print(f"Key: {file['key']}, Size: {file['size']} bytes")
else:
    print(f"Error: {result['error']}")
```

### Getting Bucket Information

```python
# Get bucket information
result = s3_info_tool._run("my-bucket-name")
if result["success"]:
    print(f"Bucket: {result['bucket']}")
    print(f"Location: {result['location']}")
    print(f"Object count: {result['object_count']}")
else:
    print(f"Error: {result['error']}")
```

## IAM Permissions

The AWS credentials must have the following permissions:

- `s3:ListBucket` - To list objects in the bucket
- `s3:GetBucketLocation` - To get bucket region information

See `DOCS/S3_TOOL_ENV_VARS.md` for a sample IAM policy.

## Error Handling

The S3 tools will return structured error responses if operations fail:

```python
result = s3_list_tool._run("non-existent-bucket")
if not result["success"]:
    print(f"Operation failed: {result['error']}")
```

Common error cases:
- Missing AWS credentials
- Invalid AWS credentials
- Bucket does not exist
- Insufficient permissions
- Network connectivity issues