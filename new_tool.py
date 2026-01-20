"""
S3 File Browser Tool - List and browse files in S3 storage bucket.

This module provides tools for viewing and getting lists of files in an S3 bucket.

REQUIRED ENVIRONMENT VARIABLES:
    AWS_ACCESS_KEY_ID     - AWS access key ID (required)
    AWS_SECRET_ACCESS_KEY - AWS secret access key (required)
    AWS_REGION            - AWS region (default: us-east-1)
    AWS_ENDPOINT_URL      - Optional S3-compatible endpoint URL (for MinIO, local S3, etc.)
    S3_BUCKET_NAME        - Default bucket name (used when bucket not specified in call)

INSTALLATION:
    pip install boto3

AVAILABLE FUNCTIONS:
    list_s3_files(bucket, prefix, max_keys)
        List files in an S3 bucket with optional prefix filter

    list_s3_buckets()
        List all S3 buckets available to the credentials

    get_s3_file_info(bucket, key)
        Get metadata information for a specific S3 object

    search_s3_files(bucket, prefix, pattern, case_sensitive)
        Search for files matching a pattern in key names

    s3_config_status()
        Check S3 configuration and connectivity status

EXAMPLE USAGE:
    # Check configuration
    result = s3_config_status()

    # List files in bucket
    result = list_s3_files(bucket="my-bucket", prefix="data/")

    # Get file metadata
    result = get_s3_file_info(bucket="my-bucket", key="data/file.csv")

    # Search files
    result = search_s3_files(bucket="my-bucket", pattern="*.csv")

ENVIRONMENT VARIABLES CONFIGURATION:

    # AWS IAM User credentials (recommended for production)
    export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
    export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    export AWS_REGION=us-east-1

    # Or use default bucket (optional, can override per call)
    export S3_BUCKET_NAME=my-data-bucket

    # For S3-compatible services (MinIO, DigitalOcean, etc.)
    export AWS_ENDPOINT_URL=https://s3.example.com

    # For local development with LocalStack or MinIO
    export AWS_ENDPOINT_URL=http://localhost:4566

ERROR HANDLING:
    - Missing credentials: Returns error with setup instructions
    - Access denied: Returns error message with permission info
    - Bucket not found: Returns error with bucket name
    - Invalid key: Returns error with key path
"""

import fnmatch
import logging
import os
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _get_s3_client():
    """Create and return S3 client from environment variables."""
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 package is required for S3 operations. "
            "Install it with: pip install boto3"
        ) from exc

    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

    if not aws_access_key or not aws_secret_key:
        raise ValueError(
            "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY environment variables."
        )

    return boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region,
        endpoint_url=endpoint_url,
    )


def _build_file_info(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Build file info dictionary from S3 object response."""
    key = obj["Key"]
    size = obj.get("Size", 0)
    last_modified = obj.get("LastModified", "")
    is_folder = key.endswith("/")

    return {
        "name": key.split("/")[-1] if not is_folder else key,
        "path": key,
        "size": format_size(size),
        "modified": str(last_modified)[:19] if last_modified else "unknown",
        "type": "folder" if is_folder else "file",
    }


def _build_file_info_search(obj: Dict[str, Any]) -> Dict[str, str]:
    """Build file info dictionary for search results."""
    return {
        "key": obj["Key"],
        "size": obj.get("Size", 0),
        "modified": str(obj.get("LastModified", ""))[:19],
    }


def list_s3_files(
    bucket: Optional[str] = None,
    prefix: Optional[str] = "",
    max_keys: int = 100,
) -> str:
    """
    List files in an S3 bucket with optional prefix filter.

    Args:
        bucket: S3 bucket name (uses S3_BUCKET_NAME env var if not provided)
        prefix: Prefix to filter objects (folder path)
        max_keys: Maximum number of keys to return (1-1000)

    Returns:
        Formatted list of files with metadata
    """
    bucket_name = bucket or os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        return (
            "Error: No bucket specified. Provide 'bucket' parameter or "
            "set S3_BUCKET_NAME environment variable."
        )

    try:
        s3_client = _get_s3_client()

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(
            Bucket=bucket_name,
            Prefix=prefix or "",
            PaginationConfig={"MaxItems": max_keys},
        )

        files_info: List[Dict[str, Any]] = []
        total_size = 0
        file_count = 0
        folder_count = 0

        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    is_folder = key.endswith("/")

                    if is_folder:
                        folder_count += 1
                    else:
                        file_count += 1
                        total_size += obj.get("Size", 0)
                        files_info.append(_build_file_info(obj))

        result = f"S3 Bucket: {bucket_name}\n"
        result += f"Prefix: {prefix or '(root)'}\n"
        result += f"Found: {file_count} files, {folder_count} folders\n"
        result += f"Total size: {format_size(total_size)}\n"
        result += "-" * 60 + "\n\n"

        for info in files_info[:50]:
            icon = "[DIR]" if info["type"] == "folder" else "[FILE]"
            result += f"{icon:8} {info['name']:<40} {info['size']:>10}  {info['modified']}\n"

        if len(files_info) > 50:
            result += f"\n... and {len(files_info) - 50} more files"

        logger.info("Listed %d files in bucket %s", file_count, bucket_name)
        return result

    except Exception as e:
        logger.error("S3 list operation failed: %s", e)
        return f"Error: {str(e)}"


def list_s3_buckets() -> str:
    """
    List all S3 buckets available to the credentials.

    Returns:
        Formatted list of buckets with creation dates
    """
    try:
        s3_client = _get_s3_client()
        response = s3_client.list_buckets()
        buckets = response.get("Buckets", [])

        if not buckets:
            return "No S3 buckets found for this AWS account."

        result = "Available S3 Buckets:\n"
        result += "-" * 50 + "\n\n"

        for bucket in buckets:
            name = bucket.get("Name", "unknown")
            creation_date = bucket.get("CreationDate", "")
            date_str = str(creation_date)[:19] if creation_date else "unknown"
            result += f"[BUCKET] {name:<40}  Created: {date_str}\n"

        logger.info("Listed %d S3 buckets", len(buckets))
        return result

    except Exception as e:
        logger.error("S3 buckets list failed: %s", e)
        return f"Error: {str(e)}"


def get_s3_file_info(
    bucket: Optional[str] = None,
    key: str = "",
) -> str:
    """
    Get metadata information for a specific S3 object.

    Args:
        bucket: S3 bucket name (uses S3_BUCKET_NAME env var if not provided)
        key: Full path to the file in S3

    Returns:
        File metadata as formatted string
    """
    if not key:
        return "Error: 'key' parameter is required (S3 object path)"

    bucket_name = bucket or os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        return (
            "Error: No bucket specified. Provide 'bucket' parameter or "
            "set S3_BUCKET_NAME environment variable."
        )

    try:
        s3_client = _get_s3_client()
        response = s3_client.head_object(Bucket=bucket_name, Key=key)

        result = "S3 Object Information\n"
        result += "-" * 40 + "\n\n"
        result += f"Bucket:    {bucket_name}\n"
        result += f"Key:       {key}\n"
        result += f"Size:      {format_size(response.get('ContentLength', 0))}\n"

        last_modified = response.get("LastModified", "")
        result += f"Modified:  {str(last_modified)[:19] if last_modified else 'unknown'}\n"

        content_type = response.get("ContentType", "unknown")
        result += f"Type:      {content_type}\n"

        etag = response.get("ETag", "").strip('"')
        result += f"ETag:      {etag}\n"

        metadata = response.get("Metadata", {})
        if metadata:
            result += "\nMetadata:\n"
            for k, v in metadata.items():
                result += f"  {k}: {v}\n"

        logger.info("Got file info for s3://%s/%s", bucket_name, key)
        return result

    except Exception as e:
        logger.error("S3 head_object failed: %s", e)
        return f"Error: {str(e)}"


def search_s3_files(
    bucket: Optional[str] = None,
    prefix: Optional[str] = "",
    pattern: str = "",
    case_sensitive: bool = True,
) -> str:
    """
    Search for files in S3 bucket matching a pattern in the key name.

    Args:
        bucket: S3 bucket name (uses S3_BUCKET_NAME env var if not provided)
        prefix: Prefix to filter objects
        pattern: Search pattern (supports * and ? wildcards)
        case_sensitive: Whether search is case-sensitive

    Returns:
        List of matching files
    """
    bucket_name = bucket or os.environ.get("S3_BUCKET_NAME")
    if not bucket_name:
        return (
            "Error: No bucket specified. Provide 'bucket' parameter or "
            "set S3_BUCKET_NAME environment variable."
        )

    if not pattern:
        return "Error: 'pattern' parameter is required for search"

    try:
        s3_client = _get_s3_client()

        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix or "")

        matches: List[Dict[str, Any]] = []
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    search_key = key if case_sensitive else key.lower()
                    search_pattern = pattern if case_sensitive else pattern.lower()

                    if fnmatch.fnmatch(search_key, search_pattern):
                        matches.append(_build_file_info_search(obj))

        if not matches:
            return (
                f"No files matching pattern '{pattern}' found in "
                f"s3://{bucket_name}/{prefix or ''}"
            )

        result = f"S3 Search Results: '{pattern}'\n"
        result += f"Bucket: {bucket_name}\n"
        result += f"Prefix: {prefix or '(root)'}\n"
        result += f"Found: {len(matches)} matching files\n"
        result += "-" * 60 + "\n\n"

        for m in matches[:100]:
            result += f"[FILE] {m['key']:<50} {format_size(m['size'])}\n"

        if len(matches) > 100:
            result += f"\n... and {len(matches) - 100} more files"

        logger.info("Found %d files matching '%s'", len(matches), pattern)
        return result

    except Exception as e:
        logger.error("S3 search failed: %s", e)
        return f"Error: {str(e)}"


def s3_config_status() -> str:
    """
    Check S3 configuration and connectivity status.

    Returns:
        Configuration status and available buckets
    """
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

    result = "S3 Configuration Status\n"
    result += "-" * 40 + "\n\n"

    result += "Environment Variables:\n"
    result += f"  AWS_ACCESS_KEY_ID:     {'Set' if aws_access_key else 'NOT SET'}\n"
    result += f"  AWS_SECRET_ACCESS_KEY: {'Set' if aws_secret_key else 'NOT SET'}\n"
    result += f"  AWS_REGION:            {aws_region}\n"
    result += f"  AWS_ENDPOINT_URL:      {endpoint_url or 'Default (AWS)'}\n"
    result += f"  S3_BUCKET_NAME:        {bucket_name or 'NOT SET'}\n\n"

    if not aws_access_key or not aws_secret_key:
        result += "Status: INCOMPLETE - Missing AWS credentials\n"
        result += "\nTo configure, set these environment variables:\n"
        result += "  export AWS_ACCESS_KEY_ID=your_access_key\n"
        result += "  export AWS_SECRET_ACCESS_KEY=your_secret_key\n"
        result += "  export AWS_REGION=us-east-1\n"
        result += "  export S3_BUCKET_NAME=your-bucket-name\n"
        return result

    result += "Status: CREDENTIALS CONFIGURED\n"

    try:
        s3_client = _get_s3_client()
        response = s3_client.list_buckets()
        buckets = response.get("Buckets", [])

        result += f"\nAvailable Buckets ({len(buckets)}):\n"
        for bucket_item in buckets[:10]:
            result += f"  - {bucket_item['Name']}\n"
        if len(buckets) > 10:
            result += f"  ... and {len(buckets) - 10} more\n"

        if bucket_name:
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                result += f"\nDefault Bucket '{bucket_name}': REACHABLE\n"
            except Exception:
                result += f"\nDefault Bucket '{bucket_name}': NOT FOUND or ACCESS DENIED\n"

    except Exception as conn_error:
        result += f"\nConnection Error: {str(conn_error)}\n"

    return result


if __name__ == "__main__":
    print("S3 File Browser Tool")
    print("=" * 50)
    print("\nEnvironment variables required:")
    print("  AWS_ACCESS_KEY_ID")
    print("  AWS_SECRET_ACCESS_KEY")
    print("  AWS_REGION (default: us-east-1)")
    print("  S3_BUCKET_NAME (optional, for default bucket)")
    print("  AWS_ENDPOINT_URL (optional, for S3-compatible services)")
    print("\nAvailable functions:")
    print("  list_s3_files(bucket, prefix, max_keys)")
    print("  list_s3_buckets()")
    print("  get_s3_file_info(bucket, key)")
    print("  search_s3_files(bucket, prefix, pattern, case_sensitive)")
    print("  s3_config_status()")
