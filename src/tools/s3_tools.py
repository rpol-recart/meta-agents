"""
S3 Tools - AWS S3 bucket operations for file listing and browsing.

This module provides tools for interacting with AWS S3 storage buckets,
including listing files, getting file metadata, and navigating prefixes.

Required environment variables:
    AWS_ACCESS_KEY_ID     - AWS access key ID
    AWS_SECRET_ACCESS_KEY - AWS secret access key
    AWS_REGION            - AWS region (default: us-east-1)
    AWS_ENDPOINT_URL      - Optional S3-compatible endpoint URL (for MinIO, etc.)
    S3_BUCKET_NAME        - Default bucket name for operations

Example usage:
    from src.tools.s3_tools import get_s3_tools
    
    # Register tools
    registry = get_s3_tools()
    
    # Or use directly
    from src.tools.s3_tools import list_s3_files
    
    result = list_s3_files(bucket="my-bucket", prefix="data/")
"""

import logging
import os
from typing import Optional

from .registry import ToolRegistry, get_default_tool_registry


logger = logging.getLogger(__name__)


def get_s3_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """
    Get S3-related tools.

    Args:
        registry: Optional registry to add tools to

    Returns:
        ToolRegistry with S3 tools
    """
    if registry is None:
        registry = get_default_tool_registry()

    def list_s3_files(
        bucket: Optional[str] = None,
        prefix: Optional[str] = "",
        max_keys: int = 100,
    ) -> str:
        """
        List files in an S3 bucket with optional prefix filtering.

        Args:
            bucket: S3 bucket name (uses S3_BUCKET_NAME env var if not provided)
            prefix: Prefix to filter objects (folder path)
            max_keys: Maximum number of keys to return (1-1000)

        Returns:
            Formatted list of files with metadata
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 package is required for S3 operations. "
                "Install it with: pip install boto3"
            )

        bucket_name = bucket or os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            return (
                "Error: No bucket specified. Provide 'bucket' parameter or "
                "set S3_BUCKET_NAME environment variable."
            )

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                endpoint_url=endpoint_url,
            )

            paginator = s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix or "",
                PaginationConfig={"MaxItems": max_keys},
            )

            files_info = []
            total_size = 0
            file_count = 0
            folder_count = 0

            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        size = obj.get("Size", 0)
                        last_modified = obj.get("LastModified", "")
                        is_folder = key.endswith("/")

                        if is_folder:
                            folder_count += 1
                        else:
                            file_count += 1
                            total_size += size
                            files_info.append(
                                {
                                    "name": key.split("/")[-1] if not is_folder else key,
                                    "path": key,
                                    "size": format_size(size),
                                    "modified": str(last_modified)[:19]
                                    if last_modified
                                    else "unknown",
                                    "type": "folder" if is_folder else "file",
                                }
                            )

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

            logger.info(f"Listed {file_count} files in bucket {bucket_name}")
            return result

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                return f"Error: Bucket '{bucket_name}' does not exist"
            elif error_code == "AccessDenied":
                return (
                    f"Error: Access denied to bucket '{bucket_name}'. "
                    "Check AWS credentials and permissions."
                )
            return f"Error: S3 operation failed - {e}"
        except Exception as e:
            logger.error(f"S3 list operation failed: {e}")
            return f"Error: {str(e)}"

    def list_s3_buckets() -> str:
        """
        List all S3 buckets available to the credentials.

        Returns:
            Formatted list of buckets with creation dates
        """
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 package is required for S3 operations. "
                "Install it with: pip install boto3"
            )

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                endpoint_url=endpoint_url,
            )

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

            logger.info(f"Listed {len(buckets)} S3 buckets")
            return result

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "AccessDenied":
                return (
                    "Error: Access denied. Check AWS credentials and permissions."
                )
            return f"Error: {e}"
        except Exception as e:
            logger.error(f"S3 buckets list failed: {e}")
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
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 package is required for S3 operations. "
                "Install it with: pip install boto3"
            )

        if not key:
            return "Error: 'key' parameter is required (S3 object path)"

        bucket_name = bucket or os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            return (
                "Error: No bucket specified. Provide 'bucket' parameter or "
                "set S3_BUCKET_NAME environment variable."
            )

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                endpoint_url=endpoint_url,
            )

            response = s3_client.head_object(Bucket=bucket_name, Key=key)

            result = f"S3 Object Information\n"
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

            logger.info(f"Got file info for s3://{bucket_name}/{key}")
            return result

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                return f"Error: Key '{key}' not found in bucket '{bucket_name}'"
            elif error_code == "NoSuchBucket":
                return f"Error: Bucket '{bucket_name}' does not exist"
            elif error_code == "AccessDenied":
                return (
                    f"Error: Access denied to s3://{bucket_name}/{key}. "
                    "Check AWS credentials and permissions."
                )
            return f"Error: S3 operation failed - {e}"
        except Exception as e:
            logger.error(f"S3 head_object failed: {e}")
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
        try:
            import fnmatch
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError(
                "boto3 package is required for S3 operations. "
                "Install it with: pip install boto3"
            )

        bucket_name = bucket or os.environ.get("S3_BUCKET_NAME")
        if not bucket_name:
            return (
                "Error: No bucket specified. Provide 'bucket' parameter or "
                "set S3_BUCKET_NAME environment variable."
            )

        if not pattern:
            return "Error: 'pattern' parameter is required for search"

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

        try:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                endpoint_url=endpoint_url,
            )

            paginator = s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=bucket_name,
                Prefix=prefix or "",
            )

            matches = []
            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        search_key = key if case_sensitive else key.lower()
                        search_pattern = pattern if case_sensitive else pattern.lower()

                        if fnmatch.fnmatch(search_key, search_pattern):
                            matches.append(
                                {
                                    "key": key,
                                    "size": obj.get("Size", 0),
                                    "modified": str(obj.get("LastModified", ""))[:19],
                                }
                            )

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

            logger.info(f"Found {len(matches)} files matching '{pattern}'")
            return result

        except ClientError as e:
            return f"Error: S3 operation failed - {e}"
        except Exception as e:
            logger.error(f"S3 search failed: {e}")
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
            import boto3
            from botocore.exceptions import ClientError

            s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
                endpoint_url=endpoint_url,
            )

            response = s3_client.list_buckets()
            buckets = response.get("Buckets", [])

            result += f"\nAvailable Buckets ({len(buckets)}):\n"
            for bucket in buckets[:10]:
                result += f"  - {bucket['Name']}\n"
            if len(buckets) > 10:
                result += f"  ... and {len(buckets) - 10} more\n"

            if bucket_name:
                try:
                    s3_client.head_bucket(Bucket=bucket_name)
                    result += f"\nDefault Bucket '{bucket_name}': REACHABLE\n"
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code == "404":
                        result += f"\nDefault Bucket '{bucket_name}': NOT FOUND\n"
                    else:
                        result += f"\nDefault Bucket '{bucket_name}': ACCESS ERROR\n"

        except Exception as e:
            result += f"\nConnection Error: {str(e)}\n"

        return result

    registry.register(
        name="list_s3_files",
        func=list_s3_files,
        description="List files in an S3 bucket with optional prefix filter",
        parameters={
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "description": "S3 bucket name (uses S3_BUCKET_NAME env var if not provided)",
                },
                "prefix": {
                    "type": "string",
                    "description": "Prefix to filter objects (folder path)",
                    "default": "",
                },
                "max_keys": {
                    "type": "integer",
                    "description": "Maximum number of files to return (1-1000)",
                    "default": 100,
                },
            },
        },
    )

    registry.register(
        name="list_s3_buckets",
        func=list_s3_buckets,
        description="List all S3 buckets available to the current AWS credentials",
    )

    registry.register(
        name="get_s3_file_info",
        func=get_s3_file_info,
        description="Get metadata information for a specific S3 object",
        parameters={
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "description": "S3 bucket name (uses S3_BUCKET_NAME env var if not provided)",
                },
                "key": {
                    "type": "string",
                    "description": "Full S3 object key (path)",
                },
            },
            "required": ["key"],
        },
    )

    registry.register(
        name="search_s3_files",
        func=search_s3_files,
        description="Search for files in S3 bucket matching a pattern",
        parameters={
            "type": "object",
            "properties": {
                "bucket": {
                    "type": "string",
                    "description": "S3 bucket name",
                },
                "prefix": {
                    "type": "string",
                    "description": "Prefix to filter objects",
                    "default": "",
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern with wildcards (* and ?)",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether search is case-sensitive",
                    "default": True,
                },
            },
            "required": ["pattern"],
        },
    )

    registry.register(
        name="s3_config_status",
        func=s3_config_status,
        description="Check S3 configuration and connectivity status",
    )

    return registry


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_default_s3_tools() -> ToolRegistry:
    """Get all default tools including S3."""
    registry = get_default_tool_registry()
    registry = get_s3_tools(registry)
    return registry
