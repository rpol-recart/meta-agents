# Required Environment Variables for S3 Tool

To use the S3 file listing tool, you need to set the following environment variables:

## Required Variables

1. `AWS_ACCESS_KEY_ID` - Your AWS access key ID
2. `AWS_SECRET_ACCESS_KEY` - Your AWS secret access key

## Optional Variables

3. `AWS_DEFAULT_REGION` - AWS region (default: us-east-1)
4. `AWS_SESSION_TOKEN` - Session token (if using temporary credentials)

## Setting Environment Variables

### Linux/Mac:
```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=us-west-2  # Optional
```

### Windows:
```cmd
set AWS_ACCESS_KEY_ID=your_access_key_id
set AWS_SECRET_ACCESS_KEY=your_secret_access_key
set AWS_DEFAULT_REGION=us-west-2  # Optional
```

### Using a .env file:
```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-west-2
```

## IAM Permissions Required

The AWS credentials must have the following permissions:
- `s3:ListBucket` - To list objects in the bucket
- `s3:GetBucketLocation` - To get bucket region information (for bucket info feature)

Example IAM policy:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::your-bucket-name/*"
            ]
        }
    ]
}
```