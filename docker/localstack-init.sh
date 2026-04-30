#!/bin/bash
set -e

awslocal secretsmanager create-secret \
  --name "seshat/postgres_url" \
  --secret-string "postgresql://seshat:seshat@postgres:5432/seshat" \
  --region eu-west-1 2>/dev/null || \
awslocal secretsmanager put-secret-value \
  --secret-id "seshat/postgres_url" \
  --secret-string "postgresql://seshat:seshat@postgres:5432/seshat" \
  --region eu-west-1

awslocal s3 mb s3://seshat-mvp --region eu-west-1 2>/dev/null || true

echo "LocalStack init complete"
