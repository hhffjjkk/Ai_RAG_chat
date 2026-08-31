# 导入Python内置模块
import os
import json
# 导入MinIO官方Python SDK核心类
from minio import Minio
# 项目内部配置与日志
from app.conf.minio_config import minio_config
from app.core.logger import logger

#创建minio
minio_client = Minio(
    endpoint=minio_config.endpoint,
    access_key=minio_config.access_key,
    secret_key=minio_config.secret_key,
    secure=False # 内网/本地部署用HTTP，公网部署需改为True并配置SSL
)
#创建一个桶，先判断是否存在
bucket_name=minio_config.bucket_name
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},  # *表示所有匿名用户（S3兼容标识）
            "Action": ["s3:GetObject"],  # 仅授权文件获取/访问操作
            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
        }]
    }
    minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))

def get_minio_client():
    return minio_client

