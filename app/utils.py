
import logging
import json
import boto3
from config import Config

boto3_session = boto3.session.Session(
            aws_secret_access_key=Config.BOTO3_SECRET_KEY)
s3 = boto3_session.resource('s3')

def catch_exceptions(func):
    def wrapped_function(*args, **kargs):
        try:
            return func(*args, **kargs)
        except Exception as e:
            return None                
    return wrapped_function

def download_from_s3(source_path,destination_path):
    s3.Bucket(Config.BOTO3_BUCKET).download_file(source_path,destination_path)
