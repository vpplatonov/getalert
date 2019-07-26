# Read fron S3 bucket
# from model.minio_s3 import main

# Save to S3 files for training XGBoost model
from libs.model.xgboost_db_save import main

if __name__ == '__main__':
    main()
