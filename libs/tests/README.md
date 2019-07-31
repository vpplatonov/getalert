#### Prepare environment in cmd
##### start mongo
"C:\Program Files\MongoDB\Server\3.4\bin\mongod"

##### start minio_S3

C:\Users\User\Downloads\minio server c:\data\db

##### start TF Serving
cd c:\Users\User\source\repos\DetectionSubsystem\Engines.ModelsServing

###### Build image for first time

    docker build -t models_serving .

###### Run container

    docker run --name models_serving --rm -p 8500:8500 -p 8501:8501 -it models_serving

##### Run test
(base) C:\Users\User\workspace\getalert\libs>conda activate getalert3.5
(getalert3.5) C:\Users\User\workspace\getalert\libs>pytest --additional_value=2

```
=============  32 passed, 2 xfailed, 1 xpassed in 11.00 seconds ===================
```