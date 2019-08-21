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

#### Start notebook
(base) C:\Users\User\workspace\getalert\libs>conda activate tensorflow
(tensorflow) c:\Users\User\workspace >jupyter notebook


#### OpenVINO
(OpenVINO) C:\Users\User\workspace\OpenVINO\object_detection>python -V
Python 3.6.9 :: Anaconda, Inc.

```
    cd C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\demo\
    demo_squeezenet_download_convert_run.bat
    demo_security_barrier_camera.bat
```

check version
```
> python
>>> import sys
>>> sys.path.append("C:\Program Files (x86)\IntelSWTools\openvino\python\python3.6")
>>> import cv2
>>> cv2.__version__
'4.1.1-openvino'
>>>
```
