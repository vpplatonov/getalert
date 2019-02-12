### Build

docker build -t getalert .

### Run

Disk C: should be shared by docker in desktop settings

```bash
docker run --rm -it -v C:\Users\User\workspace\ESC-50:/opt/ml getalert
docker exec -ti XXXXXXXXXX sh -c "python dataset.py"
```

```bash
(getalert) C:\Users\User\workspace\getalert>docker ps -a
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS                      PORTS               NAMES
97e3feaac88b        getalert            "/bin/sh -c 'python …"   20 minutes ago      Exited (1) 20 minutes ago                       flamboyant_knuth
```

```bash
docker exec -ti XXXXXXXXXX sh -c "python train.py"
```

### AWS login

aws ecr get-login --no-include-email

```bash
docker login -u AWS -p eyJwYXlsb2FkIjoiQ3hUcTZvQWwvT2....Dk0MDEzMTV9 https://XXXXXXXX.dkr.ecr.eu-west-1.amazonaws.com
```

### AWS push docker image

https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/container/build_and_push.sh