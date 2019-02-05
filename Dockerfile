FROM python:3.7.2-stretch

RUN apt-get update
RUN apt-get install -y libav-tools
RUN apt-get install -y ffmpeg

WORKDIR /usr/src/app

# add requirements (to leverage Docker cache)
ADD ./requirements.txt /usr/src/app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# add app
ADD . /usr/src/app

WORKDIR /usr/src/app/libs

CMD python prediction.py