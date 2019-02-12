FROM python:3.6.8-stretch

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

# for AWS env
# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH=/usr/src/app/libs:${PATH}
RUN chmod +x /usr/src/app/libs/model /usr/src/app/libs/predict /usr/src/app/libs/

CMD tail -f /dev/null
