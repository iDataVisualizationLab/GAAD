# kdd2020_phase2_sample

This is a baseline sample of how to build your model as a docker image(in our case, with python3 and torch):

1. Install docker on your server.

2. Freeze your library dependencies into a requirements.txt file. We recommend using pipreqs rather than pip freeze because pipreqs gives you only the ones actually imported by this project(e.g. pipreqs /home/project/location).
Current requirements:
torch
scipy
torchvision

3. Modify your_main.py to make sure it can succefully run by taking three positional arguments: the path of adj.pkl file, the path of feature.npy file, and the path of your output(e.g. python your_main.py /data/adj.pkl /data/feature.npy /app/output.csv).

4. Create a run.sh with one line: python3 your_main.py $1 $2 $3

5. Create a Dockerfile (note the capital D), copy the following into it:

       FROM nvidia/cuda
       RUN apt-get update
       RUN apt-get -y install python3
       RUN apt-get -y install python3-pip
       RUN pip install --upgrade pip
       COPY . /app
       WORKDIR /app
       RUN pip install -r requirements.txt
       ENTRYPOINT ["bash","run.sh"]
    
   Here's a break down of what we have achieved in our dockerfile:
   
       FROM nvidia/cuda
       
   This is by far the simpliest way of setting up your docker with a base image comes with nvidia driver and CUDA support. 
   The driver version is 440.33.01 and CUDA version is 11.0. 
   If you have a specific version of driver and/or CUDA in your project, you have to find a way to install it from scrach.
   
       RUN apt-get update
       RUN apt-get -y install python3
       RUN apt-get -y install python3-pip
       RUN pip install --upgrade pip
       
   Unfortunately, nvidia/cuda doesn't come with any programming languages attached, so we have to install it here.
   
       COPY . /app
       WORKDIR /app
       
   Here we specify where to put all our files inside a docker container, and make that the default working directory.
   
       RUN pip install -r requirements.txt
       
   And then install all the library dependencies.
   
       ENTRYPOINT ["bash","run.sh"]
       
   Finally we specify an entrypoint which will be executed once the building process is done.
   
    
6. Run "sudo docker build -t your_model_name:your_version" to build your image.

## Submit your code

From now on, you have two ways of submitting your work. You can either:

* Run "sudo docker save your_model_name:your_version | gzip > your_model_name.tar.gz" to compress your image into a tar.gz.

---or---

* Compress your repo into a much smaller zip file, and leave the building process to us. (After being tested, of course.)

Either way, you can submit your result at https://biendata.xyz/competition/kddcup_2020_formal/final-submission/, and we will decide how to proceed by checking file extensions.

Good luck coding.
