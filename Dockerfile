FROM pytorch/pytorch:latest
WORKDIR /usr/src/app
COPY . .
RUN apt-get update 
RUN apt-get install -y cmake g++ make ffmpeg
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
EXPOSE 8501
CMD ["streamlit","run","./webapp.py"]
