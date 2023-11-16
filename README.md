# RoadAI-project
## Project for <a href="https://www.uio.no/studier/emner/matnat/ifi/IN5490/">in5490 </a>



<p align="center">
<a href="#About"></a> •
<a href="#Installing prerequisites">Installing dependencies</a> •
<a href="#Runnning the program">How to run the program</a> •
</p>

## About
This is a MARL solution for the <a href="https://www.nora.ai/competition/roadai-competition/">ROADAI</a> competition held by NORA. The project is written for the UiO course IN5490 Advanced Topics in Artificial Intelligence for Intelligent Systems.
## Installing prerequisites


```bash
# Example using venv
$ python -m venv venv
$ source venv/bin/activate
# Install dependencies
$ pip install -r requirements.txt
```
## Running the program
```
# To start training the agents
$ python /src/main.py -t
# To render a pretrained model
$ python /src/main.py -s
# To make plots of a the previous training run
$ python /src/main.py -m
```

---




