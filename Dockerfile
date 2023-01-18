FROM rayproject/ray-ml:2.1.0-py37-gpu

RUN conda install swig -y && pip install box2d-py

RUN sudo apt update && sudo apt install xvfb -y

RUN echo 'alias python="xvfb-run python"' >> /home/ray/.bashrc

RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;6m\]World-Models-Docker🐋\[$(tput sgr0)\]:\W\\$ \[$(tput sgr0)\]"' >> /home/ray/.bashrc

ENV PYTHONPATH="$PYTHONPATH:/world-models"

RUN echo 'cat /world-models/assets/banner.txt' >> /home/ray/.bashrc

# updates:
RUN pip install flake8