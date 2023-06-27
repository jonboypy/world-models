FROM rayproject/ray:2.4.0-py310-gpu

# Install dependencies
RUN pip install -U box2d-py flake8 torch torchvision pytorch-lightning h5py wandb gymnasium[box2d] moviepy cma

# Misc.
RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;6m\]World-Models-Docker🐋\[$(tput sgr0)\]:\W\\$ \[$(tput sgr0)\]"' >> /home/ray/.bashrc
ENV PYTHONPATH="$PYTHONPATH:/world-models"
RUN echo 'cat /world-models/assets/banner.txt' >> /home/ray/.bashrc
RUN chmod 777 /home/ray
ENV HOME=/home/ray
RUN chmod -R 777 /home/ray/.cache
