FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

ARG TEMPLATE_DIR=/home/templates
ENV TEMPLATE_DIR=${TEMPLATE_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR} ${TEMPLATE_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

COPY submission.csv ${AGENT_DIR}/test/submission.csv
COPY submission.csv ${AGENT_DIR}/test/submission.csv

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${AGENT_DIR}/requirements.txt
# Copy the start script into the agent directory
COPY start.sh ${AGENT_DIR}/start.sh
RUN chmod +x ${AGENT_DIR}/start.sh 

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 python3 python3-pip python3-venv watchman libc6 libtinfo6 libgmp10 -y

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda run -n ${CONDA_ENV_NAME} pip install openai==1.51.2 httpx==0.24.0 einops==0.8.1 && \
    conda run -n ${CONDA_ENV_NAME} pip install pyglove --pre && \
    conda run -n ${CONDA_ENV_NAME} pip install wandb && \
    conda run -n ${CONDA_ENV_NAME} pip install pyre-check==0.9.18 && \
    conda run -n ${CONDA_ENV_NAME} pip install tiktoken==0.12.0 && \
    conda clean -afy

# put all the agent files in the expected location
COPY . ${AGENT_DIR}
