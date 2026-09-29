FROM nvcr.io/nvidia/pytorch:24.10-py3
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -m venv /opt/tau2 && /opt/tau2/bin/pip install --no-cache-dir pyyaml \
    "tau2-bench @ git+https://github.com/sierra-research/tau2-bench@v1.0.1"
ENV PATH=/opt/venv/bin:$PATH TAU2_PYTHON=/opt/tau2/bin/python PYTHONPATH=/workspace/immopd
WORKDIR /workspace/immopd
COPY . /workspace/immopd
