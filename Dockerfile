FROM voipmonitor/inference-vllm:25.05

ENV https_proxy=http://192.168.90.9:8899

RUN pip config set global.index-url https://mirror.nju.edu.cn/pypi/web/simple
RUN pip install lmcache
RUN pip uninstall -y flashinfer-python

ENV https_proxy=
ENTRYPOINT []
