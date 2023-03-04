FROM gcr.io/kaggle-images/python:v107

WORKDIR /root/lib

# Install eigen3
RUN curl -sSL https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz | tar xz
RUN cd eigen-3.3.9 && mkdir build && cd build && cmake .. && make && make install

# Install nlohmann-json
RUN curl -sSL https://github.com/nlohmann/json/archive/refs/tags/v3.9.1.tar.gz | tar xz
RUN cd json-3.9.1 && mkdir build && cd build && cmake .. && make && make install

# Install pybind11_json
RUN curl -sSL https://github.com/pybind/pybind11_json/archive/refs/tags/0.2.11.tar.gz | tar xz
COPY thirdParty/include/pybind11_json/pybind11_json.hpp pybind11_json-0.2.11/include/pybind11_json
RUN cd pybind11_json-0.2.11 && mkdir build && cd build && cmake -Dpybind11_DIR=/opt/conda/lib/python3.7/site-packages/pybind11/share/cmake/pybind11 .. && make install

# Install magic_enum
RUN mkdir -p /usr/local/include/magic_enum
RUN curl -sSL https://github.com/Neargye/magic_enum/releases/download/v0.7.3/magic_enum.hpp -o /usr/local/include/magic_enum/magic_enum.hpp

# Install nlopt
RUN curl -sSL https://github.com/stevengj/nlopt/archive/v2.6.2.tar.gz | tar xz
RUN cd nlopt-2.6.2 && cmake . && make && make install

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

WORKDIR /src

COPY . .

RUN pip install "ray[default, tune, rllib]==1.9.1"

RUN pip install --no-cache-dir .

RUN cd sample/OriginalModelSample && pip install --no-cache-dir .