# Installation

## Requirements

### Linux

The Lightelligence SDK (LT SDK) has been tested on both Ubuntu and CentOS. Most linux systems with Python 3.6 or higher should be able to run the LT SDK. On Linux, we provide instructions for installing the LT SDK using
[pip](#pip), [Miniconda](#miniconda), and [Docker](#docker).

### MacOS and Windows

We recommend using [Docker](https://www.docker.com/) to run the LT SDK on MacOS and Windows. See the
[Docker installation instructions](#docker) for more information.

## Installation Methods

### pip

NOTE: pip installation only works on Linux systems. For MacOS and Windows users, we recommend following
the [Docker installation instructions](#docker).

The LT SDK requires Python 3.6 or higher. Installing the LT SDK with pip will also install
other python packages that the LT SDK depends on. In order to prevent conflicts, we recommend
installing the LT SDK inside of a virtual environment. For example, we provide instructions for
installing the LT SDK inside of a [Miniconda virtual environment](#miniconda).

1. Activate your virtual environment of choice. Once inside the virtual environment,
   install Python 3.6 or higher.

2. The LT SDK is hosted on [PyPI](https://pypi.org/) and can be installed with pip using

```sh
pip install lightelligence-sdk
```

After installing the pip package, you should be ready to use the LT SDK.

### Miniconda

NOTE: Miniconda installation only works on Linux systems. For MacOS and Windows users, we recommend following the [Docker installation instructions](#docker).

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) according to the [installation instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Download the SDKExamples repo using

```sh
git clone https://github.com/Lightelligence/SDKExamples.git
```

3. From the SDKExamples repo, create a conda environment from the `environment.yml`

```sh
conda env create -f environment.yml
```

4. Activate the `lightelligence-sdk` environment

```sh
conda activate lightelligence-sdk
```

After activating the virtual environment, you should be ready to use the LT SDK.

### Docker

1. Install [Docker](https://www.docker.com/), using the [installation instructions](https://docs.docker.com/engine/install/)

2. Find the latest version number from our [Docker Repo](https://hub.docker.com/r/lightelligenceai/lightelligence-sdk/tags)

3. Pull the latest version of our image

```sh
VERSION=0.1.2 # Use latest version here
docker pull lightelligenceai/lightelligence-sdk:$VERSION
```

4. Create a container from the Docker image

```sh
docker create -it --name lt-container lightelligenceai/lightelligence-sdk:$VERSION
```

5. Start the container and attach to use an interactive shell

```sh
docker start -ia lt-container
```

Once inside the docker container, you should be ready to use the LT SDK.

Note that when running the `start` command, you can use the `-p` flag to forward a local port to a port inside the Docker container. You can also use the `-v` flag to mount a local directory inside the docker container. See the [Docker CLI documentation](https://docs.docker.com/engine/reference/commandline/cli/) for more details.
