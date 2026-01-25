# Use a multi-language base image or start from a lightweight one
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and runtimes
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    python3 \
    python3-pip \
    golang-go \
    openjdk-17-jdk-headless \
    libboost-all-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18 from NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install JavaScript dependencies
RUN npm install js-md5

# (Optional) Pre-install some common Python libraries if needed
# RUN pip3 install numpy pandas

# Add a non-root user for security (optional but recommended)
# RUN useradd -m evaluator
# USER evaluator

CMD ["/bin/bash"]
