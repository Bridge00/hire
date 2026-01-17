# Use a multi-language base image or start from a lightweight one
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and runtimes
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    nodejs \
    npm \
    python3 \
    python3-pip \
    golang-go \
    openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# (Optional) Pre-install some common Python libraries if needed
# RUN pip3 install numpy pandas

# Add a non-root user for security (optional but recommended)
# RUN useradd -m evaluator
# USER evaluator

CMD ["/bin/bash"]
