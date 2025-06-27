FROM buildpack-deps:plucky

# Set environment variables to non-interactive for apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install minimal essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl git make clang libc++-dev libc++abi-dev

# Set working directory
WORKDIR /app

# Default command (can be overridden)
CMD ["bash"]

COPY ruperts /app/ruperts
COPY cc-lib /app/cc-lib

RUN chown -R ubuntu /app
USER ubuntu
