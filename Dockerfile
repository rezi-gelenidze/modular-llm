# Use an official Python image as a base
FROM python:3.10

# Install required Python packages
RUN pip install torch transformers

# Install and cache models from hugging face


# Create directories for the source code and notebooks
WORKDIR /workspace

RUN mkdir -p /workspace/src /workspace/notebooks

COPY src/ /workspace/src

# Expose the port Jupyter will run on
EXPOSE 8888

# Command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
