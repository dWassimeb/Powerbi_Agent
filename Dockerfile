
    # Use an official Python runtime as a parent image 
    # Example : light version of Python 3.12
    FROM python:3.10
    
    # Install build dependencies
    RUN apt-get update && apt-get install -y         build-essential         libssl-dev         libffi-dev         python3-dev         gcc         && rm -rf /var/lib/apt/lists/*
    
    # Set the working directory at the root in the container to /app
    WORKDIR /app
    
    # Copy all contents of the current directory on your host machine to the /app directory of the container
    COPY . /app
    #Use this code instead if you want to copy only the contents in app directory 
    #COPY app/ /app

    # Upgrade pip
    RUN pip install --upgrade pip
    
    # Install dependencies
    RUN pip install -r requirements.txt

    # Set environment variables from build args
    ARG GPT_API_KEY

    ENV GPT_API_KEY=$GPT_API_KEY

    # Expose port
    EXPOSE 8501
    
    # Start the application
    ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    