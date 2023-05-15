# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Copy chess_model.pth and chess_data.pkl files into the container
COPY chess_model.pth /app
COPY chess_data.pkl /app

# Define environment variable
#ENV NAME World #os.environ['NAME']

# Run main.py when the container launches
#CMD ["python", "main.py"]
CMD ["gunicorn", "-b", "0.0.0.0:5008", "api:app"]


# docker build -t chess .
# docker run -d -p 5008:5008 chess
# curl -X POST -d '{"phone":"9725232498"}' -H "Content-Type: application/json" http://173.249.46.230:5007/api/start_game -v
# curl -X POST -d '{"phone":"9725232498"}' -H "Content-Type: application/json" http://localhost:5008/api/start_game -v
# curl -X POST -d '{"phone":"9725232498", "move": "f2f4"}' -H "Content-Type: application/json" http://localhost:5008/api/make_move -v
