docker run -d -p 5000:5000 -e PYTHONUNBUFFERED=1 --mount type=bind,source=$(pwd),target=/app easy-study
