from flask import Flask
app=Flask(__name__)

from flaskapp import routes
from files.process_tokenize import tokenize