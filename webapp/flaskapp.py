from flaskapp import app
from files.process_tokenize import tokenize

app.run(host='0.0.0.0', port=3000, debug=True)