import os
os.environ['dev'] = '1'
from dash_mess import dev_server


dev_server(debug=True)