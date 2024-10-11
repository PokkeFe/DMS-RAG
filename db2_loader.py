import sqlalchemy
import os
import dotenv

from langchain_community.utilities import SQLDatabase
import sqlalchemy.event

import base64

dotenv.load_dotenv()

if os.name == 'nt':
    os.add_dll_directory(os.getenv("CLI_DRIVER_PATH"))

with open("ca.txt", "w") as f:
    decoded = base64.standard_b64decode(os.getenv("DB2_SSL"))
    f.write(decoded.decode(encoding="utf-8"))

def get_db2_database() -> SQLDatabase:

    username = os.getenv("DB2_USERNAME")
    password = os.getenv("DB2_PASSWORD")
    hostname = os.getenv("DB2_HOSTNAME")
    port = os.getenv("DB2_PORT")
    schema = os.getenv("DB2_SCHEMA")
    db_name = os.getenv("DB2_DATABASE")

    db2_connection_string = f"ibm_db_sa://{username}:{password}@{hostname}:{port}/{db_name}?security=SSL;SSLServerCertificate=ca.txt;CurrentSchema={schema}"
    db = SQLDatabase.from_uri(db2_connection_string, schema=schema)
    return db