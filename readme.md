# Carlo: A Catholic Chat bot

## How to get up and running

1. Open PowerShell or Terminal
2. Clone the repository with `git clone...`
3. Enter the repo: `cd Carlo`
4. Create a virtual env: `python3 -m venv venv`
5. Enter the virtual environment -- Windows: `venv/Scripts/activate` Mac/Linux: `source venv/bin/activate`
6. Install dependencies: `pip install -r requirements.txt`
7. Pull .env file: `npx dotenv-vault@latest pull`
7. Run Carlo: `chainlit run Carlo.py -w`