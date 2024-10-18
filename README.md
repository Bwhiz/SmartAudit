## Simple schematic Representation of the SmartAudit workflow

![Schematic](https://github.com/Bwhiz/SmartAudit/blob/main/assets/SmartAudit_schema.jpg)


## Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.8+**: [Download and install Python](https://www.python.org/downloads/)
- **Docker** (optional if running via Docker): [Install Docker](https://docs.docker.com/get-docker/)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Bwhiz/SmartAudit.git
   cd your-repo
   ```
2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install the required dependencies:
    ```bash
    # you can run the bash script 'install.sh' for easier installation
    chmod +x install.sh
    ./install.sh
    ```
## Running the Script
To execute the main.py script, follow these steps:

1. Direct Execution (after installing dependencies):
    ```bash
    python main.py
    ```
2. Using Docker:
    
    If you want to run the script in a Docker container:

    - Build the Docker image:
        ```bash
        docker build -t my_project_image .
        ```
    - Run the container:
        ```bash
        docker run my_project_image
        ```

## Scripts Overview
- **main.py:** The main script that runs the core functionality of the project.

- **src/prototype.py:** A prototype script for experimenting with the execution of the schematic diagram.

- **src/utilities/functions.py:** Contains helper functions that are used by other scripts in the project.

## Additional Notes
To successfully run this project you need to have created a collection on Milvus Vector Database cloud solution [zilli cloud](https://zilliz.com/cloud).

As seen in the `.gitignore` file there's a `.env` file that houses the api-keys and environment variables needed to run this project locally, the `.env` file structure looks like what's shown below:
```env
OPEN_API_KEY=your-openai-key
zilliUser=your-zilli-user
zilliPassword=your-zilli-password
zilliAPI_KEY=your-zilli-api-key
connectionUri=the-connection-uri-to-the-collection
```

The overall directory structure is shown below:

```bash
.
├── Dockerfile            # Docker setup file
├── README.md             # This README file
├── install.sh            # Shell script for installation
├── main.py               # Main Python script
├── notebooks/            # Jupyter notebooks for indexing
│   └── indexing.ipynb
├── requirements.txt      # Python dependencies
└── src/                  # Source folder for utility functions
    ├── prototype.py
    └── utilities/
        ├── __init__.py
        ├── functions.py  # Additional utility functions
```