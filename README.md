# Soap Dashboard

**View it here: [dash.soap.fyi](https://dash.soap.fyi)**

This dashboard is for the [Soap polling project](https://soap.fyi). It allows users to view overall approval ratings, track weekly approval rating trends, and analyze the similarity in sentiment profiles between different politicians.



## Features

*   **Approval Ratings Tab:**
    *   Displays a stacked horizontal bar chart showing the percentage of 'Approve', 'Neutral', and 'Disapprove' votes for various politicians.
    *   Filterable by the minimum number of scorable votes a politician has received.
    *   Expandable section to view the raw data table.
*   **Trends & Comparison Tab:**
    *   Shows weekly approval rating trends as a multi-line chart.
    *   Allows users to select one or more politicians for comparison.
    *   Option to view trends for "All Available" politicians.
    *   Expandable section to view the raw weekly trend data.
*   **Valence Similarity Tab:**
    *   Presents a heatmap showing the cosine similarity of sentiment distributions (approve/neutral/disapprove percentages) between selected politicians.
    *   Users can select multiple politicians; the heatmap displays up to 30 for readability, ordered by total submissions.
    *   Option to view similarity for "All Available" politicians.
    *   Expandable section to view the raw similarity matrix data.

## Tech Stack

*   **Backend:** Python, Flask
*   **Data Handling:** Pandas, SQLAlchemy (for PostgreSQL interaction)
*   **Plotting:** Matplotlib, Seaborn (generating images served to the frontend)
*   **Frontend:** HTML, CSS (custom styles), Jinja2 (Flask templating)
*   **Database:** PostgreSQL
*   **Deployment (Example):** Railway.app (or specify your platform)

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip (Python package installer)
*   PostgreSQL server (running locally or accessible remotely)
*   Git (for cloning the repository)

### Local Development Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/brayvid/soap-dash.git
    cd soap-dash 
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip3 install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file in the root of the project directory (`soap-dash/.env`). This file will store your database credentials and Flask settings.
    ```env
    # --- .env file for LOCAL DEVELOPMENT ---
    FLASK_APP=app.py
    FLASK_ENV=development # Enables debug mode and auto-reloader
    # DEPLOY_ENV=DEVELOPMENT # Defaults to DEVELOPMENT if not set in app.py

    # PostgreSQL Database Credentials (unsuffixed for local dev)
    DB_USERNAME=your_local_db_username
    DB_PASSWORD=your_local_db_password
    DB_HOST=localhost
    DB_DATABASE=your_local_db_name
    DB_PORT=5432 # Or your PostgreSQL port if different
    ```
    Replace `your_local_db_username`, etc., with your actual local PostgreSQL credentials.

5.  **Database Schema & Data:**
    Ensure your PostgreSQL database has the necessary tables (`politicians`, `words`, `votes`) and is populated with data.
    *(Optional: Add a note here if you have a SQL script to set up the schema, e.g., `schema.sql`)*

6.  **Run the Flask Application:**
    *   **Using Flask CLI (recommended for development):**
        ```bash
        # Ensure FLASK_APP and FLASK_ENV are set in .env or exported
        python3 -m flask run
        ```
    *   **Directly running `app.py`:**
        ```bash
        python3 app.py
        ```
    The application should now be running on `http://127.0.0.1:5000`.

### Directory Structure

```
soap-dash/
├── app.py                 # Main Flask application logic and routes
├── static/
│   ├── styles.css         # Custom CSS for styling
│   └── favicon.ico        # Application favicon
├── templates/
│   ├── base.html          # Base HTML template (navbar, footer, common structure)
│   ├── index.html         # Main multi-tab dashboard template
│   ├── error.html         # Template for critical database errors
│   └── 404.html           # Custom 404 page not found template
├── .env                   # Local environment variables (ignored by Git)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Deployment

This application is designed to be easily deployable on platforms like [Railway.app](https://railway.app/), Heroku, or any other platform that supports Python/Flask applications.

**For Railway (Example):**

1.  Push your code to a GitHub repository.
2.  Create a new project on Railway and connect it to your GitHub repository.
3.  Railway should automatically detect the `Procfile` (if you add one) or you can set the start command: `gunicorn app:app`.
4.  Configure the necessary environment variables in the Railway service settings:
    *   `DEPLOY_ENV=PRODUCTION`
    *   `FLASK_ENV=production` (optional, Gunicorn handles production typically)
    *   `DB_USERNAME_PROD`
    *   `DB_PASSWORD_PROD`
    *   `DB_HOST_PROD`
    *   `DB_DATABASE_PROD`
    *   `DB_PORT_PROD` (if applicable)
    *   (And any other API keys or secrets your application might need)

*(Optional: Add a Procfile to your repository root for PaaS platforms like Heroku/Railway)*
```Procfile
web: gunicorn app:app --log-file=-
```

---
<p align="center">&copy; Copyright 2024-2025 <a href="https://soap.fyi">soap.fyi</a>. All rights reserved.</p>
