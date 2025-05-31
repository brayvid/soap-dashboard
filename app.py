# app.py
from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64 # For image conversion

app = Flask(__name__)

# app.py
from dotenv import load_dotenv
load_dotenv() # <<< THIS MUST BE VERY EARLY

# --- Database Connection (Adapted from Streamlit) ---
DEPLOY_ENV = os.environ.get('DEPLOY_ENV', 'DEVELOPMENT').upper()
# ... rest of your app.py

def get_env_var(var_name_prefix, key):
    # For Railway, env vars might be like DB_USER_PROD, DB_USER_LOCAL
    prod_var = f"{var_name_prefix}_{key.upper()}_PROD"
    local_var = f"{var_name_prefix}_{key.upper()}" # Or just DB_USER if not distinguishing local/dev this way
    
    if DEPLOY_ENV == 'PRODUCTION':
        return os.environ.get(prod_var) or os.environ.get(f"{var_name_prefix}_{key.upper()}")
    else: # DEVELOPMENT or other
        return os.environ.get(local_var) or os.environ.get(f"{var_name_prefix}_{key.upper()}")

# Using a simple dictionary for db_config from environment variables
# You will set these in Railway:
# DB_USERNAME_PROD, DB_PASSWORD_PROD, DB_HOST_PROD, DB_DATABASE_PROD, DB_PORT_PROD (optional)
# DB_USERNAME_LOCAL, DB_PASSWORD_LOCAL, DB_HOST_LOCAL, DB_DATABASE_LOCAL, DB_PORT_LOCAL (optional, for local dev)
# Or simpler: DB_USERNAME, DB_PASSWORD etc. if DEPLOY_ENV handles which actual DB it points to.

def get_engine():
    db_config = {}
    required_keys = ["username", "password", "host", "database"]
    prefix = "DB" # Example prefix for env vars

    for k in required_keys:
        db_config[k] = get_env_var(prefix, k)
        if db_config[k] is None:
            app.logger.error(f"Missing database configuration for: {k.upper()}")
            return None
    
    db_config["port"] = get_env_var(prefix, "port") or 5432 # Default port

    try:
        db_connection_str = f'postgresql+psycopg2://{db_config["username"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["database"]}'
        engine = create_engine(db_connection_str)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        app.logger.info("Database connection successful.")
        return engine
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        return None

engine = get_engine() # Initialize globally

def fetch_sentiment_distribution_per_politician(_engine, min_total_votes_threshold=10, sort_by_total_votes=False):
    if not _engine: return pd.DataFrame()
    approve_threshold = 0.1; disapprove_threshold = -0.1
    
    order_by_clause = "ORDER BY approve_percent DESC, ptv.total_votes DESC, p.name ASC"
    if sort_by_total_votes:
        order_by_clause = "ORDER BY ptv.total_votes DESC, approve_percent DESC, p.name ASC"

    query = text(f"""
        WITH VoteSentimentCategories AS (
            SELECT v.politician_id,
                CASE WHEN w.sentiment_score > {approve_threshold} THEN 'Approve'
                     WHEN w.sentiment_score < {disapprove_threshold} THEN 'Disapprove'
                     ELSE 'Neutral' END AS sentiment_category
            FROM votes AS v JOIN words AS w ON v.word_id = w.word_id WHERE w.sentiment_score IS NOT NULL 
        ), PoliticianSentimentCounts AS (
            SELECT politician_id, sentiment_category, COUNT(*) AS category_count
            FROM VoteSentimentCategories GROUP BY politician_id, sentiment_category
        ), PoliticianTotalScorableVotes AS (
            SELECT politician_id, COUNT(*) AS total_votes FROM VoteSentimentCategories GROUP BY politician_id
        )
        SELECT p.politician_id, p.name AS politician_name,
            COALESCE(SUM(CASE WHEN psc.sentiment_category = 'Approve' THEN psc.category_count ELSE 0 END) * 100.0 / NULLIF(ptv.total_votes, 0), 0) AS approve_percent,
            COALESCE(SUM(CASE WHEN psc.sentiment_category = 'Disapprove' THEN psc.category_count ELSE 0 END) * 100.0 / NULLIF(ptv.total_votes, 0), 0) AS disapprove_percent,
            COALESCE(SUM(CASE WHEN psc.sentiment_category = 'Neutral' THEN psc.category_count ELSE 0 END) * 100.0 / NULLIF(ptv.total_votes, 0), 0) AS neutral_percent,
            ptv.total_votes
        FROM politicians AS p JOIN PoliticianTotalScorableVotes AS ptv ON p.politician_id = ptv.politician_id
        LEFT JOIN PoliticianSentimentCounts AS psc ON p.politician_id = psc.politician_id
        WHERE ptv.total_votes >= :min_votes_threshold 
        GROUP BY p.politician_id, p.name, ptv.total_votes
        {order_by_clause}; 
    """)
    try:
        with _engine.connect() as connection:
            df = pd.read_sql(query, connection, params={'min_votes_threshold': min_total_votes_threshold})
        for col in ['approve_percent', 'disapprove_percent', 'neutral_percent', 'total_votes']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e: 
        return pd.DataFrame()

def fetch_weekly_approval_trends_for_selected_politicians(_engine, politician_ids_list):
    if not _engine or not politician_ids_list: return pd.DataFrame()
    safe_politician_ids = tuple(int(pid) for pid in politician_ids_list)
    if not safe_politician_ids: return pd.DataFrame()
    if len(safe_politician_ids) == 1: in_clause_sql = f"({safe_politician_ids[0]})"
    else: in_clause_sql = str(safe_politician_ids)
    query = text(f"""
        SELECT p.name AS politician_name, p.politician_id,           
            TO_CHAR(v.created_at, 'IYYY-IW') AS year_week,
            DATE_TRUNC('week', v.created_at)::date AS week_start_date,
            COUNT(v.vote_id) AS total_votes_in_week,
            CASE WHEN COUNT(v.vote_id) > 0 
                THEN (((SUM(w.sentiment_score) / COUNT(v.vote_id)) / 2.0) + 0.5) * 100.0 
                ELSE NULL END AS weekly_approval_rating_percent 
        FROM votes AS v JOIN words AS w ON v.word_id = w.word_id
        JOIN politicians AS p ON v.politician_id = p.politician_id 
        WHERE v.politician_id IN {in_clause_sql} 
            AND w.sentiment_score IS NOT NULL AND v.created_at IS NOT NULL
        GROUP BY p.politician_id, p.name, year_week, week_start_date 
        ORDER BY p.name ASC, week_start_date ASC;
    """)
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection) 
        if 'weekly_approval_rating_percent' in df.columns: df['weekly_approval_rating_percent'] = pd.to_numeric(df['weekly_approval_rating_percent'], errors='coerce') 
        if 'week_start_date' in df.columns: df['week_start_date'] = pd.to_datetime(df['week_start_date'], errors='coerce')
        return df
    except Exception as e: return pd.DataFrame()

# --- Plotting Helper Functions ---
def plot_stacked_horizontal_bar_to_image(df, categories, category_colors, title, xlabel, ylabel, top_n=20, decimal_places=1):
    if df.empty or not all(cat in df.columns for cat in categories): return None
    data_to_plot = df.head(top_n).copy();
    for cat in categories: data_to_plot[cat] = pd.to_numeric(data_to_plot[cat], errors='coerce').fillna(0)
    plot_df_ready = data_to_plot.set_index('politician_name')[categories] 
    # Data is assumed to be sorted by SQL (e.g. approve_percent DESC).
    # For barh, to have the first row of the original sorted df at the top, reverse it for plotting.
    plot_df_ready = plot_df_ready.iloc[::-1] 
    num_politicians = len(plot_df_ready)
    fig_height = max(6, min(15, 2 + num_politicians * 0.7)); fig_width = 12
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_df_ready.plot(kind='barh', stacked=True, color=category_colors, ax=ax, width=0.8)
    ax.set_title(title, fontsize=16, pad=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(0, 105); ax.legend(title="Sentiment Category", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    for p_index, (idx, row) in enumerate(plot_df_ready.iterrows()):
        cumulative_width = 0
        for i, val in enumerate(row):
            if val > 3: 
                 label_x_pos = cumulative_width + (val / 2)
                 ax.text(label_x_pos, p_index, f'{val:.{decimal_places}f}%', 
                         ha='center', va='center', color='white', fontsize=9, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.1", fc='black', alpha=0.3, ec='none'))
            cumulative_width += val
    plt.tight_layout(rect=[0, 0, 0.85, 1]); img_buf = BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=100); img_buf.seek(0)
    plt.close(fig); return img_buf

def plot_multiline_chart_to_image(df, x_col, y_col, group_col, title, xlabel, ylabel, color_palette="viridis", decimal_places=0):
    if df.empty or x_col not in df.columns or y_col not in df.columns or group_col not in df.columns or df[y_col].isnull().all(): return None
    unique_groups = df[group_col].nunique(); fig_height = max(6, min(12, 5 + unique_groups * 0.3)); fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if unique_groups <= 10: colors = sns.color_palette("tab10", n_colors=unique_groups)
    else: colors = sns.color_palette(color_palette, n_colors=unique_groups)
    sns.lineplot(data=df, x=x_col, y=y_col, hue=group_col, marker='o', ax=ax, linewidth=2, palette=colors)
    ax.set_title(title, fontsize=16, pad=15, weight='bold'); ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
    plt.xticks(rotation=30, ha='right', fontsize=10); plt.yticks(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7); ax.set_ylim(-5, 105)
    for line_index, politician_name_val in enumerate(df[group_col].unique()):
        politician_df = df[df[group_col] == politician_name_val]
        for index, row in politician_df.iterrows():
            if pd.notna(row[y_col]):
                ax.text(row[x_col], row[y_col] + 2, f"{row[y_col]:.{decimal_places}f}%", 
                        color=colors[line_index % len(colors)], ha="center", va="bottom", fontsize=8, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.1", fc='white', alpha=0.5, ec='none'))
    if unique_groups > 7: 
        ax.legend(title=group_col.replace('_', ' ').title(), bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    else:
        ax.legend(title=group_col.replace('_', ' ').title(), fontsize='small'); plt.tight_layout()
    img_buf = BytesIO(); fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=100); img_buf.seek(0)
    plt.close(fig); return img_buf

def plot_similarity_heatmap_to_image(similarity_matrix_df, title="Submitted Valence Similarity"): 
    if similarity_matrix_df.empty or len(similarity_matrix_df) < 2: return None # Need at least 2 for heatmap
    
    plot_df = similarity_matrix_df.copy() 
    display_n = len(plot_df) 

    for col in plot_df.columns: plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    fig_height = max(8, min(25, 3 + display_n * 0.9)) # Allow taller for more politicians
    fig_width = fig_height * 1.2 
    if display_n < 5 : fig_height = 6; fig_width = 8
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, ax=ax, 
                cbar=True, square=True, annot_kws={"size": 6 if display_n > 20 else (7 if display_n > 15 else 8)},
                cbar_kws={'label': 'Cosine Similarity of Sentiment Distribution', 'shrink': 0.7}) 
    ax.set_title(title, fontsize=16, pad=20, weight='bold')
    ax.tick_params(axis='x', rotation=70, labelsize=9) # Smaller labels for more items
    ax.tick_params(axis='y', rotation=0, labelsize=8)
    
    fig.subplots_adjust(left=0.3 if display_n > 5 else 0.25, bottom=0.3 if display_n > 5 else 0.25, right=0.98, top=0.95)
    
    img_buf = BytesIO(); 
    fig.savefig(img_buf, format="png", bbox_inches='tight', dpi=120) # Slightly higher DPI for clarity
    img_buf.seek(0)
    plt.close(fig); 
    return img_buf

def fetch_politicians_list(_engine): # Fetches all politicians, sorted by name
    if not _engine: return pd.DataFrame({'politician_id': [], 'name': []})
    query = text("SELECT politician_id, name FROM politicians ORDER BY name ASC;") # Explicitly sort by name ASC
    try:
        with _engine.connect() as connection: df = pd.read_sql(query, connection)
        return df
    except Exception as e: return pd.DataFrame({'politician_id': [], 'name': []})

# Helper to convert BytesIO image to base64 string for HTML
def get_image_as_base64(img_buf):
    if img_buf:
        img_buf.seek(0)
        return base64.b64encode(img_buf.read()).decode('utf-8')
    return None


@app.route('/', methods=['GET'])
def dashboard():
    if not engine:
        return render_template('error.html', message="CRITICAL: Database connection failed. Dashboard cannot operate.")

    # Tab handling (using URL parameter ?tab=...)
    active_tab = request.args.get('tab', 'tab1_dist') # Default to first tab

    # Data for all tabs (some might be conditional)
    politicians_list_df = fetch_politicians_list(engine)
    
    # Data specific to tabs
    tab1_data = {}
    tab2_data = {}
    tab3_data = {}

    # --- Tab 1: Approval Ratings ---
    if active_tab == 'tab1_dist':
        min_votes_tab1 = int(request.args.get('min_votes_tab1', 10)) # Get from URL param or default
        df_sentiment_dist_tab1 = fetch_sentiment_distribution_per_politician(
            engine,
            min_total_votes_threshold=min_votes_tab1,
            sort_by_total_votes=False
        )
        dist_img_base64 = None
        if not df_sentiment_dist_tab1.empty:
            sentiment_categories = ['approve_percent', 'neutral_percent', 'disapprove_percent']
            category_colors_map = {'approve_percent': 'mediumseagreen', 'neutral_percent': 'lightgrey', 'disapprove_percent': 'lightcoral'}
            plotted_categories = [cat for cat in sentiment_categories if cat in df_sentiment_dist_tab1.columns]
            category_colors = [category_colors_map[cat] for cat in plotted_categories]
            if plotted_categories:
                dist_img_buf = plot_stacked_horizontal_bar_to_image(
                    df_sentiment_dist_tab1, categories=plotted_categories, category_colors=category_colors,
                    title='Approval Rating Distribution by Politician', xlabel='Percentage of Votes (%)', ylabel='',
                    top_n=len(df_sentiment_dist_tab1), decimal_places=1
                )
                dist_img_base64 = get_image_as_base64(dist_img_buf)
        
        tab1_data = {
            'min_votes_current': min_votes_tab1,
            'df_sentiment_dist': df_sentiment_dist_tab1,
            'dist_img_base64': dist_img_base64
        }

    # --- Tab 2: Weekly Trends ---
    elif active_tab == 'tab2_trends':
        all_politician_names_tab2 = []
        if not politicians_list_df.empty:
            all_politician_names_tab2 = sorted(politicians_list_df['name'].unique().tolist())

        selected_politician_ids_str_tab2 = request.args.getlist('politician_ids_tab2')
        selected_politician_ids_tab2 = [int(pid) for pid in selected_politician_ids_str_tab2 if pid.isdigit()]

        if not selected_politician_ids_tab2 and "All" in request.args.get('politician_select_mode_tab2', '') and not politicians_list_df.empty:
            selected_politician_ids_tab2 = politicians_list_df['politician_id'].tolist()
        elif not selected_politician_ids_tab2 and not politicians_list_df.empty:
             if not politicians_list_df.empty:
                selected_politician_ids_tab2 = [politicians_list_df['politician_id'].iloc[0]]

        weekly_df_multiple_tab2 = pd.DataFrame()
        weekly_trend_img_base64_tab2 = None
        selected_politician_names_tab2 = []

        if selected_politician_ids_tab2 and not politicians_list_df.empty:
            selected_politician_names_tab2 = politicians_list_df[politicians_list_df['politician_id'].isin(selected_politician_ids_tab2)]['name'].tolist()
            weekly_df_multiple_tab2 = fetch_weekly_approval_trends_for_selected_politicians(engine, selected_politician_ids_tab2)
            if not weekly_df_multiple_tab2.empty and 'weekly_approval_rating_percent' in weekly_df_multiple_tab2.columns and weekly_df_multiple_tab2['weekly_approval_rating_percent'].notna().any():
                weekly_trend_img_buf_tab2 = plot_multiline_chart_to_image(
                    weekly_df_multiple_tab2, x_col='week_start_date', y_col='weekly_approval_rating_percent',
                    group_col='politician_name', title=f'', xlabel='Week Start Date', ylabel='Approval Rating (%)', decimal_places=0
                )
                weekly_trend_img_base64_tab2 = get_image_as_base64(weekly_trend_img_buf_tab2)
        
        # ***** START OF MODIFICATION FOR TAB 2 DATAFRAME DISPLAY *****
        df_display_ready_tab2 = pd.DataFrame() # Default to empty
        if not weekly_df_multiple_tab2.empty: # Use the fetched weekly_df_multiple_tab2
            original_df_tab2 = weekly_df_multiple_tab2 # Use the data already fetched for the plot
            cols_to_display_in_template_tab2 = ['politician_name', 'year_week', 'week_start_date', 'weekly_approval_rating_percent', 'total_votes_in_week']
            
            actual_cols_for_df_tab2 = [col for col in cols_to_display_in_template_tab2 if col in original_df_tab2.columns]
            
            if actual_cols_for_df_tab2:
                df_display_ready_tab2 = original_df_tab2[actual_cols_for_df_tab2].copy()
        # ***** END OF MODIFICATION FOR TAB 2 DATAFRAME DISPLAY *****

        tab2_data = {
            'all_politicians': politicians_list_df,
            'selected_politician_ids': selected_politician_ids_tab2,
            'selected_politician_names': selected_politician_names_tab2,
            'weekly_df': weekly_df_multiple_tab2, # Keep original for any other potential use or if plot needs it
            'weekly_trend_img_base64': weekly_trend_img_base64_tab2,
            'df_display_ready': df_display_ready_tab2 # Add the prepared DataFrame for HTML table
        }

    # --- Tab 3: Valence Similarity ---
    elif active_tab == 'tab3_similarity':
        df_all_for_selection_tab3 = fetch_sentiment_distribution_per_politician(
            engine,
            min_total_votes_threshold=1,
            sort_by_total_votes=True
        )
        available_politicians_for_similarity_tab3 = []
        if not df_all_for_selection_tab3.empty:
            available_politicians_for_similarity_tab3 = df_all_for_selection_tab3[['politician_id', 'politician_name', 'total_votes']].to_dict(orient='records')
        
        selected_politician_ids_str_tab3 = request.args.getlist('politician_ids_tab3')
        selected_politician_ids_tab3 = [int(pid) for pid in selected_politician_ids_str_tab3 if pid.isdigit()]
        
        if not selected_politician_ids_tab3 and not df_all_for_selection_tab3.empty:
            selected_politician_ids_tab3 = df_all_for_selection_tab3['politician_id'].head(min(5, len(df_all_for_selection_tab3))).tolist()
        elif "All" in request.args.get('politician_select_mode_tab3', '') and not df_all_for_selection_tab3.empty:
             selected_politician_ids_tab3 = df_all_for_selection_tab3['politician_id'].tolist()

        heatmap_img_base64_tab3 = None
        similarity_df_valence_html_tab3 = None
        df_for_similarity_calc_tab3 = pd.DataFrame()
        
        # Define the constant for max politicians for the heatmap
        # This makes it accessible throughout this block and can be passed to the template
        MAX_HEATMAP_POLITICIANS_CONST = 30 

        if selected_politician_ids_tab3 and not df_all_for_selection_tab3.empty:
            df_selected_politicians_full_info = df_all_for_selection_tab3[df_all_for_selection_tab3['politician_id'].isin(selected_politician_ids_tab3)]
            
            # Use the constant defined above
            if len(df_selected_politicians_full_info) > MAX_HEATMAP_POLITICIANS_CONST:
                df_for_similarity_calc_tab3 = df_selected_politicians_full_info.head(MAX_HEATMAP_POLITICIANS_CONST).copy()
            else:
                df_for_similarity_calc_tab3 = df_selected_politicians_full_info.copy()

            if not df_for_similarity_calc_tab3.empty and len(df_for_similarity_calc_tab3) > 1:
                politician_names_for_matrix = df_for_similarity_calc_tab3['politician_name'].tolist()
                feature_vectors = df_for_similarity_calc_tab3[['approve_percent', 'neutral_percent', 'disapprove_percent']].values

                if feature_vectors.ndim == 2 and feature_vectors.shape[0] > 1:
                    similarity_matrix = cosine_similarity(feature_vectors)
                    similarity_df = pd.DataFrame(similarity_matrix, index=politician_names_for_matrix, columns=politician_names_for_matrix)
                    heatmap_buf = plot_similarity_heatmap_to_image(similarity_df, title="Politician Valence Similarity")
                    heatmap_img_base64_tab3 = get_image_as_base64(heatmap_buf)
                    similarity_df_valence_html_tab3 = similarity_df.style.format("{:.3f}").to_html(classes=['table', 'table-sm', 'table-striped', 'table-bordered'], border=0)

        tab3_data = {
            'available_politicians': available_politicians_for_similarity_tab3,
            'selected_politician_ids': selected_politician_ids_tab3,
            'df_for_similarity_calc': df_for_similarity_calc_tab3,
            'heatmap_img_base64': heatmap_img_base64_tab3,
            'similarity_df_html': similarity_df_valence_html_tab3,
            'MAX_HEATMAP_POLITICIANS': MAX_HEATMAP_POLITICIANS_CONST # <<< THIS IS THE FIX
        }

    return render_template('index.html',
                           active_tab=active_tab,
                           tab1_data=tab1_data,
                           tab2_data=tab2_data, # This now contains 'df_display_ready'
                           tab3_data=tab3_data,
                           engine_available=bool(engine)
                           )

if __name__ == '__main__':
    # Make sure to set HOST, PORT, DB_... vars as environment variables
    # For local dev, you might use a .env file with python-dotenv
    app.run(debug=True, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 5000)))